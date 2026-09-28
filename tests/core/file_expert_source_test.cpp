#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#if defined(STRATA_NATIVE_EXPERTS)
#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml.h"
#endif

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

struct TempDirectory {
    fs::path path;

    TempDirectory() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = fs::temp_directory_path() / ("strata-file-source-test-" + std::to_string(stamp));
        fs::create_directories(path);
    }

    ~TempDirectory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

void create_pack(const fs::path& dir, uint64_t bytes, const std::vector<std::pair<uint64_t, char>>& markers = {}) {
    const fs::path file = dir / "experts.bin";
    {
        std::ofstream out(file, std::ios::binary | std::ios::trunc);
        require((bool) out, "could not create synthetic experts.bin");
        if (bytes > 0) {
            out.seekp((std::streamoff) (bytes - 1));
            out.put('\0');
        }
        require((bool) out, "could not size synthetic experts.bin");
    }
    for (const auto& [offset, marker] : markers) {
        require(offset < bytes, "synthetic marker lies beyond experts.bin");
        std::fstream file_out(file, std::ios::binary | std::ios::in | std::ios::out);
        require((bool) file_out, "could not open synthetic experts.bin for markers");
        file_out.seekp((std::streamoff) offset);
        file_out.put(marker);
        require((bool) file_out, "could not write synthetic expert marker");
    }
}

void check_file_size_rejection(strata::core::FileExpertSource& source, const fs::path& dir, int64_t layers,
                               int64_t experts, uint64_t expected_bytes) {
    std::string err;
    source.close();
    fs::resize_file(dir / "experts.bin", expected_bytes - 1);
    require(!source.open(dir.string(), layers, experts, err), "a truncated expert file was accepted");
    require(!source.mapped() && !err.empty(), "truncated-file rejection left the source mapped or unreported");

    fs::resize_file(dir / "experts.bin", expected_bytes + 1);
    err.clear();
    require(!source.open(dir.string(), layers, experts, err), "an oversized expert file was accepted");
    require(!source.mapped() && !err.empty(), "oversized-file rejection left the source mapped or unreported");
}

void test_canonical_layout() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    constexpr int64_t layers = 2;
    constexpr int64_t experts = 3;
    const uint64_t layer_bytes = (uint64_t) experts * BLOB;
    const uint64_t total = (uint64_t) layers * layer_bytes;
    TempDirectory dir;

    std::string err;
    const bool layout_ok = expert_layout_load(dir.path.string(), layers, experts, err);
    require(layout_ok, "could not load canonical layout: " + err);
    create_pack(dir.path, total, {{0, 'a'}, {(uint64_t) BLOB, 'b'}, {layer_bytes, 'c'},
                                  {layer_bytes + (uint64_t) BLOB, 'd'}});

    FileExpertSource source;
    bool opened = source.open(dir.path.string(), layers, experts, err);
    require(opened, "could not map canonical pack: " + err);
    require(source.blobs() == layers * experts, "canonical blob count is wrong");
    const uint8_t* first = source.blob(0, 0);
    const uint8_t* second = source.blob(0, 1);
    const uint8_t* next_layer = source.blob(1, 0);
    require(first && second && next_layer, "valid canonical blob lookup failed");
    require(second - first == (ptrdiff_t) BLOB && next_layer - first == (ptrdiff_t) layer_bytes,
            "canonical expert or layer stride is wrong");
    require(first[0] == 'a' && second[0] == 'b' && next_layer[0] == 'c',
            "canonical blob lookup returned bytes from the wrong expert");
    require(source.blob(-1, 0) == nullptr && source.blob(0, -1) == nullptr &&
                source.blob(layers, 0) == nullptr && source.blob(0, experts) == nullptr,
            "canonical bounds check accepted an invalid layer or expert");
    require(source.reads() == 3, "invalid canonical lookups changed the read count");

    check_file_size_rejection(source, dir.path, layers, experts, total);
    create_pack(dir.path, total, {{layer_bytes, 'c'}});
    err.clear();
    opened = source.open(dir.path.string(), layers, experts, err);
    require(opened, "source failed to reopen after size errors: " + err);
    const uint8_t* reopened = source.blob(1, 0);
    require(reopened && source.reads() == 1 && reopened[0] == 'c', "reopened canonical source kept stale state");
    source.close();
}

#if defined(STRATA_NATIVE_EXPERTS)
void test_native_variable_layout() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    constexpr int64_t layers = 2;
    constexpr int64_t experts = 3;
    constexpr int64_t embedding = 2560;
    constexpr int64_t feed_forward = 640;
    TempDirectory dir;

    NativeFmt first_fmt, second_fmt;
    std::string err;
    const bool first_ok = native_fmt(GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, embedding, feed_forward, first_fmt, err);
    require(first_ok, "IQ3_XXS/IQ4_NL synthetic format is unavailable: " + err);
    err.clear();
    const bool second_ok = native_fmt(GGML_TYPE_IQ2_XS, GGML_TYPE_IQ4_NL, embedding, feed_forward, second_fmt, err);
    require(second_ok, "IQ2_XS/IQ4_NL synthetic format is unavailable: " + err);
    require(first_fmt.bytes != second_fmt.bytes, "chosen native formats do not exercise variable layer sizes");

    const uint64_t layer0_bytes = (uint64_t) first_fmt.bytes * experts;
    const uint64_t total = layer0_bytes + (uint64_t) second_fmt.bytes * experts;
    {
        std::ofstream metadata(dir.path / "native_experts.txt");
        require((bool) metadata, "could not create synthetic native_experts.txt");
        metadata << "0 " << GGML_TYPE_IQ3_XXS << ' ' << GGML_TYPE_IQ4_NL << " 0 " << first_fmt.bytes << '\n';
        metadata << "1 " << GGML_TYPE_IQ2_XS << ' ' << GGML_TYPE_IQ4_NL << ' ' << layer0_bytes << ' '
                 << second_fmt.bytes << '\n';
        require((bool) metadata, "could not write synthetic native_experts.txt");
    }
    create_pack(dir.path, total, {{0, 'a'}, {(uint64_t) first_fmt.bytes, 'b'},
                                  {layer0_bytes, 'c'}, {layer0_bytes + (uint64_t) second_fmt.bytes, 'd'}});

    const bool layout_ok = expert_layout_load(dir.path.string(), layers, experts, err);
    require(layout_ok, "could not load synthetic native layout: " + err);
    FileExpertSource source;
    bool opened = source.open(dir.path.string(), layers, experts, err);
    require(opened, "could not map native pack: " + err);
    require(source.blobs() == layers * experts, "native blob count is wrong");
    const uint8_t* first = source.blob(0, 0);
    const uint8_t* second = source.blob(0, 1);
    const uint8_t* next_layer = source.blob(1, 0);
    const uint8_t* next_layer_second = source.blob(1, 1);
    require(first && second && next_layer && next_layer_second, "valid native blob lookup failed");
    require(second - first == (ptrdiff_t) first_fmt.bytes && next_layer - first == (ptrdiff_t) layer0_bytes &&
                next_layer_second - next_layer == (ptrdiff_t) second_fmt.bytes,
            "native per-layer expert stride is wrong");
    require(first[0] == 'a' && second[0] == 'b' && next_layer[0] == 'c' && next_layer_second[0] == 'd',
            "native blob lookup returned bytes from the wrong expert");
    require(source.blob(-1, 0) == nullptr && source.blob(0, -1) == nullptr &&
                source.blob(layers, 0) == nullptr && source.blob(0, experts) == nullptr,
            "native bounds check accepted an invalid layer or expert");
    require(source.reads() == 4, "invalid native lookups changed the read count");

    check_file_size_rejection(source, dir.path, layers, experts, total);
    create_pack(dir.path, total, {{layer0_bytes, 'c'}});
    err.clear();
    opened = source.open(dir.path.string(), layers, experts, err);
    require(opened, "source failed to reopen after size errors: " + err);
    const uint8_t* reopened = source.blob(1, 0);
    require(reopened && source.reads() == 1 && reopened[0] == 'c', "reopened native source kept stale state");
    source.close();
}
#endif

}  // namespace

int main() {
    try {
        test_canonical_layout();
#if defined(STRATA_NATIVE_EXPERTS)
        test_native_variable_layout();
#endif
        std::cout << "file_expert_source_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "file_expert_source_test: " << error.what() << '\n';
        return 1;
    }
}
