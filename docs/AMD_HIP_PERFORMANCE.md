# RX 7900 XTX support and performance evidence

This opt-in Linux `gfx1100` backend supersedes the initial support in
[PR #94](https://github.com/Niko1221/Strata/pull/94). It retains HIP runtime and
wave32 integer-dot compatibility, native mmap layout validation, and MTP, then
adds HIP MMQ, optional calibrated dense hipBLASLt GEMM, native prefill batching,
and the host-memory / SSD paths used by the measured configuration. HIP blocking
expert uploads use one reusable pinned staging buffer, including prompt-cache
refills, to avoid repeated pageable-source registrations.

## Reproduce the configuration

Build instructions are in [AMD_HIP.md](AMD_HIP.md). Use the repository-pinned
llama.cpp dependency; do not silently substitute another revision. Build with
`STRATA_PREFILL_MMQ=ON` and use these runtime variables for the measured arm:

```sh
export STRATA_PREFILL_MMQ=1
export STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1100-hipblaslt-100100.txt"
export STRATA_PF_IDX_BATCH=1
export STRATA_PF_EMBED_BATCH=1
export STRATA_PREFILL_RING=96
export STRATA_PLE_IO_THREADS=32
```

The supplied table is calibrated for gfx1100 and hipBLASLt version 100100.
It is not a universal ROCm tuning table. The runtime guards architecture,
library version and actual shape/stride/workspace requirements, falling back
when a table entry is unavailable or incompatible. Batched embedding/indexer
paths and HIP MMQ are opt-in at runtime. Default CUDA selection is preserved.

Measured engine configuration: Orca Flash Next IQ3_XXS, native pack plus matching
GGUF/tokenizer/template, `--mmap-experts --resident-cpu-experts`, fixed ranked
`--expert-profile`, `--expert-cache auto`, `--prefill 8192`, `--spec 4`,
`--spec-min-p 0.5`, matching MTP runtime, `--max-context 262144`, `--kv int8`,
`--kv-resident 32768`, `--pool-workers 15`, `--adapt-every 0`, `--pcie-frac 0`,
and `--vram-reserve-mib 1024`. Keep PLE on SSD.

`--resident-cpu-experts` copies the complement of the static GPU cache to ordinary
RAM. It requires mmap, a fixed expert profile, and disabled adaptation. It does
not lock all data into RAM: temporary prefill borrowing and cache refill may
still use the mapped fallback. Allocation needs sufficient available RAM;
on Linux this option requires readable standard cgroup-v2 mounts. The guard
checks available RAM and ancestor limits with 8 GiB headroom,
but cannot reserve memory against concurrent system or process allocations.
The POSIX PLE path issues direct reads through a configurable worker pool.

Use a dedicated idle server, restart it between arms, and capture its engine log:

```sh
python3 tools/hip/bench_prefill.py \
  --model MODEL_NAME --url http://127.0.0.1:8080 \
  --engine-log /path/to/engine.log --label candidate --output candidate.json
```

The script uses the same synthetic source prompts and request order as the
reported measurements: a small warmup, then 140/280/140/280 functions, with one
follow-up after each fresh prompt. It enforces zero reused tokens for the fresh
prompts and records prefill, decode, wall time and stop reason. It requires the
unbuffered log to contain exactly one completed timing record per request. The
128-token output cap is intentional for throughput measurement, not task success.
Never use cancelled-request timing lines as throughput evidence.

## Final revision benchmark

Measured on 2026-09-29, source revision `b90591e88ce806da6e6966d2e5d45b444d040190`.
HIP executable SHA-256: `5ca416d9864b5fddad54fb2310707675c4a10e8cd19cb0ab7d8371f4d4b057f0`.
The control and candidate below were both measured anew in this session; no
historical benchmark values are mixed into the table. The control binary SHA-256
is `4522d4937ca4a41ca294d31b60df8c2bc4df5b7a6caf65eaad10c036e08491f7`;
its source was an existing local runtime snapshot rather than a clean upstream commit.
Both arms use an RX 7900 XTX 24 GiB / gfx1100, Ryzen 9 7950X3D,
64 GiB installed RAM, Fedora-family Linux, ROCm 7.1 and a 272 W GPU cap.
Other model services are stopped for measurement and restored afterward.
Sampling: temperature 0, top-k 1, top-p 1, min-p 0, seed 42, reasoning disabled.

| Fresh request, execution order | Control prefill t/s | Candidate prefill t/s | Candidate output t/s | Candidate request wall time |
| --- | ---: | ---: | ---: | ---: |
| 4,210 tokens, first use | 240.0 | 437.6 | 54.0 | 12.000 s |
| 8,830 tokens, first use at this size | 484.8 | 863.4 | 53.1 | 12.654 s |
| 4,210 tokens, warmed | 457.2 | 878.8 | 55.1 | 7.122 s |
| 8,830 tokens, warmed | 478.9 | 968.2 | 57.5 | 11.363 s |

Each request generated exactly 128 tokens and finished at the intentional length
cap. All four fresh prompts had zero reused KV tokens. Fresh-prompt prefill was
1.78–2.02x the freshly measured existing AMD runtime; fresh-request wall time was
39–46% lower. This is a comparison with our existing HIP runtime, not unmodified
upstream, which does not provide this backend. The candidate also completed four
cached follow-ups in 3.06–3.40 seconds. Those follow-ups are not apples-to-apples
prefill comparisons: cache reuse differs with generated text and checkpoint
selection. Full sanitized measurements are in
[the fresh-run JSON](benchmarks/2026-09-29-gfx1100.json).

Before this revision, an intermediate package missing the HIP upload staging
buffer stalled on the 8,830-token request and was rejected. Restoring staging
allowed the complete sequence above to finish. That failed attempt is not
included in the throughput table. No watchdog limit was raised to obtain these
results.

Zero KV reuse is not equivalent to a cold filesystem cache. First-use speed
and warmed speed are reported separately. Individual observations do not
establish confidence intervals or a general rate at every context length.

## Correctness and quality limits

Both HIP and CUDA executables built from the measured source revision. CUDA was
compile-checked, not performance-tested by this run. The selected HIP CTest suite
passed **29/29** with the tuning table enabled, including expert-upload readback,
asynchronous handoff, QSA, MMQ, Lt GEMM, KV streaming and PLE reading. The excluded
`ple_parity` requires an external fixture; `platform_memory_test` requires a
larger locked-memory limit than the test account provides. Separately, three
real IQ3_XXS PLE matrix graph replays passed, and the POSIX direct-file test passed
queued reads, short EOF, wake, close/drain and reopen checks.

Development checks passed eight MMQ numerical comparisons, four actual Lt GEMM
comparisons, and native QSA/indexer/embedding parity including tail states,
chunk continuation and image overrides. Real quantized MMQ relative L2 error
was approximately 0.0026 against raw-FP32/dequantized-weight reference, reflecting
Q8 activation arithmetic; this is not bitwise numerical equivalence.

End-to-end coding quality is not established by these numerical checks or capped
throughput requests. Earlier coding trials did not demonstrate a successful repair,
so this contribution makes no broad answer-quality or agentic-reliability claim.

The changes do not claim better model reasoning, verified full-context behavior,
end-to-end vision validation, Windows HIP, other AMD architectures, or mixed
AMD/NVIDIA execution. Existing CUDA multi-GPU code remains present but is not
validation of HIP multi-GPU support.

## Attribution and rejected experiments

Native embedding gather and QSA batching are selectively adapted from
[PR #108](https://github.com/Niko1221/Strata/pull/108), commit
`acd487233c0bbe2217a6881c5bb43f8a283b0de5`. Its optional GDN parallel/split path is
not included. Existing upstream MMQ orchestration is retained and enabled for
HIP with AMD architecture identification and the correct backend compilation.

Static 16K chunks, expanded FP16 expert tuning, and a 192-slot ring did not offer
a consistent end-to-end win in the evaluated workload. The table includes only
the selected 30 dense-shape rows. Microkernel speedups alone were not sufficient
to select a configuration. No increase in GPU power cap was used.
