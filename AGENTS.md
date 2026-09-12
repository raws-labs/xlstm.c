# xlstm.c

Portable C99 inference kernels for xLSTM (sLSTM and mLSTM), f32 and INT8, for
embedded and resource-constrained targets. Public, github.com/raws-labs/xlstm.c,
Apache-2.0.

## What it is and is not
- A kernel library: cells only. No xLSTM block (no pre-LN, conv1d, projections,
  GroupNorm, residual), no multi-layer stack, no export toolchain, no model runtime.
- `hidden_size` is the per-head width (DH in the reference); heads are the caller's
  outer loop. The mLSTM state is DH x DH per head.
- sLSTM carries four states: y (output), c (cell), n (normalizer), m (log-space max
  stabilizer). m stays float32 even in the INT8 kernels; quantizing it buys nothing
  and costs stability. Gate math is documented in `include/slstm.h` and `include/mlstm.h`.
- Correctness gates performance: no performance number from a kernel that has not
  passed `make test` against `test/reference_data.json`.

## Build, test, bench
- `make`: kernel objects, SIMD backend auto-detected. `make XLSTM_SIMD=<ref|sse2|neon|esp|cortexm|helium>` forces one.
- `make test`: f32 and INT8, both cells, H = 1, 2, 8, 16, 17, 64, plus the two-head
  composition contract. `make test-ref`, `make test-neon` (cross-compile, QEMU),
  `make test-cortexm`, `make test-esp`, `make test-helium`.
- `XLSTM_GATES=approx` selects approximate gate math for both precisions; opt-in
  because it is core-dependent (faster on M4F and M7, slightly slower on M33).
- `make bench`: wall-clock, ungated. `make perf`: callgrind instruction counts against
  `test/perf_baseline.txt`, gated in CI (`make perf-baseline` re-pins). `make mutants`: mutation battery.
- `make test-docker-{ort,tvm,tflm,espdl}`: adapter integration tests in Docker.
- `make reference`: regenerate golden data (needs `.venv` with torch and xlstm).
- `make check-refs`: every path and document a tracked file cites must exist in the repository; runs in CI.
- Python tooling uses the in-repo `.venv`; the Makefile hardcodes `.venv/bin/python3`.

## Layout
- `include/`: public API; `xlstm.h` is the umbrella header, `*_s8.h` the INT8 kernels,
  `xlstm_simd.h` the four-function backend contract, `xlstm_quant.h` shared quantization
- `src/`: f32 and INT8 kernels, `xlstm_simd_{ref,sse2,neon,esp,cortexm,helium}.c`,
  `xlstm_simd_scalar.h` (shared scalar bodies compiled by ref)
- `adapters/{onnxruntime,tflm,microtvm,esp-dl}/`: thin, unpack tensors and call core; no math
- `test/`: PyTorch-referenced unit tests, gate tests per backend, bench harness,
  `generate_reference.py`, `derive_multihead_layout.py`, `mutants.py`
- `bench/results/`: published measurements, one file per board and gate build
- `tools/`: INT8 calibration and per-board measurement records

## Conventions
- Prefixes: `slstm_*`, `mlstm_*`, `xlstm_*` (shared infrastructure).
- `.gitignore` excludes all dotfiles and all `*.md` except README, CONTRIBUTING and
  the PR template; do not add a tracked `*.md` without asking.
- README is for adopting the library; process and gate mechanics go to CONTRIBUTING.
- Comments state the fact itself; anything they cite must exist in the repository.

## Gotchas
- The `esp` backend's Docker suite builds and passes, but ESP-DL needs esp32s3 and
  ESP-IDF's QEMU emulates esp32 only, so those tests compile without executing.
  On-hardware esp timing uses CCOUNT cross-checked against `esp_timer`; emulated runs
  decline to report.
- Hardware-in-the-loop gates run on demand via make targets, never in CI.
- Do not assume the whole cell can be INT8; what must stay wide is measured and reported.

## References
- xLSTM paper: arxiv.org/abs/2405.04517. Reference implementation: github.com/NX-AI/xlstm.
