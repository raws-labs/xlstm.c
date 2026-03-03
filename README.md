# xlstm.c

Portable xLSTM kernels in C99. Implements sLSTM and mLSTM — the two custom cell types from the [xLSTM paper](https://arxiv.org/abs/2405.04517) (Hochreiter et al., 2024). No framework, no allocator, no OS — runs anywhere a C99 compiler does.

```c
#include "xlstm.h"  // single header, everything included
```

Tested against the NX-AI/xlstm PyTorch reference implementation.

## Architecture

### Kernels

| Kernel | Weights | Activations | States | m-stabilizer |
|--------|---------|-------------|--------|-------------|
| `slstm_f32` / `mlstm_f32` | float32 | float32 | float32 | float32 |
| `slstm_s8` / `mlstm_s8` | int8 | int8 | int16 | float32 |

The INT8 kernels use INT8 x INT8 -> INT32 matmul, dequantize to float for gating, and requantize states/output back to integer. The `m` stabilizer stays float32 — it prevents exponential overflow via log-space arithmetic and doesn't benefit from quantization.

### SIMD backends

Compute-intensive primitives (matvec, rank-1 update) dispatch to a SIMD backend selected at compile time:

| Backend | Target |
|---------|--------|
| `ref` | Scalar fallback (any C99) |
| `sse2` | x86/x86-64 with SSE2 |
| `neon` | ARM with NEON |
| `esp` | ESP32-S3 (Xtensa) |

Auto-detection probes the compiler's predefined macros. Override with `XLSTM_SIMD=ref|sse2|neon|esp`.

### Naming convention

| Prefix | Scope |
|--------|-------|
| `slstm_*` | sLSTM-specific (kernel, params, API) |
| `mlstm_*` | mLSTM-specific (kernel, params, API) |
| `xlstm_*` | Shared infrastructure (SIMD, quantization, utilities) |

## Build & test

```bash
make                     # compile all kernel objects (auto-detect SIMD)
make XLSTM_SIMD=ref      # force scalar backend
make test                # run all tests (f32 + INT8, sLSTM + mLSTM)
make test-ref            # test with scalar backend
make test-sse2           # test with SSE2 backend
make test-neon           # cross-compile ARM + run via QEMU
make bench               # benchmark all kernels (H = 16, 32, 64, 128)
make bench-ref           # benchmark scalar backend
make bench-sse2          # benchmark SSE2 backend
make reference           # regenerate golden data from PyTorch reference
make clean               # remove build artifacts
```

Requires `gcc` (C99) and `g++` (C++17 for tests/bench). `make reference` requires Python with `torch` and `xlstm`.

## Adapters

Thin wrappers that register custom ops in each framework. No math lives in the adapter — they unpack framework tensors and call the core C99 functions.

| Adapter | Framework | README |
|---------|-----------|--------|
| `adapters/onnxruntime/` | ONNX Runtime | [README](adapters/onnxruntime/README.md) |
| `adapters/tflm/` | TensorFlow Lite Micro | [README](adapters/tflm/README.md) |
| `adapters/microtvm/` | Apache TVM Micro | [README](adapters/microtvm/README.md) |
| `adapters/esp-dl/` | Espressif ESP-DL | [README](adapters/esp-dl/README.md) |

Docker-based integration tests run each adapter against its real framework:

```bash
make test-docker-ort       # ONNX Runtime
make test-docker-tvm       # Apache TVM
make test-docker-tflm      # TensorFlow Lite Micro
make test-docker-espdl     # ESP-DL (ESP32-S3 cross-compilation)
```

## References

- [xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517) (Hochreiter et al., 2024)
- [NX-AI/xlstm](https://github.com/NX-AI/xlstm) — PyTorch reference (Apache-2.0)
