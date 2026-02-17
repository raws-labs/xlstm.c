# xlstm.c

[![xlstm](https://img.shields.io/badge/xlstm-NX--AI%2Fxlstm-blue)](https://github.com/NX-AI/xlstm)

Portable xLSTM kernels in C99. Implements sLSTM and mLSTM — the two custom cell types from the xLSTM paper. Includes framework adapters for deployment.

The core library is pure C99 with no dependencies beyond `math.h`. No framework, no allocator, no OS — runs anywhere a C99 compiler does. Tested against the NX-AI/xlstm PyTorch reference implementation.

Includes INT8 quantized variants of both kernels (`slstm_q8`, `mlstm_q8`) with INT8 weights/activations, INT16 states, and float gating — giving integer storage and bandwidth savings while preserving numerical stability through the log-space m-stabilizer.

## Build

```bash
make               # compile all kernel objects (f32 + INT8)
make clean         # remove build artifacts
```

Requires a C99 compiler (`gcc`).

## Test

```bash
make test          # build and run all tests (f32 + INT8, sLSTM + mLSTM)
make reference      # regenerate test data from NX-AI/xlstm PyTorch reference
```

`make test` additionally requires `g++`. `make reference` requires Python with `torch` and `xlstm`.

### Kernels

| Kernel | Weights | Activations | States | m-stabilizer |
|--------|---------|-------------|--------|-------------|
| `slstm_f32` / `mlstm_f32` | float32 | float32 | float32 | float32 |
| `slstm_q8` / `mlstm_q8` | int8 | int8 | int16 | float32 |

The INT8 kernels use INT8x INT8 → INT32 matmul (SIMD-ready), dequantize to float for gating, and requantize states/output back to integer. The `m` state stays float32.

## Adapters

Each adapter registers custom ops that unpack framework-specific tensor formats and forward to the core C99 functions. No math lives in the adapter. See each adapter's README for build and usage instructions.

| Adapter | Framework | README |
|---------|-----------|--------|
| `adapters/onnxruntime/` | ONNX Runtime | [README](adapters/onnxruntime/README.md) |
| `adapters/tflm/` | TensorFlow Lite Micro | [README](adapters/tflm/README.md) |
| `adapters/microtvm/` | Apache TVM Micro | [README](adapters/microtvm/README.md) |
| `adapters/esp-dl/` | Espressif ESP-DL | [README](adapters/esp-dl/README.md) |

## Integration tests

Docker-based tests run each adapter against its real framework:

```bash
make test-docker-ort       # ONNX Runtime
make test-docker-tvm       # Apache TVM
make test-docker-tflm      # TensorFlow Lite Micro
make test-docker-espdl     # ESP-DL (QEMU)
```
