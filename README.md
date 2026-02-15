# xlstm.c

[![xlstm](https://img.shields.io/badge/xlstm-NX--AI%2Fxlstm-blue)](https://github.com/NX-AI/xlstm)

Portable xLSTM kernels in C99. Implements sLSTM and mLSTM — the two custom cell types from the xLSTM paper. Includes framework adapters for deployment.

The core library is pure C99 with no dependencies beyond `math.h`. No framework, no allocator, no OS — runs anywhere a C99 compiler does. Tested against the NX-AI/xlstm PyTorch reference implementation.

## Build

```bash
make               # compile sLSTM and mLSTM kernel objects
make clean         # remove build artifacts
```

Requires a C99 compiler (`gcc`).

## Test

```bash
make test          # build and run all tests (sLSTM + mLSTM)
make reference      # regenerate test data from NX-AI/xlstm PyTorch reference
```

`make test` additionally requires `g++`. `make reference` requires Python with `torch` and `xlstm`.

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
