# xlstm.c

Portable xLSTM kernels in C99. Implements sLSTM and mLSTM, the two custom cell types from the [xLSTM paper](https://arxiv.org/abs/2405.04517) (Beck et al., 2024). No framework, no allocator, no OS. Runs anywhere a C99 compiler does.

```c
#include "xlstm.h"  // single header, everything included
```

Tested against the NX-AI/xlstm PyTorch reference implementation.

Two things to know before reading further:

- **`hidden_size` is the per-head width** (`DH` in the reference), not a layer width. Heads are the caller's outer loop: run the kernel once per head, over that head's slice of the weights and its own state.
- **These are the cells, not the xLSTM block.** Pre-LayerNorm, causal conv1d, the up and down projections, output GroupNorm and the residual all belong to the block, and none of them are here. There is no multi-layer stacking and no model.

## Kernels

| Kernel | Weights | Activations | States | m-stabilizer |
|--------|---------|-------------|--------|-------------|
| `slstm_f32` / `mlstm_f32` | float32 | float32 | float32 | float32 |
| `slstm_s8` / `mlstm_s8` | int8 | int8 | int16 | float32 |

The INT8 kernels use INT8 x INT8 -> INT32 matmul, dequantize to float for gating, and requantize states and output back to integer. The `m` stabilizer stays float32: it prevents exponential overflow via log-space arithmetic and gains nothing from quantization.

## State memory

sLSTM state is linear in `hidden_size`. **mLSTM state is quadratic**, because its cell state is a `hidden_size x hidden_size` matrix per head. That, not the weights, is usually what decides whether a width fits an MCU.

Per head, with `H` = `hidden_size`:

| | sLSTM | mLSTM |
|---|---|---|
| f32 | `16H` bytes | `4H^2 + 4H + 4` bytes |
| INT8 | `9H` bytes | `2H^2 + 2H + 4` bytes |

For mLSTM that is 16.3 KB at `H` = 64 and 64.5 KB at `H` = 128 in f32, halved in INT8. Multiply by the head count: an 8-head mLSTM layer at `H` = 64 needs 130 KB of f32 state before any weights, which does not fit a 128 KB part. `python3 tools/footprint.py 64 32 8` prints state and weight bytes for a configuration, both cells and both precisions.

## SIMD backends

Compute-intensive primitives (matvec, rank-1 update) dispatch to a backend selected at compile time:

| Backend | Target |
|---------|--------|
| `ref` | Scalar fallback (any C99) |
| `sse2` | x86/x86-64 with SSE2 |
| `neon` | ARM with NEON |
| `esp` | ESP32-S3 (Xtensa) |
| `cortexm` | Cortex-M4/M7/M33 (ARMv7E-M / ARMv8-M DSP extension) |

Auto-detection probes the compiler's predefined macros; override with `XLSTM_SIMD=ref|sse2|neon|esp|cortexm`.

A backend need not accelerate everything. `cortexm` implements the two matrix-vector kernels (`SXTAB16` + `SMLAD` for INT8, fused `VFMA` for f32) and defers the rest to `src/xlstm_simd_scalar.h`, which is the same text `ref` compiles rather than a copy of it.

## Using your own weights

You supply the weights. There is no exporter, no weight format and no compatibility promise, by design - but the three steps that cost a first afternoon are worked examples in `tools/`, each of which re-derives its claim from `test/reference_data.json` and fails the build if it drifts:

```bash
python3 tools/extract_heads.py       # fused reference weights -> per-head W, R, b
python3 tools/calibrate_int8.py      # float tensors -> the INT8 scales and zero points
python3 tools/footprint.py 64 32 8   # bytes of state and weights, f32 and INT8
```

Standard library only, apart from one function that reads tensors out of a live PyTorch module. Copy them into your own pipeline; they are examples, not an interface.

Per head, with `H` = `hidden_size` and `I` = `input_size`: sLSTM takes `W [4H, I]`, `R [4H, H]` and `b [4H]`; mLSTM takes `W [(4H+2), I]` and `b [(4H+2)]`, and no `R`. Gate rows run `i, f, z, o`.

**Head slicing is not the obvious one, and getting it wrong is silent.** The reference stores fused weights gate-major, so one head's gate blocks are strided across the matrix rather than contiguous. The rule, why the natural guess is wrong, and a self-checking worked example live in [`test/head_slicing_example.py`](test/head_slicing_example.py) - standard library only:

```bash
python3 test/head_slicing_example.py
```

For INT8, quantize with `xlstm_quant.h`: weights symmetric, activations asymmetric, `c` and `n` to int16, `m` stays float32. Every case in `test/reference_data.json` carries a worked calibration and the exact integers the kernel must produce.

## Build and test

```bash
make                     # compile all kernel objects (auto-detect SIMD)
make XLSTM_SIMD=ref      # force a backend
make test                # all tests (f32 + INT8, sLSTM + mLSTM)
make bench               # benchmark (H = 16, 32, 64, 128)
make clean
```

Requires `gcc` (C99) and `g++` (C++17, for tests and bench).

Each backend has its own gate, including the cross-compiled ones: `make test-ref`, `make test-sse2`, `make test-neon`, `make test-cortexm` and `make test-esp` run the same golden vectors, the last three under emulation with no hardware. See [CONTRIBUTING.md](CONTRIBUTING.md) for those and for the regression gates.

## Adapters

Thin wrappers that register custom ops in each framework. No math lives in an adapter; they unpack framework tensors and call the core C99 functions. Both f32 and INT8.

| Adapter | Framework | README |
|---------|-----------|--------|
| `adapters/onnxruntime/` | ONNX Runtime | [README](adapters/onnxruntime/README.md) |
| `adapters/tflm/` | TensorFlow Lite Micro | [README](adapters/tflm/README.md) |
| `adapters/microtvm/` | Apache TVM Micro | [README](adapters/microtvm/README.md) |
| `adapters/esp-dl/` | Espressif ESP-DL | [README](adapters/esp-dl/README.md) |

If your model already runs under one of these, the adapter unpacks that framework's tensors for you - but your graph has to call this op, which a stock export will not do on its own.

## References

- [xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517) (Beck et al., 2024)
- [NX-AI/xlstm](https://github.com/NX-AI/xlstm) - PyTorch reference (Apache-2.0)
