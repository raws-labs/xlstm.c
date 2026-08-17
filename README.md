# xlstm.c

Portable xLSTM kernels in C99. Implements sLSTM and mLSTM, the two custom cell types from the [xLSTM paper](https://arxiv.org/abs/2405.04517) (Beck et al., 2024). No framework, no allocator, no OS. Runs anywhere a C99 compiler does.

```c
#include "xlstm.h"  // single header, everything included
```

Tested against the NX-AI/xlstm PyTorch reference implementation.

**`hidden_size` is the per-head width** (`DH` in the reference), not a layer width. Heads are
the caller's outer loop: run the kernel once per head over that head's slice of the weights and
its own state. This is the single most important thing to get right when adopting these
kernels, and it is why the mLSTM state cost below is quadratic *per head*.

These are the **cells**, not the xLSTM block. Pre-LayerNorm, causal conv1d, the up and down
projections, output GroupNorm and the residual are the block's, and none of them are here.
There is no multi-layer stacking and no model.

## Architecture

### Kernels

| Kernel | Weights | Activations | States | m-stabilizer |
|--------|---------|-------------|--------|-------------|
| `slstm_f32` / `mlstm_f32` | float32 | float32 | float32 | float32 |
| `slstm_s8` / `mlstm_s8` | int8 | int8 | int16 | float32 |

The INT8 kernels use INT8 x INT8 -> INT32 matmul, dequantize to float for gating, and requantize states/output back to integer. The `m` stabilizer stays float32: it prevents exponential overflow via log-space arithmetic and doesn't benefit from quantization.

### State memory, per head

sLSTM state is linear in `hidden_size`; mLSTM state is **quadratic**, because its cell state
is a `hidden_size x hidden_size` matrix per head. That difference, not the weights, is what
decides whether a given width fits an MCU.

Per head, with `H` = `hidden_size`:

| | sLSTM | mLSTM |
|---|---|---|
| f32 | `16H` bytes (`y`,`c`,`n`,`m`) | `4H^2 + 4H + 4` bytes (`C`,`n`,`m`) |
| INT8 | `9H` bytes (`y` int8, `c`/`n` int16, `m` float32) | `2H^2 + 2H + 4` bytes (`C`/`n` int16, `m` float32) |

Which works out, for mLSTM, as:

| H | f32 | INT8 |
|---|---|---|
| 16 | 1.1 KB | 0.6 KB |
| 32 | 4.1 KB | 2.1 KB |
| 64 | 16.3 KB | 8.1 KB |
| 128 | 64.5 KB | 32.3 KB |
| 256 | 257 KB | 128.5 KB |

Multiply by the number of heads: heads are the caller's outer loop, and each carries its own
state. An mLSTM layer with 8 heads at `H` = 64 needs 130 KB of f32 state before any weights,
which does not fit a 128 KB part. INT8 halves it, since `C` and `n` narrow to int16 while `m`
stays float32.

### SIMD backends

Compute-intensive primitives (matvec, rank-1 update) dispatch to a SIMD backend selected at compile time:

| Backend | Target |
|---------|--------|
| `ref` | Scalar fallback (any C99) |
| `sse2` | x86/x86-64 with SSE2 |
| `neon` | ARM with NEON |
| `esp` | ESP32-S3 (Xtensa) |
| `cortexm` | Cortex-M4/M7/M33 (ARMv7E-M / ARMv8-M DSP extension) |

Auto-detection probes the compiler's predefined macros. Override with `XLSTM_SIMD=ref|sse2|neon|esp|cortexm`.

A backend need not accelerate everything. `cortexm` implements the two matrix-vector kernels (`SXTAB16` + `SMLAD` for INT8; fused `VFMA` for f32) and defers the other two to the scalar bodies in `src/xlstm_simd_scalar.h`, which is the same text `ref` compiles rather than a copy of it.

`make test-cortexm` gates that backend without a board. `SXTAB16` and `SMLAD` are ARMv6 DSP instructions that A-profile also has, so the kernels cross-compile for `armv7-a` and run the full suite under `qemu-arm`, against the same golden vectors as every other backend. It gates the arithmetic and only the arithmetic:

- armv7-a Linux permits unaligned word access, so the INT8 matvec's `-mno-unaligned-access` load path is never compiled and M-profile alignment behaviour is not exercised at all.
- `XLSTM_FPU_HAS_MINMAX_ROUND` resolves to 0 there, as on Cortex-M4, so the FPv5 `vminnm`/`vrinta` path that M7 and M33 declare is not covered.
- Emulated execution says nothing about cycles.

Those three are checked on real parts, from a hardware-in-the-loop harness that lives in a separate repository.

### Naming convention

| Prefix | Scope |
|--------|-------|
| `slstm_*` | sLSTM-specific (kernel, params, API) |
| `mlstm_*` | mLSTM-specific (kernel, params, API) |
| `xlstm_*` | Shared infrastructure (SIMD, quantization, utilities) |

## Adopting this with your own weights

You supply the weights; there is no exporter or calibration tool here, by design.

Per head, with `H` = `hidden_size` and `I` = `input_size`: sLSTM takes `W [4H, I]`, `R [4H, H]`
and `b [4H]`; mLSTM takes `W [(4H+2), I]` and `b [(4H+2)]` and no `R`. Gate rows run `i, f, z, o`.
Call the kernel once per head, over that head's slice and its own state.

**Head slicing is not the obvious one, and getting it wrong is silent** - the reference stores
fused weights gate-major, so a head's gate blocks are strided rather than contiguous. The rule,
why the natural guess is wrong, and a self-checking worked example are in
[`test/head_slicing_example.py`](test/head_slicing_example.py), which runs on a bare Python with
no dependencies:

```
python3 test/head_slicing_example.py
```

For INT8, quantize with `xlstm_quant.h` - weights symmetric, activations asymmetric, `c` and `n`
to int16, `m` stays float32. Every case in `test/reference_data.json` carries a worked
calibration and the exact integers the kernel must produce.

If your model already runs under ONNX Runtime, TFLM, MicroTVM or ESP-DL, the adapters unpack
that framework's tensors for you - but your graph has to call this op, which a stock export will
not do on its own.

## Build & test

```bash
make                     # compile all kernel objects (auto-detect SIMD)
make XLSTM_SIMD=ref      # force scalar backend
make test                # run all tests (f32 + INT8, sLSTM + mLSTM)
make test-ref            # test with scalar backend
make test-sse2           # test with SSE2 backend
make test-neon           # cross-compile aarch64 + run via QEMU
make test-cortexm        # cross-compile armv7-a + run via QEMU (DSP path)
make bench               # benchmark all kernels (H = 16, 32, 64, 128)
make bench-ref           # benchmark scalar backend
make bench-sse2          # benchmark SSE2 backend
make perf                # instruction-count regression gate (needs valgrind)
make perf-baseline       # re-record that gate's baseline, deliberately
make reference           # regenerate golden data from PyTorch reference
make clean               # remove build artifacts
```

Requires `gcc` (C99) and `g++` (C++17 for tests/bench). `make reference` requires Python with `torch` and `xlstm`.

### Performance gate

`make bench` prints wall-clock, which no shared CI runner reproduces closely enough to fail a build on. So `make perf` counts retired instructions under `valgrind --tool=callgrind` instead, collection toggled on one kernel entry point at a time. The same binary on the same input gives the same count every run, so a 2% move against `test/perf_baseline.txt` is a real change in work done and fails CI. A deliberate change is a `make perf-baseline` and a one-line-per-case diff.

Two things it does not cover, both worth knowing before trusting a green run:

- **It is a proxy for time, not time.** A change that leaves the instruction count alone and worsens cache or memory behaviour passes. That is not hypothetical here: a change with *identical* instruction counts cost 10% on one Cortex-M part, and a smaller binary measured slower on all three.
- **It gates the host backends only** (`ref` and `sse2`). `cortexm` and `esp` performance is a property of those cores, measured on hardware, not by this gate.

Counts are also specific to the compiler that produced them, so the gate refuses to compare across a toolchain it did not record.

## Adapters

Thin wrappers that register custom ops in each framework. No math lives in the adapter; they unpack framework tensors and call the core C99 functions.

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

- [xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517) (Beck et al., 2024)
- [NX-AI/xlstm](https://github.com/NX-AI/xlstm) - PyTorch reference (Apache-2.0)
