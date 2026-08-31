# xlstm.c

sLSTM and mLSTM inference kernels in portable C99, sized for microcontrollers.
The two cell types from the [xLSTM paper](https://arxiv.org/abs/2405.04517)
(Beck et al., 2024). Freestanding: nothing outside the C standard library, and
you own every buffer.

## Use it

```c
#include "xlstm.h"

enum { I = 32, H = 64, SEQ = 128 };      /* H is the PER-HEAD width */

float W[4*H*I], R[4*H*H], b[4*H];        /* your weights   */
float x[SEQ][I];                         /* your input     */

float y[H] = {0}, c[H] = {0};            /* state, carried across steps */
float n[H] = {0}, m[H] = {0};
float scratch[4*H];                      /* you own it; nothing allocates */
SlstmParams params = {0};

for (int t = 0; t < SEQ; ++t)
    slstm_step_f32(x[t], W, R, b, y, c, n, m, scratch, I, H, &params);
```

`slstm_step_s8` is the same shape with quantized weights and scales: `c` and `n`
narrow to int16, `m` stays float. Heads are the caller's outer loop - one call
per head, over that head's slice and its own state.

## Footprint

Per head, state only, `H` = `hidden_size`:

| | sLSTM | mLSTM |
|---|---|---|
| f32 | 16H B | 4H^2 + 4H + 4 B |
| INT8 | 9H B | 2H^2 + 2H + 4 B |

mLSTM is quadratic in `H` and independent of sequence length - at H = 64 that is
16,644 B per head, flat. `tools/footprint.py 64 64 8` adds weights and scratch
and sizes a configuration against an SRAM budget.

## Backends

Selected at compile time with `XLSTM_SIMD=`, auto-detected by default.

| backend | target |
|---|---|
| `ref` | any C99 (also Cortex-M0/M23, ESP32 classic) |
| `sse2` | x86, x86-64 |
| `neon` | ARM Cortex-A |
| `cortexm` | Cortex-M4/M7/M33, ARMv7E-M and ARMv8-M DSP |
| `esp` | ESP32-S3, Xtensa LX7 PIE |
| `helium` | Cortex-M55/M85, Armv8.1-M MVE |

All six are gated per commit against the same PyTorch-derived vectors, under
emulation.

## Gate transcendentals

Both cells evaluate exp, log-sigmoid, sigmoid and tanh once per hidden unit
per timestep. On a Cortex-M4F that, and not the matmul, is where the time goes:
70% of everything outside the two INT8 matvecs is inside newlib's `expf`, `logf`
and `tanhf`.

`XLSTM_GATES=approx` (`-DXLSTM_APPROX_GATES=1`) swaps them for polynomial
approximations - portable C99, no tables, the same code on every backend. They
run in 2.1x fewer instructions than the libm they replace, and cost 1.3 to 3.0
ulp against a double-precision reference where that libm costs 0.5 to 2.1 -
except `log(sigmoid(x))`, which libm reaches through `logf(1.0f + expf(-x))` at
1.7e7 ulp, and which gets *more* accurate. On a Cortex-M4F the swap takes 28%
to 41% off the INT8 sLSTM step, most at small `H`, where the tail dominates. It
is core-dependent: the M7 gains too, the M33 comes out 1% to 3% slower.

Over the gated sizes it changes no INT8 output at all; f32 state moves by up to
32 ulp after 24 steps. It is opt-in because that is still a numerical change,
and whether it is acceptable depends on what you calibrated against. The
default build is exactly the arithmetic this library has always done; both are
gated on the same vectors, and `make test` asserts the default is bit-identical
to libm.

## Build

```sh
make            # kernels, auto-detected backend
make test       # sLSTM and mLSTM, f32 and INT8
make bench      # H = 16, 32, 64, 128
```

Needs `gcc` and `g++`. See [CONTRIBUTING.md](CONTRIBUTING.md) for the per-backend
gates, the instruction-count and mutation gates, and how to run them.

## Measured

Cortex-M7, M4F and M33, on silicon, INT8 against f32 with operands in SRAM on
both sides. Every run is in [bench/results/](bench/results/); method and limits
are in [CONTRIBUTING.md](CONTRIBUTING.md).

sLSTM in INT8 reaches 3.4x on the M7, 3.2x on the M4F and 1.4x on the M33, the
gain growing with width. Below H = 8 it gives nothing back. mLSTM halves its
state for roughly the same latency: the input projection quantizes, the state
update stays float because the log-space stabilizer needs the range. So quantize
mLSTM for footprint and sLSTM for speed.

Against CMSIS-NN's `arm_lstm_unidirectional_s8` at equal MAC count, INT8 sLSTM
built with `XLSTM_GATES=approx` takes 0.29x to 0.98x its cycles on M7 and M4F,
and 0.37x to 1.1x on M33. The default exact build is up to 1.5x slower on the
M4F.

## Adapters

Custom-op wrappers, f32 and INT8, that unpack framework tensors and call the core:
[ONNX Runtime](adapters/onnxruntime/), [TFLM](adapters/tflm/),
[MicroTVM](adapters/microtvm/), [ESP-DL](adapters/esp-dl/).

## Scope

Cells only. Pre-LN, causal conv1d, projections, GroupNorm and the residual are
caller-side, as is stacking.

`tools/` covers weight extraction from a PyTorch xLSTM, INT8 calibration, and
footprint sizing.

Apache-2.0. Reference implementation: [NX-AI/xlstm](https://github.com/NX-AI/xlstm).
