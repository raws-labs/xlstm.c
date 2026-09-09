# xlstm.c

sLSTM and mLSTM inference kernels in portable C99. These are the two recurrent
cell types from the [xLSTM paper](https://arxiv.org/abs/2405.04517) (Beck et
al., 2024), written as freestanding code: nothing outside the C standard
library, no allocator, no operating system, and every buffer belongs to the
caller.

Every available xLSTM implementation is shaped around training: PyTorch, CUDA,
JAX. That is the right shape for research and the wrong one for a part with a
few hundred kilobytes of RAM and no Python on it. Running one of these models
outside a framework has meant writing the cells yourself.

The recurrence is what makes that worth doing. Both cells carry a fixed amount
of state between timesteps, and that state is independent of sequence length.
The memory a deployment needs can therefore be worked out before it ships,
rather than bounded by a worst case and hoped for.

It is built for embedded targets, but nothing in it is embedded-specific. The
same source compiles on a workstation, and there is an x86 backend because that
is where most people will first try it.

## Usage

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

`slstm_step_s8` has the same shape, with quantized weights and their scales;
`c` and `n` narrow to int16 while `m` stays float, because the log-space
stabilizer needs the range.

Three things are worth knowing before the first call.

Nothing allocates. The scratch buffer is yours, the state arrays are yours, and
the kernels touch neither the heap nor any global.

`H` is the per-head width, `DH` in the reference implementation, not the fused
width across all heads. Getting this wrong still runs, and produces plausible
numbers from a different model.

Heads are the caller's outer loop: one call per head, over that head's slice of
the weights and its own state. The reference stores fused weights gate-major,
so a head's rows are strided rather than contiguous, and the natural guess about
the layout selects the wrong rows without complaining.
`test/head_slicing_example.py` is a runnable worked example.

## Memory and precision

Two choices decide what a configuration needs: how wide the heads are, and what
precision they run at.

Memory per head, state only:

| | sLSTM | mLSTM |
|---|---|---|
| f32 | 16H B | 4H^2 + 4H + 4 B |
| INT8 | 9H B | 2H^2 + 2H + 4 B |

sLSTM is linear in `H` and stays cheap at any width worth deploying. mLSTM is
quadratic, because its state is a matrix rather than a vector, and past roughly
H = 32 that term dominates everything else in the budget. At H = 64 it is
16,644 B per head, and it stays that size no matter how long the sequence runs.

mLSTM's two widths need not be equal. `mlstm_step_f32` takes `qk_size` and
`v_size` separately, C is `[qk_size x v_size]` and n is `[qk_size]`, so the
table above is the case where they match. Pass `--dv` to size one where they do
not. `tools/footprint.py 64 64 8` adds weights and scratch to this and sizes a
whole configuration against an SRAM budget, which is usually the form the
question actually takes.

Precision is the other lever, and it does not pull evenly on the two cells.
Quantizing sLSTM buys speed, several times over on the wider heads. Quantizing
mLSTM halves its state but leaves latency roughly where it was, because only the
input projection quantizes: the matrix update stays in float. So the useful rule
is to quantize mLSTM when memory is the constraint and sLSTM when time is.

There is a third lever if you want it. Both cells evaluate exp, log-sigmoid,
sigmoid and tanh once per hidden unit per timestep, and on a Cortex-M4F that,
rather than the matrix arithmetic, is where the time goes. `XLSTM_GATES=approx`
swaps those four for polynomial approximations in portable C99, which takes
roughly a third off the INT8 sLSTM step. It is opt-in because it is a real
numerical change, and it is worth measuring on your own target rather than
assuming: the M7 gains as well, the M33 comes out slightly slower.
[CONTRIBUTING.md](CONTRIBUTING.md) has the accuracy bounds and the per-core
figures.

## Backends

One backend is selected at compile time with `XLSTM_SIMD=`, auto-detected if you
do not pick.

| backend | target |
|---|---|
| `ref` | any C99, including Cortex-M0/M23 and ESP32 classic |
| `sse2` | x86, x86-64 |
| `neon` | Arm Cortex-A |
| `cortexm` | Cortex-M4/M7/M33, ARMv7E-M and ARMv8-M DSP |
| `esp` | ESP32-S3, Xtensa LX7 PIE |
| `helium` | Cortex-M55/M85, Armv8.1-M MVE |

`ref` is not a fallback in any apologetic sense. It is the correct choice on
parts with no vector or DSP extension, and it is the implementation every
accelerated backend is checked against.

## Build

```sh
make            # kernels, auto-detected backend
make test       # sLSTM and mLSTM, f32 and INT8
make bench      # H = 16, 32, 64, 128
```

Needs `gcc` and `g++`. [CONTRIBUTING.md](CONTRIBUTING.md) covers the per-backend
checks and how to run them.

## Scope

The scope of this kernel library is the two cells and nothing around them. An
xLSTM block is more than its cell: pre-normalization, the causal conv1d, the up
and down projections, GroupNorm and the residual all sit outside these
functions, and stacking blocks into a model is yours as well. That is a chosen
boundary rather than an unfinished edge. The cells are the part that repays
being hand-written; the rest is ordinary code that a framework, or fifty lines
of your own, will do just as well.

`tools/` covers the parts of getting a trained model in here that are easy to
get wrong: extracting per-head weights from a PyTorch xLSTM, INT8 calibration,
and footprint sizing. Each one checks itself against the reference data rather
than asking you to trust it.

## Adapters

Custom-op wrappers, f32 and INT8, that unpack framework tensors and call the
core: [ONNX Runtime](adapters/onnxruntime/), [TFLM](adapters/tflm/),
[MicroTVM](adapters/microtvm/), [ESP-DL](adapters/esp-dl/).

## Validation

Every backend is checked on each commit against vectors derived from the PyTorch
reference, at H = 1, 2, 8, 16, 17 and 64, in both precisions, under emulation
where the instruction set is not the host's. `cortexm` is additionally verified
on Cortex-M7, M4F and M33 silicon.

Those boards are where the performance figures come from too. INT8 sLSTM runs
several times faster than f32 on the wider heads, and built with
`XLSTM_GATES=approx` it matches or beats CMSIS-NN's INT8 LSTM on the M7 and M4F
at equal multiply-accumulate count. Raw runs are in
[bench/results/](bench/results/), one file per board, and
[CONTRIBUTING.md](CONTRIBUTING.md) describes the method and its limits.

Reference implementation: [NX-AI/xlstm](https://github.com/NX-AI/xlstm).
