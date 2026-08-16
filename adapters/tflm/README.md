# TensorFlow Lite Micro adapter

Register the custom ops with a `MicroMutableOpResolver`, then run via `MicroInterpreter` as usual.

## Usage

```cpp
#include "slstm_tflm.h"
#include "mlstm_tflm.h"

tflite::MicroMutableOpResolver<2> resolver;
resolver.AddCustom("SLSTM", tflite::Register_SLSTM());
resolver.AddCustom("MLSTM", tflite::Register_MLSTM());

tflite::MicroInterpreter interpreter(model, resolver, arena, kArenaSize);
interpreter.AllocateTensors();
// fill input tensors ...
interpreter.Invoke();
```

## Build

Compile the adapter alongside your TFLM project:

```bash
# Add to your TFLM build:
#   adapters/tflm/slstm_tflm.cc
#   adapters/tflm/mlstm_tflm.cc
#   src/slstm.c  src/mlstm.c  src/xlstm_simd_ref.c
#   src/slstm_s8.c  src/mlstm_s8.c  src/xlstm_quant.c   # INT8
# Include paths: -Iinclude -Iadapters/tflm
```

## Tensor layout

**sLSTM** - 8 inputs, 1 output:
- Inputs: `X[B,T,I]`, `W[4H,I]`, `R[4H,H]`, `b[4H]`, `y[B,H]`, `c[B,H]`, `n[B,H]`, `m[B,H]`
- Output: `output[B,T,H]`

**mLSTM** - 7 inputs, 1 output:
- Inputs: `X[B,T,I]`, `W[4H+2,I]`, `b[4H+2]`, `y[B,H]`, `C[B,H*H]`, `n[B,H]`, `m[B,1]`
- Output: `output[B,T,H]`

State tensors are updated in-place.

## INT8

Both ops dispatch on the input tensor's type: `kTfLiteFloat32` runs the f32
kernel, `kTfLiteInt8` runs the quantized one. Same op name, same tensor
indices - only the types change:

| Tensor | INT8 model type |
|---|---|
| `X`, `W`, `R`, `y`, `output` | `INT8` |
| `b` | `INT32`, scale = input scale x weight scale |
| `c` / `C`, `n` | `INT16` |
| `m` | `FLOAT32` - the log-space stabilizer is never quantized |

Scale and zero-point are read from each tensor's own quantization
parameters, so a normally quantized `.tflite` needs no extra inputs.
Weights must be symmetric (zero-point 0); `output` must carry the same
quantization as `y`.

Measured on Cortex-M7, M4F and M33, INT8 is **not** a latency win for
mLSTM: 0.64x to 0.98x, at or below parity. Only the input projection is
quantized; the DH x DH state update stays f32, and it dominates. For
sLSTM the speedup is 1.14x to 2.59x, growing with hidden size. For mLSTM
choose INT8 for memory footprint, not speed.

## Test

```bash
make test-docker-tflm
```
