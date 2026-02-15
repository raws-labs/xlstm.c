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
#   src/slstm.c  src/mlstm.c
# Include paths: -Iinclude -Iadapters/tflm
```

## Tensor layout

**sLSTM** — 8 inputs, 1 output:
- Inputs: `X[B,T,I]`, `W[4H,I]`, `R[4H,H]`, `b[4H]`, `y[B,H]`, `c[B,H]`, `n[B,H]`, `m[B,H]`
- Output: `output[B,T,H]`

**mLSTM** — 7 inputs, 1 output:
- Inputs: `X[B,T,I]`, `W[4H+2,I]`, `b[4H+2]`, `y[B,H]`, `C[B,H*H]`, `n[B,H]`, `m[B,1]`
- Output: `output[B,T,H]`

State tensors are updated in-place.

## Test

```bash
make test-docker-tflm
```
