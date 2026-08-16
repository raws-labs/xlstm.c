# ONNX Runtime adapter

Build a shared library and load it as a custom ops library in your ORT session.

## Build

```bash
ORT_DIR=$(python3 -c "import onnxruntime; import os; print(os.path.dirname(onnxruntime.__file__))")

g++ -std=c++17 -shared -fPIC -O2 \
    -Iinclude -Iadapters/onnxruntime -I"$ORT_DIR/capi" \
    adapters/onnxruntime/slstm_ort.cc \
    adapters/onnxruntime/mlstm_ort.cc \
    adapters/onnxruntime/xlstm_ort_register.cc \
    src/slstm.c src/mlstm.c src/xlstm_simd_ref.c \
    src/slstm_s8.c src/mlstm_s8.c src/xlstm_quant.c \
    -lm -o libxlstm_ort.so
```

## Usage

```python
import onnxruntime as ort

opts = ort.SessionOptions()
opts.register_custom_ops_library("libxlstm_ort.so")
sess = ort.InferenceSession("model.onnx", opts)
```

## Custom ops

Registered under domain `com.raws.xlstm`:

**`SLSTM`**
- Inputs: `X[B,T,I]`, `W[4H,I]`, `R[4H,H]`, `b[4H]`, `y_init[B,H]`, `c_init[B,H]`, `n_init[B,H]`, `m_init[B,H]`
- Outputs: `output[B,T,H]`, `y[B,H]`, `c[B,H]`, `n[B,H]`, `m[B,H]`

**`MLSTM`**
- Inputs: `X[B,T,I]`, `W[4H+2,I]`, `b[4H+2]`, `y_init[B,H]`, `C_init[B,H*H]`, `n_init[B,H]`, `m_init[B,1]`
- Outputs: `output[B,T,H]`, `y[B,H]`, `C[B,H*H]`, `n[B,H]`, `m[B,1]`

## INT8 ops

`SLSTM_S8` and `MLSTM_S8` take the same tensors quantized, with scale and
zero-point as scalar tensor inputs beside the tensor they describe - the
convention ONNX's own `QLinearConv` / `QLinearMatMul` use, so a calibration
pass can feed them without rebuilding the graph.

**`SLSTM_S8`**
- Inputs: `X`(int8), `x_scale`, `x_zero_point`, `W`(int8), `W_scale`,
  `R`(int8), `R_scale`, `b`(int32), `y_init`(int8), `y_scale`,
  `y_zero_point`, `c_init`(int16), `c_scale`, `n_init`(int16), `n_scale`,
  `m_init`(float)
- Outputs: `output`(int8), `y`(int8), `c`(int16), `n`(int16), `m`(float)

**`MLSTM_S8`** is the same without `R` / `R_scale`, with `C_init`(int16) and
`C_scale` in place of `c_init` / `c_scale`.

Weights are symmetric, so `W` and `R` take a scale and no zero-point. `b`
is INT32 at `x_scale * W_scale`. `c`/`C` and `n` are INT16 and symmetric.
`m` stays float32 - the log-space stabilizer is not quantized. `output`
carries `y`'s scale and zero-point.

Measured on Cortex-M7, M4F and M33, INT8 is **not** a latency win for
mLSTM: 0.64x to 0.98x, at or below parity. Only the input projection is
quantized; the DH x DH state update stays f32, and it dominates. For
sLSTM the speedup is 1.14x to 2.59x, growing with hidden size. For mLSTM
choose INT8 for memory footprint, not speed.

## Test

```bash
make test-docker-ort
```
