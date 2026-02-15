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
    src/slstm.c src/mlstm.c \
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

## Test

```bash
make test-docker-ort
```
