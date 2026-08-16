# Apache TVM (microTVM) adapter

The adapter exports `TVMBackendPackedCFunc` functions that unpack DLTensor arguments. Link them into your TVM module or use the registration wrapper for Python access.

## Build

```bash
g++ -std=c++17 -shared -fPIC -O2 \
    -Iinclude -Iadapters/microtvm \
    -I$TVM_HOME/include \
    -I$TVM_HOME/3rdparty/dlpack/include \
    -I$TVM_HOME/3rdparty/dmlc-core/include \
    test/adapters/microtvm/tvm_register_wrapper.cc \
    adapters/microtvm/slstm_tvm.c \
    adapters/microtvm/mlstm_tvm.c \
    src/slstm.c src/mlstm.c src/xlstm_simd_ref.c \
    src/slstm_s8.c src/mlstm_s8.c src/xlstm_quant.c \
    -lm -o libxlstm_tvm.so
```

> **Note**: Do not link against `-ltvm_runtime`. TVM symbols are resolved
> at load time from the runtime already present in the host process.
> Linking a second copy causes duplicate-registration crashes.

## Usage

```python
import ctypes
import tvm

ctypes.CDLL("./libxlstm_tvm.so", ctypes.RTLD_GLOBAL)

f = tvm.get_global_func("xlstm.slstm_eval")
f(x, W, R, b, y, c, n, m, output)  # all tvm.nd.array
```

## Packed function signatures

**`xlstm.slstm_eval`** - 9 DLTensor args:

`X[B,T,I]`, `W[4H,I]`, `R[4H,H]`, `b[4H]`, `y[B,H]`, `c[B,H]`, `n[B,H]`, `m[B,H]`, `output[B,T,H]`

**`xlstm.mlstm_eval`** - 8 DLTensor args:

`X[B,T,I]`, `W[4H+2,I]`, `b[4H+2]`, `y[B,H]`, `C[B,H*H]`, `n[B,H]`, `m[B,1]`, `output[B,T,H]`

States are updated in-place.

## INT8

Same two function names. The adapter dispatches on the input DLTensor's
dtype: `kDLFloat` runs the f32 kernel, `kDLInt` the quantized one, with
`X`/`W`/`R`/`y`/`output` int8, `b` int32, `c`/`C`/`n` int16 and `m`
float32. Quantization follows the tensors as plain scalar args:

**`xlstm.slstm_eval`** - 9 DLTensors, then

`x_scale`, `x_zero_point`, `W_scale`, `R_scale`, `y_scale`, `y_zero_point`, `c_scale`, `n_scale`

**`xlstm.mlstm_eval`** - 8 DLTensors, then

`x_scale`, `x_zero_point`, `W_scale`, `y_scale`, `y_zero_point`, `C_scale`, `n_scale`

Weights and the INT16 states are symmetric, so they take a scale with no
zero-point. `b` is INT32 at `x_scale * W_scale`. `m` stays float32 - the
log-space stabilizer is not quantized. `output` shares `y`'s quantization.

Measured on Cortex-M7, M4F and M33, INT8 is **not** a latency win for
mLSTM: 0.64x to 0.98x, at or below parity. Only the input projection is
quantized; the DH x DH state update stays f32, and it dominates. For
sLSTM the speedup is 1.14x to 2.59x, growing with hidden size. For mLSTM
choose INT8 for memory footprint, not speed.

## Test

```bash
make test-docker-tvm
```
