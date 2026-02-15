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
    src/slstm.c src/mlstm.c \
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

**`xlstm.slstm_eval`** — 9 DLTensor args:

`X[B,T,I]`, `W[4H,I]`, `R[4H,H]`, `b[4H]`, `y[B,H]`, `c[B,H]`, `n[B,H]`, `m[B,H]`, `output[B,T,H]`

**`xlstm.mlstm_eval`** — 8 DLTensor args:

`X[B,T,I]`, `W[4H+2,I]`, `b[4H+2]`, `y[B,H]`, `C[B,H*H]`, `n[B,H]`, `m[B,1]`, `output[B,T,H]`

States are updated in-place.

## Test

```bash
make test-docker-tvm
```
