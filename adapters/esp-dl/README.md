# Espressif ESP-DL adapter

Provides `dl::module::SLSTM` and `dl::module::MLSTM` classes that inherit from ESP-DL's `Module` base. States are owned by the module and persist across calls.

## Usage

```cpp
#include "slstm_espdl.hpp"
#include "mlstm_espdl.hpp"

// Instantiate with hidden and input dimensions
auto* slstm = new dl::module::SLSTM("slstm_0", /*hidden=*/64, /*input=*/32);
auto* mlstm = new dl::module::MLSTM("mlstm_0", /*hidden=*/64, /*input=*/32);

// In your model graph:
slstm->forward(context);  // reads inputs[0..3], writes outputs[0]
mlstm->forward(context);  // reads inputs[0..2], writes outputs[0]
```

## INT8

`forward()` branches on the module's `quant_type`, as ESP-DL's own modules
do. `QUANT_TYPE_SYMM_8BIT` runs the quantized kernel over int8 `X`/`W`/`R`/
`output` and int32 `b`; anything else runs the f32 one.

```cpp
// c/C and n are INT16 state buffers the module owns rather than tensors it
// is handed, so their exponents are given here. Everything else comes off
// the tensors.
auto* slstm = new dl::module::SLSTM("slstm_0", 64, 32,
                                    dl::MODULE_NON_INPLACE,
                                    dl::QUANT_TYPE_SYMM_8BIT,
                                    /*c_exponent=*/-10, /*n_exponent=*/-10);
```

Scales come from each tensor's `exponent` (`DL_SCALE(exponent)` is
`2^exponent`). ESP-DL quantization is symmetric power-of-two with no
zero-point, so the kernel is handed zero-point 0 throughout - it cannot
express an asymmetric input quantization. `m` stays float32: the log-space
stabilizer is not quantized.

Measured on Cortex-M7, M4F and M33, INT8 is **not** a latency win for
mLSTM: 0.64x to 0.98x, at or below parity. Only the input projection is
quantized; the DH x DH state update stays f32, and it dominates. For
sLSTM the speedup is 1.14x to 2.59x, growing with hidden size. For mLSTM
choose INT8 for memory footprint, not speed.

## Build

Add to your ESP-IDF component's `CMakeLists.txt`:

```cmake
idf_component_register(
    SRCS "slstm_espdl.cpp" "mlstm_espdl.cpp" "slstm.c" "mlstm.c" "xlstm_simd_esp.c"
         "slstm_s8.c" "mlstm_s8.c" "xlstm_quant.c"
    INCLUDE_DIRS "include" "adapters/esp-dl"
    REQUIRES esp-dl
)
```

## Test

```bash
make test-docker-espdl
```

Compile-only: ESP-DL needs esp32s3 and ESP-IDF v5.3's QEMU emulates esp32
only, so the suite cross-compiles the adapter and stops. The INT8 path here
is therefore build-verified, not numerically verified - unlike the ONNX
Runtime, microTVM and TFLM adapters, whose suites execute.
