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
make test-docker-espdl   # this adapter, executed on an emulated ESP32-S3
make test-esp            # the kernels and esp SIMD backend underneath it
```

`test-docker-espdl` builds these classes into an ESP-IDF v5.4 project for
ESP32-S3 against real ESP-DL and runs it under the QEMU that image ships,
which does have an esp32s3 machine. The firmware exits the emulator with its
own verdict, so a failure is a non-zero exit status rather than a line in a
log.

**What it checks.** This adapter's own dispatch: that `forward()` reads the
tensors it was handed in the documented order, turns each tensor's exponent
into the scale the kernel expects, and carries its states across calls. The
INT8 tests require the module's output to equal, exactly, what
`slstm_eval_s8` / `mlstm_eval_s8` produce when driven with the scales
`DL_SCALE` derives from the same exponents.

**What it cannot check.** It is not the golden-value gate the ONNX Runtime,
microTVM and TFLM suites run, and cannot be made into one. ESP-DL
quantization is power-of-two symmetric with no zero point, structurally: the
`.espdl` FlatBuffers schema carries an `exponents` field and has no scale or
zero-point field at all, and the exporter discards the float scale as
`exponent = int(log2(scale))`. The reference vectors were produced with
arbitrary scales and an asymmetric input zero point, which this framework
cannot represent. Checking the adapter against the core kernels at matching
scales is exactly the claim Espressif make for their own operators, and it is
the claim here. Read a green run as "wired correctly", not as "numerically
verified against PyTorch".

Bit-exactness is also per target rather than per framework: ESP-DL rounds
`ROUND_HALF_UP` on ESP32 and ESP32-S3 and `ROUND_HALF_EVEN` on ESP32-P4.

The kernels and the `esp` SIMD backend these classes call are gated
separately, against the golden vectors: `make test-esp` builds them into an
ESP32-S3 image and runs the full vector set (f32 and INT8, both cells) under
the same emulated part. See `CONTRIBUTING.md` for what that gate does and does
not cover.
