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

## Build

Add to your ESP-IDF component's `CMakeLists.txt`:

```cmake
idf_component_register(
    SRCS "slstm_espdl.cpp" "mlstm_espdl.cpp" "slstm.c" "mlstm.c"
    INCLUDE_DIRS "include" "adapters/esp-dl"
    REQUIRES esp-dl
)
```

## Test

```bash
make test-docker-espdl
```
