# Docker integration test for the ESP-DL adapter (sLSTM + mLSTM).
#
# Builds the adapter into an ESP-IDF project for ESP32-S3 against real ESP-DL
# and RUNS it on the emulator. The firmware's verdict is the container's exit
# status: on the last line of app_main it issues the Xtensa `simcall` that
# newlib's own _exit uses on this target, and QEMU exits with the status it
# was handed. A red run is a non-zero `docker run`, not a line someone has to
# read.
#
# Usage:
#   docker build -f test/adapters/esp-dl/Dockerfile -t xlstm-test-espdl .
#   docker run --rm xlstm-test-espdl
#
# WHAT A GREEN RUN CLAIMS is narrower than the ONNX Runtime, microTVM and
# TFLM suites claim, and cannot be widened. Those check adapter output against
# the PyTorch-derived golden vectors. ESP-DL cannot be driven to produce them:
# its quantization is power-of-two symmetric with no zero-point, and that is
# structural rather than a setting - the .espdl FlatBuffers schema carries an
# `exponents` field and has no scale or zero-point field at all, and the
# exporter discards the float scale as exponent = int(log2(scale)). So this
# suite checks the adapter against the core kernels driven with matching
# scales, which is the same claim Espressif make for their own operators. The
# kernels themselves, and the esp SIMD backend under them, are gated against
# the golden vectors separately by `make test-esp`.

FROM espressif/idf:v5.4

# v5.4, not the v5.3 this suite used to pin: v5.3 bundles a QEMU with an esp32
# machine and no esp32s3, which is what made the suite compile-only. v5.4's
# compiler is also the one `make test-esp` gates the kernels with.
#
# ESP-DL stays at v3.0.0. It declares esp32s3 and builds unchanged here;
# v3.3.9 does not build against this IDF at all (its image code uses
# MALLOC_CAP_SIMD, which IDF v5.4 does not define), and a newer IDF would
# compile this adapter with a different toolchain than the kernels are gated
# with.
#
# There is no esp-dsp here on purpose. This adapter and the esp SIMD backend
# it calls issue the ESP32-S3 PIE instructions directly and depend on no DSP
# library.
RUN git clone --depth 1 --branch v3.0.0 https://github.com/espressif/esp-dl.git /opt/esp-dl

# The emulator is pinned rather than taken from the image, and the version
# matters. The QEMU inside espressif/idf:v5.4 (esp_develop_9.0.0_20240606)
# executes ESP-DL's own ESP32-S3 block copy incorrectly: TensorBase deep-copies
# the weights it is handed, and under that build the copy does not equal its
# source, so the module runs on corrupted weights and the INT8 checks fail on
# numbers that have nothing to do with this adapter. The release pinned below
# is the one `make test-esp` and CI already use; under it the copy is intact
# and all seven pass. The three libraries are that binary's DT_NEEDED entries;
# SDL2 is linked even though this runs headless, so it will not start without
# it.
ARG QEMU=https://github.com/espressif/qemu/releases/download/esp-develop-9.2.2-20260417/qemu-xtensa-softmmu-esp_develop_9.2.2_20260417-x86_64-linux-gnu.tar.xz
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpixman-1-0 libsdl2-2.0-0 libslirp0 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL "$QEMU" | tar -xJ -C /opt

WORKDIR /workspace

COPY include/ include/
COPY src/ src/
COPY adapters/esp-dl/ adapters/esp-dl/
COPY test/adapters/esp-dl/ test/adapters/esp-dl/

# merge_bin because QEMU boots a flash image rather than an ELF: bootloader at
# 0x0, partition table at 0x8000, app at 0x10000, exactly as a part would.
WORKDIR /workspace/test/adapters/esp-dl
RUN bash -c "source $IDF_PATH/export.sh && \
    idf.py set-target esp32s3 && \
    idf.py build && \
    cd build && \
    esptool.py --chip esp32s3 merge_bin --fill-flash-size 4MB \
        -o /flash.bin @flash_args"

# The IDF entrypoint sources export.sh and prints a banner; nothing below needs
# the IDF environment, and test output should be the only output.
ENTRYPOINT []

# timeout, because a firmware that hangs never returns from a system emulator:
# a hang has to be red rather than a stuck runner. -semihosting is what makes
# the exit status work at all.
CMD ["timeout", "120", "/opt/qemu/bin/qemu-system-xtensa", \
     "-M", "esp32s3", "-semihosting", "-nographic", \
     "-drive", "file=/flash.bin,if=mtd,format=raw"]
