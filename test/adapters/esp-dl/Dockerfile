# Docker integration test for ESP-DL adapter (sLSTM + mLSTM).
#
# Builds as an ESP-IDF project targeting ESP32-S3, runs on QEMU.
# Tests that our Module subclasses instantiate, compute output shapes,
# and survive init/destroy cycles against the real ESP-DL framework.
#
# Usage:
#   docker build -f test/adapters/esp-dl/Dockerfile -t xlstm-test-espdl .
#   docker run --rm xlstm-test-espdl

FROM espressif/idf:v5.3

# Clone ESP-DL v3.0.0 (compatible with IDF v5.3)
RUN git clone --depth 1 --branch v3.0.0 https://github.com/espressif/esp-dl.git /opt/esp-dl

# Clone ESP-DSP v1.5.2 (released Aug 2024, contemporaneous with IDF v5.3;
# provides dsps_dotprod_* used by the esp SIMD backend in
# src/xlstm_simd_esp.c). ESP-DL does not pull this in itself - it is a
# separate component our own adapter code depends on directly.
RUN git clone --depth 1 --branch v1.5.2 https://github.com/espressif/esp-dsp.git /opt/esp-dsp

WORKDIR /workspace

# Copy source tree
COPY include/ include/
COPY src/ src/
COPY adapters/esp-dl/ adapters/esp-dl/
COPY test/reference_data.h test/reference_data.h
COPY test/adapters/esp-dl/ test/adapters/esp-dl/

# Build as ESP-IDF project for ESP32-S3.
#
# COMPILE-ONLY SUITE. The QEMU bundled with IDF v5.3 has an esp32 machine and
# no esp32s3, and ESP-DL requires s3+, so the test binary is built and never
# run. A green result here means the adapter compiles and links against real
# ESP-DL on a real cross-compilation target - it says nothing about the
# numbers. The assertions in main/test_espdl.cpp are written for a runner that
# does not exist yet.
#
# That limit is this image's, not the platform's: test/esp/ pins IDF v5.4,
# whose QEMU does have an esp32s3 machine, and executes the full golden vector
# set against the kernels and the esp SIMD backend (`make test-esp`). What is
# still unexecuted is this adapter's own ESP-DL dispatch, which is what needs
# ESP-DL and therefore this image.
WORKDIR /workspace/test/adapters/esp-dl
RUN bash -c "source $IDF_PATH/export.sh && \
    idf.py set-target esp32s3 && \
    idf.py build"

CMD ["echo", "ESP-DL adapter: esp32s3 cross-compilation OK. Tests were NOT run - no s3 emulator."]
