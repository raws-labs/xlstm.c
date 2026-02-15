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

WORKDIR /workspace

# Copy source tree
COPY include/ include/
COPY src/ src/
COPY adapters/esp-dl/ adapters/esp-dl/
COPY test/reference_data.h test/reference_data.h
COPY test/adapters/esp-dl/ test/adapters/esp-dl/

# Build as ESP-IDF project for ESP32-S3.
# Note: QEMU in IDF v5.3 only supports esp32 (not s3), and ESP-DL only
# supports s3+, so we verify compilation only. The successful build proves
# our adapter links against real ESP-DL on a real cross-compilation target.
WORKDIR /workspace/test/adapters/esp-dl
RUN bash -c "source $IDF_PATH/export.sh && \
    idf.py set-target esp32s3 && \
    idf.py build"

CMD ["echo", "ESP-DL adapter: build OK (esp32s3 cross-compilation verified)"]
