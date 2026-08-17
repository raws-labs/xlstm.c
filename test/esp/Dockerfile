# Emulated ESP32-S3 gate for the `esp` SIMD backend - see the Makefile's
# test-esp target for what it does and does not cover.
#
# Builds the four golden-vector suites plus main/esp_gate.cc as an ESP-IDF
# project; the CMD boots the result headless under QEMU and exits with the
# firmware's own verdict, so `docker run` exits 0 only if all four suites
# and the fast-path check passed. A hang, a crash or a missing sentinel is a
# non-zero exit, never a silent pass.
#
# v5.4, not the v5.3 that test/adapters/esp-dl/Dockerfile pins: v5.3's
# bundled QEMU has an esp32 machine and no esp32s3, and this image's
# (esp_develop_9.0.0) has both. That is the whole reason this gate can exist.
#
#   docker build -f test/esp/Dockerfile -t xlstm-test-esp .
#   docker run --rm xlstm-test-esp

FROM espressif/idf:v5.4

WORKDIR /workspace

# Only what main/CMakeLists.txt names and what those sources include. The
# backend issues its own 128-bit loads and calls no DSP library, so unlike
# test/adapters/esp-dl/Dockerfile there is nothing to clone and nothing to
# pull from the component registry either - just the IDF.
COPY include/ include/
COPY src/ src/
COPY test/test_config.h test/test_util.h test/reference_data.h test/
COPY test/slstm_test.cc test/mlstm_test.cc \
     test/slstm_s8_test.cc test/mlstm_s8_test.cc test/
COPY test/esp/ test/esp/

WORKDIR /workspace/test/esp

RUN bash -c "source \$IDF_PATH/export.sh && \
    idf.py set-target esp32s3 && \
    idf.py build"

CMD ["bash", "/workspace/test/esp/qemu_gate.sh"]
