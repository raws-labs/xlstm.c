# Docker build for the ESP32-S3 hardware-in-the-loop (HIL) firmware.
#
# Builds test/hil_runner.cc (the four existing suites fused into one
# pass/fail run) plus main/hil_esp32s3.c (the ESP32-S3 board layer) as an
# ESP-IDF project, then emits a single merged, flashable binary a rig can
# write with one `esptool.py write_flash` at offset 0x0.
#
# v5.4, not the v5.3 test/adapters/esp-dl/Dockerfile uses: v5.3's bundled
# QEMU has no esp32s3 machine, and this image's QEMU (esp_develop_9.0.0)
# does.
#
# This is a REAL gate, not a build-only smoke test: CMD (see the bottom of
# this file and test/hil/qemu_gate.sh) boots the firmware headless under
# QEMU, reads its serial output, and exits with the firmware's own
# "##srig-exit:N##" verdict. `docker run` on this image only exits 0 if
# all four suites actually ran and passed - a hang, crash, or missing
# sentinel is a non-zero exit, never a silent pass.
#
# Usage:
#   docker build -f test/hil/Dockerfile -t xlstm-hil .
#   docker run --rm xlstm-hil
#   docker create --name hil-extract xlstm-hil && \
#     docker cp hil-extract:/workspace/test/hil/build/xlstm_hil_merged.bin . && \
#     docker rm hil-extract

FROM espressif/idf:v5.4

WORKDIR /workspace

# Selective COPY, matching test/adapters/esp-dl/Dockerfile's pattern: only
# what main/CMakeLists.txt's SRCS/INCLUDE_DIRS reference by absolute
# /workspace/... path, plus what those sources themselves #include.
# Deliberately excludes test/adapters/ (a separate, unrelated build; see
# CLAUDE.md) and the other test/*.cc integration harnesses.
COPY include/ include/
COPY src/ src/
COPY test/test_config.h test/test_config.h
COPY test/test_util.h test/test_util.h
COPY test/hil_platform.h test/hil_platform.h
COPY test/hil_runner.cc test/hil_runner.cc
COPY test/reference_data.h test/reference_data.h
COPY test/slstm_test.cc test/slstm_test.cc
COPY test/mlstm_test.cc test/mlstm_test.cc
COPY test/slstm_s8_test.cc test/slstm_s8_test.cc
COPY test/mlstm_s8_test.cc test/mlstm_s8_test.cc
COPY test/hil/ test/hil/

WORKDIR /workspace/test/hil

# esp-dsp is resolved by the IDF Component Manager from main/idf_component.yml
# (pinned to 1.5.2, matching test/adapters/esp-dl/Dockerfile's `git clone
# --branch v1.5.2` of the same library) - no manual clone needed here.
RUN bash -c "source \$IDF_PATH/export.sh && \
    idf.py set-target esp32s3 && \
    idf.py build"

# Merge bootloader + partition table + app into one flashable image at their
# real flash offsets (0x0 / 0x8000 / 0x10000, matching the partition table
# idf.py build prints and this project's sdkconfig.defaults flash settings).
# This is what a rig actually writes with one `esptool.py write_flash`, not
# the three separate files idf.py build produces on its own.
RUN bash -c "source \$IDF_PATH/export.sh && \
    python -m esptool --chip esp32s3 merge_bin \
      -o build/xlstm_hil_merged.bin \
      --flash_mode dio --flash_size 4MB --flash_freq 80m \
      0x0     build/bootloader/bootloader.bin \
      0x8000  build/partition_table/partition-table.bin \
      0x10000 build/xlstm_hil.bin"

# Generate the two artifacts qemu_gate.sh's CMD needs to boot this
# firmware under QEMU directly: qemu_flash.bin (a 4 MB flash image, padded
# to the full flash size unlike xlstm_hil_merged.bin above, which real
# hardware does not need) and qemu_efuse.bin (1024 bytes of efuse state).
#
# `idf.py qemu` is the only thing that knows how to produce these two
# files - but it BLOCKS forever once QEMU actually launches, since its
# job is to run a live emulator, not just generate its input files, and
# it has no flag to stop short of that. Both files are written to disk
# (confirmed by their timestamps) within a couple of seconds of startup,
# well before QEMU itself finishes booting the guest - so this runs it
# under `timeout` and discards whatever exit status that produces (124
# from the timeout firing is the EXPECTED outcome here, not a failure);
# a `&`-backgrounded alternative was tried first and does not survive
# here - see the "on backgrounding idf.py qemu" note in
# test/hil/README.md for why. What actually gates this step is the
# explicit file-existence check below, not idf.py's own exit code.
RUN bash -c "source \$IDF_PATH/export.sh && \
    rm -f build/qemu_flash.bin build/qemu_efuse.bin && \
    (timeout -k 5 20 idf.py qemu > /tmp/qemu_gen.log 2>&1 || true) && \
    if [ ! -f build/qemu_flash.bin ] || [ ! -f build/qemu_efuse.bin ]; then \
      echo 'FATAL: idf.py qemu did not produce qemu_flash.bin/qemu_efuse.bin' >&2; \
      cat /tmp/qemu_gen.log >&2; \
      exit 1; \
    fi; \
    ls -la build/qemu_flash.bin build/qemu_efuse.bin"

CMD ["bash", "/workspace/test/hil/qemu_gate.sh"]
