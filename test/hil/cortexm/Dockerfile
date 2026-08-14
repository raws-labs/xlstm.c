# Docker build+gate for the Cortex-M QEMU hardware-in-the-loop (HIL)
# harness. See test/hil/cortexm/README.md and .docs/plans/2026-08-14-
# cortexm-backend.md for the design background this implements.
#
# Builds test/hil_runner.cc (the four existing suites fused into one
# pass/fail run) plus this directory's Cortex-M board layer
# (hil_cortexm.c) and bare-metal startup/linker files into THREE firmware
# ELFs - one per emulated Cortex-M core (M4/an386, M7/an500, M33/an505) -
# all with XLSTM_SIMD=ref (see CLAUDE.md work items 1-2: harness only, no
# SIMD backend - src/xlstm_simd_cortexm.c does not exist yet).
#
# This is a REAL gate, not a build-only smoke test: CMD (qemu_gate.sh)
# boots each of the three images headless under qemu-system-arm, reads its
# semihosting console, and exits with a verdict built from all three
# cores' own "##srig-exit:N##" sentinels. `docker run` on this image only
# exits 0 if every core actually ran and passed - a hang, a crash, or a
# missing sentinel on any one of the three is a non-zero exit, never a
# silent pass.
#
# Usage:
#   docker build -f test/hil/cortexm/Dockerfile -t xlstm-hil-cortexm .
#   docker run --rm xlstm-hil-cortexm

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc-arm-none-eabi \
      libnewlib-arm-none-eabi \
      libstdc++-arm-none-eabi-newlib \
      qemu-system-arm \
      make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Selective COPY, matching test/hil/Dockerfile's pattern for the ESP32-S3
# precedent: only what test/hil/cortexm/Makefile references, plus what
# those sources themselves #include. Deliberately excludes test/adapters/
# (a separate, unrelated build) and the other test/*.cc integration
# harnesses (see CLAUDE.md).
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
COPY test/hil/cortexm/ test/hil/cortexm/

WORKDIR /workspace/test/hil/cortexm

# Builds all three cores and prints each one's flash (text+data) / SRAM
# (data+bss) footprint - informational at build time; qemu_gate.sh (CMD,
# below) is the actual gate.
RUN make all size

CMD ["bash", "qemu_gate.sh"]
