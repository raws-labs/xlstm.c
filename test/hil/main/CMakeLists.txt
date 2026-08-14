# Fuses the four existing suite entry points and the board-independent HIL
# runner (test/hil_runner.cc, test/hil_platform.h) with this board's
# implementation of the platform shim (hil_esp32s3.c) into one firmware
# image. Nothing here reimplements or forks anything under test/ - it only
# points idf_component_register at the existing sources and supplies the
# per-translation-unit defines test/hil_runner.cc documents.
#
# XLSTM_ROOT hardcodes /workspace the same way test/adapters/esp-dl's
# main/CMakeLists.txt hardcodes its own /workspace/... source paths - see
# ../Dockerfile for what actually lands there and why.
set(XLSTM_ROOT /workspace)

idf_component_register(
    SRCS
        "hil_esp32s3.c"
        "${XLSTM_ROOT}/src/slstm.c"
        "${XLSTM_ROOT}/src/mlstm.c"
        "${XLSTM_ROOT}/src/xlstm_quant.c"
        "${XLSTM_ROOT}/src/slstm_s8.c"
        "${XLSTM_ROOT}/src/mlstm_s8.c"
        "${XLSTM_ROOT}/src/xlstm_simd_esp.c"
        "${XLSTM_ROOT}/test/hil_runner.cc"
        "${XLSTM_ROOT}/test/slstm_test.cc"
        "${XLSTM_ROOT}/test/mlstm_test.cc"
        "${XLSTM_ROOT}/test/slstm_s8_test.cc"
        "${XLSTM_ROOT}/test/mlstm_s8_test.cc"
    INCLUDE_DIRS
        "${XLSTM_ROOT}/include"
        "${XLSTM_ROOT}/test"
    REQUIRES
        esp-dsp
    PRIV_REQUIRES
        esp_driver_uart
        esp_hw_support
        spi_flash
        esp_psram
        freertos
        esp_common
)

# test/test_config.h's XLSTM_TEST_MAX_H knob, shared by all four suite
# runners' static state/scratch/output buffers - shrunk from the 256 default
# to fit the S3's 512 KB SRAM (~108 KB total BSS across all four suites at
# 64; the INT8 binaries alone exceed SRAM at 256). See test/test_config.h
# and the W5 report for the measured breakdown. This is a single build-wide
# define (unlike XLSTM_TEST_MAIN below) because every translation unit that
# includes test/test_config.h should agree on one buffer size.
target_compile_definitions(${COMPONENT_LIB} PRIVATE
    XLSTM_TEST_MAX_H=64
)

# test/hil_runner.cc calls the four suites' entry points by fixed name
# (slstm_test_main, mlstm_test_main, slstm_s8_test_main, mlstm_s8_test_main).
# Each of the four .cc files below still defines its entry point as
# `XLSTM_TEST_MAIN` (default: plain `main`), so it must be renamed per
# translation unit via a COMPILE_DEFINITIONS source property, not a
# component-wide define - a single global XLSTM_TEST_MAIN would give all
# four the same name and fail to link.
set_source_files_properties(
    "${XLSTM_ROOT}/test/slstm_test.cc"
    PROPERTIES COMPILE_DEFINITIONS "XLSTM_TEST_MAIN=slstm_test_main"
)
set_source_files_properties(
    "${XLSTM_ROOT}/test/mlstm_test.cc"
    PROPERTIES COMPILE_DEFINITIONS "XLSTM_TEST_MAIN=mlstm_test_main"
)
set_source_files_properties(
    "${XLSTM_ROOT}/test/slstm_s8_test.cc"
    PROPERTIES COMPILE_DEFINITIONS "XLSTM_TEST_MAIN=slstm_s8_test_main"
)
set_source_files_properties(
    "${XLSTM_ROOT}/test/mlstm_s8_test.cc"
    PROPERTIES COMPILE_DEFINITIONS "XLSTM_TEST_MAIN=mlstm_s8_test_main"
)

# test/hil_runner.cc's backend guard (#error if unset, hard-fail at runtime
# if the linked backend disagrees) - this is the whole point of this build:
# prove the `esp` SIMD backend, not a silent fallback to `ref`, is what ran.
set_source_files_properties(
    "${XLSTM_ROOT}/test/hil_runner.cc"
    PROPERTIES COMPILE_DEFINITIONS "XLSTM_HIL_EXPECT_BACKEND=esp"
)
