# Fuses the four suite entry points, the kernels and the esp backend into one
# esp32s3 image. Nothing here forks anything under src/ or test/ - it points
# idf_component_register at the same files the host build compiles.
#
# Naming src/xlstm_simd_esp.c explicitly (rather than going through the
# Makefile's XLSTM_SIMD dispatch) is deliberate: which backend object got
# linked is then readable straight out of build/xlstm_esp_gate.map, and not
# only from the runtime check in esp_gate.cc.
#
# /workspace is where ../Dockerfile stages the repository.
set(XLSTM_ROOT /workspace)

idf_component_register(
    SRCS
        "esp_gate.cc"
        "${XLSTM_ROOT}/src/slstm.c"
        "${XLSTM_ROOT}/src/mlstm.c"
        "${XLSTM_ROOT}/src/xlstm_quant.c"
        "${XLSTM_ROOT}/src/slstm_s8.c"
        "${XLSTM_ROOT}/src/mlstm_s8.c"
        "${XLSTM_ROOT}/src/xlstm_simd_esp.c"
        "${XLSTM_ROOT}/test/slstm_test.cc"
        "${XLSTM_ROOT}/test/mlstm_test.cc"
        "${XLSTM_ROOT}/test/slstm_s8_test.cc"
        "${XLSTM_ROOT}/test/mlstm_s8_test.cc"
    INCLUDE_DIRS
        "${XLSTM_ROOT}/include"
        "${XLSTM_ROOT}/test"
    REQUIRES
        esp-dsp
)

# XLSTM_TEST_MAX_H bounds the four runners' own static buffers
# (test/test_config.h); the 256 default does not fit the S3's 512 KB of
# SRAM, and 64 covers every case in test/reference_data.h. It does not touch
# XLSTM_MAX_HIDDEN, so the kernels under test are the shipping ones.
#
# XLSTM_ESP_FASTPATH_COUNTERS compiles the two fast-path counters into
# src/xlstm_simd_esp.c. esp_gate.cc references them unconditionally, so a
# build that lost this define fails to link rather than quietly running an
# uninstrumented gate.
target_compile_definitions(${COMPONENT_LIB} PRIVATE
    XLSTM_TEST_MAX_H=64
    XLSTM_ESP_FASTPATH_COUNTERS
)

# Each suite still spells its entry point XLSTM_TEST_MAIN (default: main),
# so the rename has to be per translation unit - one component-wide define
# would give all four the same name and fail to link.
set_source_files_properties("${XLSTM_ROOT}/test/slstm_test.cc"
    PROPERTIES COMPILE_DEFINITIONS "XLSTM_TEST_MAIN=slstm_test_main")
set_source_files_properties("${XLSTM_ROOT}/test/mlstm_test.cc"
    PROPERTIES COMPILE_DEFINITIONS "XLSTM_TEST_MAIN=mlstm_test_main")
set_source_files_properties("${XLSTM_ROOT}/test/slstm_s8_test.cc"
    PROPERTIES COMPILE_DEFINITIONS "XLSTM_TEST_MAIN=slstm_s8_test_main")
set_source_files_properties("${XLSTM_ROOT}/test/mlstm_s8_test.cc"
    PROPERTIES COMPILE_DEFINITIONS "XLSTM_TEST_MAIN=mlstm_s8_test_main")
