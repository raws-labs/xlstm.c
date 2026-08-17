get_filename_component(XLSTM_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)

idf_component_register(
    SRCS
        "test_espdl.cpp"
        "${XLSTM_ROOT}/adapters/esp-dl/slstm_espdl.cpp"
        "${XLSTM_ROOT}/adapters/esp-dl/mlstm_espdl.cpp"
        "${XLSTM_ROOT}/src/slstm.c"
        "${XLSTM_ROOT}/src/mlstm.c"
        "${XLSTM_ROOT}/src/slstm_s8.c"
        "${XLSTM_ROOT}/src/mlstm_s8.c"
        "${XLSTM_ROOT}/src/xlstm_quant.c"
        "${XLSTM_ROOT}/src/xlstm_simd_esp.c"
    INCLUDE_DIRS
        "${XLSTM_ROOT}/include"
        "${XLSTM_ROOT}/adapters/esp-dl"
    REQUIRES
        esp-dl
)
