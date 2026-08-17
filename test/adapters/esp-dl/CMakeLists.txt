cmake_minimum_required(VERSION 3.16)

# Where ESP-DL is checked out. The Dockerfile clones it here; pass
# -DESP_DL_DIR=<path> to build against a checkout of your own.
if(NOT DEFINED ESP_DL_DIR)
    set(ESP_DL_DIR /opt/esp-dl)
endif()
set(EXTRA_COMPONENT_DIRS ${ESP_DL_DIR})

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(xlstm_espdl_test)
