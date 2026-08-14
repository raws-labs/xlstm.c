# ESP-IDF top-level project for the hardware-in-the-loop (HIL) firmware.
#
# Builds test/hil_runner.cc + test/hil_platform.h (main/hil_esp32s3.c is the
# board implementation) fused with the four existing suite entry points
# (test/slstm_test.cc, test/mlstm_test.cc, test/slstm_s8_test.cc,
# test/mlstm_s8_test.cc) into one ESP32-S3 firmware image.
#
# Modeled on test/adapters/esp-dl/CMakeLists.txt, minus EXTRA_COMPONENT_DIRS:
# that adapter vendors esp-dl/esp-dsp by `git clone` into /opt and points at
# them directly, because esp-dl is not (at the pinned IDF v5.3) available
# through the Component Manager. Here the only external dependency is
# esp-dsp, which IS available through the registry, so main/idf_component.yml
# declares it there instead - see that file for the version pin.
cmake_minimum_required(VERSION 3.16)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(xlstm_hil)
