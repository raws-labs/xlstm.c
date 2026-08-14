# xlstm.c ESP32-S3 hardware-in-the-loop (HIL) firmware

Runs the four existing golden-data suites (sLSTM/mLSTM f32 and INT8) against
the `esp` SIMD backend (`src/xlstm_simd_esp.c`) on real ESP32-S3 hardware,
and reports the result over UART0 the way the SiliconRig rig expects: a
provenance banner, then a `##srig-exit:N##` sentinel matching the run's
pass/fail status.

This directory does not reimplement anything - it is an ESP-IDF project
that points at `test/hil_runner.cc` (the board-independent runner),
`test/hil_platform.h` (the shim it talks to), the four existing suite
`.cc` files, and the kernel sources under `src/`/`include/`, and supplies
`main/hil_esp32s3.c` as this board's implementation of that shim.

## Build

```
docker build -f test/hil/Dockerfile -t xlstm-hil .
```

Emits `build/xlstm_hil_merged.bin` inside the image: bootloader +
partition table + app pre-merged at their real flash offsets
(0x0 / 0x8000 / 0x10000), ready for a single `esptool.py write_flash` at
offset 0x0. Extract it with:

```
docker create --name hil-extract xlstm-hil
docker cp hil-extract:/workspace/test/hil/build/xlstm_hil_merged.bin .
docker rm hil-extract
```

`docker run --rm xlstm-hil` (no extract) prints the `idf.py size` memory
breakdown and confirms the merged binary exists.

## Flash and run

```
esptool.py --chip esp32s3 -p <PORT> write_flash 0x0 xlstm_hil_merged.bin
```

Then open the port at 115200 8N1. The rig may optionally send one byte
within 5 seconds of opening the port to positively trigger the run; if
none arrives, the firmware proceeds anyway after the timeout (see "Boot
race" below). Expect, in order: a boot log from the ROM/IDF bootloader (not
part of this protocol), an `XLSTM_PROVENANCE:{...}` line, four
`HIL_SUITE_BEGIN`/`HIL_SUITE_END` pairs with each suite's own pass/fail
detail in between, then `##srig-exit:0##` (all four suites passed) or
`##srig-exit:1##` (something failed, or the linked backend was not `esp` -
see the backend guard below). After that the firmware loops forever,
re-printing a short summary and the sentinel every 5 seconds - it never
resets and never returns, so a rig that attaches late or misses the
trigger window still observes the result.

## Backend guard

`test/hil_runner.cc` requires `-DXLSTM_HIL_EXPECT_BACKEND="esp"`
(`main/CMakeLists.txt` sets it, per translation unit, only on
`hil_runner.cc`) and fails loudly before running any suite if the SIMD
backend actually linked disagrees. `main/CMakeLists.txt`'s `SRCS` list
names `src/xlstm_simd_esp.c` explicitly (not the dispatch-by-macro
`Makefile` path the host build uses), so this is not just a runtime
check - the object file linked is also directly verifiable in
`build/xlstm_hil.map`.

## Configuration

Every non-default `sdkconfig.defaults` choice is commented in that file
with its specific reason (large INT8 kernel stack frames, the task
watchdog vs. a multi-second suite run with no yield points, `reference_data.h`'s
size against the default 1 MB app partition, and the rig's undocumented
flash size). `main/hil_esp32s3.c`'s provenance fields report the
runtime-detected flash size and PSRAM presence specifically so the
conservative guesses in `sdkconfig.defaults` can be corrected from a real
run's evidence rather than guessed again.

## Boot race

The rig flashes, opens serial, then optionally sends a trigger byte -
anything printed before the port finishes opening can be missed. `app_main`
in `main/hil_esp32s3.c` waits up to 5 seconds for that byte (proceeding
either way), runs the suite, prints the sentinel, then loops forever
re-printing a short summary and the sentinel - see that file's top comment
for the full rationale and the UART0 driver-takeover ordering this needed
(console owns UART0 in polling mode by default; a blocking, timeout-bounded
read needs the real interrupt-driven driver installed and the VFS console
layer switched onto it, in that order, or output can stop entirely).

## Validated under QEMU before any real hardware

IDF v5.4's bundled QEMU (unlike v5.3's) emulates the esp32s3 machine.
`idf.py qemu` boots this exact firmware, and all four suites pass under
it - this is how the image was verified end to end (boot, the trigger-byte
timeout path, the provenance banner, the backend guard, all four suites,
the repeating sentinel loop) before any real board was involved.
