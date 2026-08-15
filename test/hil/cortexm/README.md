# xlstm.c Cortex-M QEMU HIL harness

Runs the four sLSTM/mLSTM golden-data suites on three emulated Cortex-M
cores (M4/an386, M7/an500, M33/an505) under QEMU, inside Docker. Harness
only - all three link the scalar `xlstm_simd_ref` backend; there is no
Cortex-M SIMD backend yet.

## Build and run

    make hil-cortexm-build   # build the image, all three cores
    make hil-cortexm-qemu    # build, then run the QEMU gate

`docker run`'s exit code is the gate's verdict: 0 only if all three cores
print `##srig-exit:0##`; a hang, crash, or missing sentinel is non-zero.

Restrict to one core with `make hil-cortexm-m4` / `-m7` / `-m33`, or
`HIL_CORTEXM_CORES=m4`. Override the per-core boot timeout with
`HIL_QEMU_TIMEOUT` (seconds, default 60).

## Real hardware

No board target yet - QEMU only. No `SRIG_API_KEY` needed here; that is
for the ESP32-S3 board gate (`test/hil/`).
