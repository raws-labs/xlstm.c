#!/usr/bin/env bash
# Headless QEMU gate for the xlstm.c ESP32-S3 HIL firmware.
#
# This is what test/hil/Dockerfile's CMD runs. It replaces a CMD that used
# to only print "idf.py size" and list the merged binary - a check that
# exited 0 having executed zero kernels. This script actually boots the
# firmware under QEMU, watches its serial output for the
# "##srig-exit:N##" sentinel test/hil_runner.cc prints, and exits with N.
# If the sentinel never shows up within the timeout below - hang, crash,
# boot failure, anything - this exits non-zero. It never exits 0 by
# default; only a sentinel of exactly 0 does that.
#
# Mechanics, established empirically against this exact image (see
# test/hil/README.md for the summary):
#
#   - `idf.py qemu` blocks forever once QEMU launches - it is meant to run
#     a live emulator, not just produce input files, and there is no way
#     to ask it to stop after boot. So this script drives
#     qemu-system-xtensa directly instead (Dockerfile's build stage still
#     uses `idf.py qemu`, backgrounded and killed once its two output
#     files exist - see the Dockerfile for that half).
#   - `idf.py monitor` requires a TTY; this container has none.
#   - The firmware waits up to 5s for a trigger byte on UART0, then runs
#     regardless, then loops forever re-printing its summary and the
#     sentinel every ~5s. QEMU will NOT exit on its own - this script
#     must kill it once the sentinel is seen (or the timeout expires).
#
# QEMU invocation note: a literal reading of idf.py's own log line
# ("Running qemu (fg): qemu-system-xtensa ... -nographic ... -serial
# mon:stdio") suggests swapping "mon:stdio" for a plain "-serial stdio"
# to get non-interactive headless output. That combination is REJECTED by
# QEMU at startup here: "-serial stdio: cannot use stdio by multiple
# character devices" - "-nographic" has already claimed stdio for its own
# default serial+monitor mux, so adding an explicit "-serial" for the same
# target conflicts. Verified fix: drop the explicit "-serial" entirely.
# Bare "-nographic" alone reproduces idf.py qemu's own working behavior
# exactly (full boot log, provenance banner, all four suites, sentinel),
# needs no TTY (confirmed with stdin closed, output redirected to a plain
# file), and is what is used below.
set -uo pipefail

cd /workspace/test/hil

# shellcheck source=/dev/null
source "$IDF_PATH/export.sh" > /tmp/idf_export.log 2>&1

echo "=== idf.py size (informational only - not the gate) ==="
idf.py size || true
echo
echo "=== merged flashable artifact for hardware rigs (informational only) ==="
ls -la build/xlstm_hil_merged.bin || true
echo

# Overridable so the hang path can be exercised without rebuilding the
# image, e.g.:
#   docker run --rm -e HIL_QEMU_TIMEOUT=3 xlstm-hil
# 3s is below the firmware's 5s trigger-byte fallback, so the sentinel
# can never appear in time and this must exit non-zero.
TIMEOUT="${HIL_QEMU_TIMEOUT:-120}"
LOG="/tmp/hil_serial.log"
: > "$LOG"

echo "=== booting firmware headless under QEMU (gate timeout ${TIMEOUT}s) ==="
qemu-system-xtensa -M esp32s3 \
    -drive file=build/qemu_flash.bin,if=mtd,format=raw \
    -drive file=build/qemu_efuse.bin,if=none,format=raw,id=efuse \
    -global driver=nvram.esp32c3.efuse,property=drive,value=efuse \
    -global driver=timer.esp32s3.timg,property=wdt_disable,value=true \
    -nographic > "$LOG" 2>&1 &
QEMU_PID=$!

SENTINEL=""
ELAPSED=0
while [ "$ELAPSED" -lt "$TIMEOUT" ]; do
    if grep -qE '##srig-exit:[0-9]+##' "$LOG" 2>/dev/null; then
        SENTINEL="$(grep -oE '##srig-exit:[0-9]+##' "$LOG" | head -n 1)"
        break
    fi
    if ! kill -0 "$QEMU_PID" 2>/dev/null; then
        # QEMU itself died (crash, bad image, ...) before any sentinel
        # showed up. Definitely a failure - stop polling, fall through to
        # the "no sentinel" path below.
        break
    fi
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

# The firmware loops forever - QEMU will not exit on its own even after a
# passing sentinel. Kill it ourselves either way.
kill "$QEMU_PID" 2>/dev/null
wait "$QEMU_PID" 2>/dev/null

echo "=== captured serial log ==="
cat "$LOG"
echo "=== end of serial log ==="
echo

if [ -z "$SENTINEL" ]; then
    echo "HIL GATE: no ##srig-exit:N## sentinel seen within ${TIMEOUT}s - FAIL (hang, crash, or boot failure)"
    exit 1
fi

CODE="$(echo "$SENTINEL" | grep -oE '[0-9]+')"
echo "HIL GATE: sentinel ${SENTINEL} -> exit ${CODE}"
exit "$CODE"
