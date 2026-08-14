#!/usr/bin/env bash
# Headless QEMU gate for the xlstm.c Cortex-M HIL firmware, all three
# emulated cores in one run: M4 (mps2-an386), M7 (mps2-an500), M33
# (mps2-an505). Mirrors ../qemu_gate.sh's contract (the ESP32-S3
# precedent - see that file for the background this borrows) but drives
# THREE images against THREE QEMU machine types in a loop, instead of one.
#
# This is what test/hil/cortexm/Dockerfile's CMD runs. `docker run`'s own
# exit code IS this script's exit code: 0 only if every core in
# HIL_CORTEXM_CORES actually printed its own "##srig-exit:0##" sentinel
# within HIL_QEMU_TIMEOUT seconds. A hang, a crash, or a missing sentinel
# on ANY core is a non-zero exit - never a silent pass. This is a real
# gate, not a build-only smoke test.
#
# Overridable for testing the gate itself without rebuilding the image:
#   HIL_CORTEXM_CORES=m4              docker run --rm ... # one core only
#   HIL_QEMU_TIMEOUT=3                docker run --rm ... # force a timeout
set -uo pipefail

cd "$(dirname "$0")"

CORES="${HIL_CORTEXM_CORES:-m4 m7 m33}"
TIMEOUT="${HIL_QEMU_TIMEOUT:-60}"

qemu_machine_for() {
    case "$1" in
        m4)  echo mps2-an386 ;;
        m7)  echo mps2-an500 ;;
        m33) echo mps2-an505 ;;
        *)   echo "" ;;
    esac
}

overall_rc=0
mkdir -p build

for core in $CORES; do
    elf="build/hil_${core}.elf"
    mach="$(qemu_machine_for "$core")"
    log="build/qemu_${core}.log"
    : > "$log"

    echo "=== ${core} (${mach:-UNKNOWN MACHINE}): booting ${elf} headless under QEMU (gate timeout ${TIMEOUT}s) ==="

    if [ -z "$mach" ]; then
        echo "HIL GATE [$core]: unknown core name (expected m4, m7, or m33) - FAIL"
        overall_rc=1
        continue
    fi
    if [ ! -f "$elf" ]; then
        echo "HIL GATE [$core]: $elf does not exist (build step never produced it) - FAIL"
        overall_rc=1
        continue
    fi

    # Semihosting console routing has varied across QEMU builds/configs
    # (trap 4, .docs/plans/2026-08-14-cortexm-backend.md) - redirect BOTH
    # stdout and stderr into the one log this script greps, so the
    # sentinel is found regardless of which stream QEMU chose.
    qemu-system-arm -M "$mach" -kernel "$elf" -nographic \
        -semihosting-config enable=on,target=native > "$log" 2>&1 &
    qemu_pid=$!

    sentinel=""
    elapsed=0
    while [ "$elapsed" -lt "$TIMEOUT" ]; do
        if grep -qE '##srig-exit:[0-9]+##' "$log" 2>/dev/null; then
            sentinel="$(grep -oE '##srig-exit:[0-9]+##' "$log" | head -n 1)"
            break
        fi
        if ! kill -0 "$qemu_pid" 2>/dev/null; then
            # QEMU itself died (crash, bad image, ...) before any sentinel
            # showed up. Fall through to the "no sentinel" path below.
            break
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    # The firmware spins forever after printing its sentinel (see
    # startup.c) - QEMU will not exit on its own even after a passing
    # run. Kill it ourselves either way.
    kill "$qemu_pid" 2>/dev/null
    wait "$qemu_pid" 2>/dev/null

    echo "--- ${core} captured console log ---"
    cat "$log"
    echo "--- end ${core} log ---"
    echo

    if [ -z "$sentinel" ]; then
        echo "HIL GATE [$core]: no ##srig-exit:N## sentinel seen within ${TIMEOUT}s - FAIL (hang, crash, or boot failure)"
        overall_rc=1
        continue
    fi

    code="$(echo "$sentinel" | grep -oE '[0-9]+')"
    echo "HIL GATE [$core]: sentinel ${sentinel} -> exit ${code}"
    if [ "$code" != "0" ]; then
        overall_rc=1
    fi
done

echo
if [ "$overall_rc" -eq 0 ]; then
    echo "HIL GATE: all cores (${CORES}) passed"
else
    echo "HIL GATE: at least one core FAILED (see per-core detail above)"
fi
exit "$overall_rc"
