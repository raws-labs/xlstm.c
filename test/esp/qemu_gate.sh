#!/usr/bin/env bash
# Boots the built firmware headless under QEMU and exits with its verdict.
# This is the gate: the image's CMD, and the reason `docker run` exits 0
# only when the firmware printed ##xlstm-esp-gate:0##.
#
# Three mechanics, each of which has to be this way:
#   - `idf.py monitor` needs a TTY, and a container has none.
#   - `idf.py qemu` blocks forever once QEMU launches - it exists to run a
#     live emulator and has no flag to stop after boot - so this drives
#     qemu-system-xtensa directly and generates the two input files that
#     wrapper would have generated.
#   - `-nographic` has already claimed stdio for its own serial+monitor mux,
#     so adding `-serial stdio` is rejected outright ("cannot use stdio by
#     multiple character devices"). Bare `-nographic` is the spelling that
#     works, needs no TTY, and matches what `idf.py qemu` itself runs.
# The firmware's app_main returns but FreeRTOS keeps running, so QEMU never
# exits on its own and this has to kill it once it has the answer.
set -uo pipefail

cd /workspace/test/esp/build
source "$IDF_PATH/export.sh" > /dev/null 2>&1

# The two files a direct qemu-system-xtensa needs: the flash image padded to
# the full flash size (real hardware does not need the padding), and the
# target's default efuse state. Both steps are what `idf.py qemu` does; the
# efuse blob is read out of the IDF's own table rather than pinned here, so
# it tracks whichever IDF this image ships.
python -m esptool --chip=esp32s3 merge_bin --output=qemu_flash.bin \
    --fill-flash-size=4MB @flash_args > /dev/null
python - <<'PY'
import os, sys
sys.path.insert(0, os.environ['IDF_PATH'] + '/tools')
from idf_py_actions.qemu_ext import QEMU_TARGETS
open('qemu_efuse.bin', 'wb').write(QEMU_TARGETS['esp32s3'].default_efuse)
PY

# Overridable so the no-sentinel path can be exercised without a rebuild.
# A full run takes about 2s under this QEMU, so 1 is short enough that the
# sentinel cannot arrive in time and the gate must go red:
#   docker run --rm -e XLSTM_ESP_GATE_TIMEOUT=1 xlstm-test-esp   # exits 1
TIMEOUT="${XLSTM_ESP_GATE_TIMEOUT:-180}"
LOG=/tmp/esp_gate.log
: > "$LOG"

echo "=== booting the gate firmware under QEMU (timeout ${TIMEOUT}s) ==="
# The efuse device really is nvram.esp32c3.efuse on the esp32s3 machine -
# this QEMU registers no esp32s3-named one. Not a typo.
qemu-system-xtensa -M esp32s3 \
    -drive file=qemu_flash.bin,if=mtd,format=raw \
    -drive file=qemu_efuse.bin,if=none,format=raw,id=efuse \
    -global driver=nvram.esp32c3.efuse,property=drive,value=efuse \
    -global driver=timer.esp32s3.timg,property=wdt_disable,value=true \
    -nographic > "$LOG" 2>&1 &
qemu=$!

sentinel=""
for ((t = 0; t < TIMEOUT; ++t)); do
    sentinel=$(grep -oE '##xlstm-esp-gate:[0-9]+##' "$LOG" | head -n 1)
    [ -n "$sentinel" ] && break
    kill -0 "$qemu" 2>/dev/null || break   # QEMU died before saying anything
    sleep 1
done
kill "$qemu" 2>/dev/null
wait "$qemu" 2>/dev/null

cat "$LOG"
if [ -z "$sentinel" ]; then
    echo "esp gate: no ##xlstm-esp-gate:N## within ${TIMEOUT}s - FAIL" \
         "(hang, crash, or boot failure)"
    exit 1
fi

code="${sentinel//[^0-9]/}"
echo "esp gate: $sentinel -> exit $code"
exit "$code"
