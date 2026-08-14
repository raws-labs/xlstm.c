# xlstm.c ESP32-S3 hardware-in-the-loop (HIL) harness

Runs the four existing golden-data suites (sLSTM/mLSTM f32 and INT8) against
the `esp` SIMD backend (`src/xlstm_simd_esp.c`) on ESP32-S3, and reports the
result over UART0 the way the SiliconRig rig expects: a provenance banner,
then a `##srig-exit:N##` sentinel matching the run's pass/fail status.

This directory does not reimplement anything - it is an ESP-IDF project
that points at `test/hil_runner.cc` (the board-independent runner),
`test/hil_platform.h` (the shim it talks to), the four existing suite
`.cc` files, and the kernel sources under `src/`/`include/`, and supplies
`main/hil_esp32s3.c` as this board's implementation of that shim.

There are two ways to run it, and they check different things:

- **Emulated (`make hil-esp32s3-qemu`)**: builds the Docker image and runs
  the firmware headless under QEMU inside the container. `docker run`'s
  own exit code IS the firmware's verdict - a hang, crash, or missing
  sentinel is a non-zero exit, never a silent pass. This is the CI-safe
  path and what a human runs to check the firmware actually works before
  touching a real board.
- **Hardware (`make hil-esp32s3`)**: flashes the same merged binary to a
  real ESP32-S3 through [SiliconRig](https://siliconrig.io) (`srig`) and
  watches its serial output. This is the only path that proves anything
  about real silicon - QEMU's Xtensa FPU emulation is not guaranteed to
  match real hardware bit-for-bit (see "A gate that actually fails" below
  for a concrete case). Manual only, **never** wired into CI.

## Quick start: the emulated gate

```
make hil-esp32s3-qemu
```

This builds `test/hil/Dockerfile` and runs the resulting image. Exit code
0 means all four suites passed on the `esp` backend under QEMU; anything
else is a real failure (see "A gate that actually fails" below). The
image's `CMD` is `test/hil/qemu_gate.sh` - read it for exactly how the
firmware is booted and how its serial output maps to the exit code.

Equivalent by hand:

```
docker build -f test/hil/Dockerfile -t xlstm-hil .
docker run --rm xlstm-hil
```

`docker run`'s output includes, in order: an informational `idf.py size`
memory breakdown and merged-binary listing (not part of the gate itself),
the firmware's full boot log, an `XLSTM_PROVENANCE:{...}` line, four
`HIL_SUITE_BEGIN`/`HIL_SUITE_END` pairs with each suite's own pass/fail
detail in between, the captured serial log echoed back in full, and
finally `HIL GATE: sentinel ##srig-exit:N## -> exit N` - the line that
explains the container's own exit code.

To exercise the hang/timeout path without rebuilding (the gate's serial
watch has its own timeout, independent of anything QEMU or Docker impose):

```
docker run --rm -e HIL_QEMU_TIMEOUT=3 xlstm-hil
```

3 seconds is below the firmware's 5-second trigger-byte fallback (see
"Boot race" below), so the sentinel can never appear in time and this
must exit non-zero - confirming a hang fails instead of silently passing.

### Extracting the merged binary

```
make hil-esp32s3-build
```

Builds the image and copies `build/xlstm_hil_merged.bin` out to this
repo's own `build/` directory: bootloader + partition table + app
pre-merged at their real flash offsets (0x0 / 0x8000 / 0x10000), ready
for a single `esptool.py write_flash` at offset 0x0. `hil-esp32s3` (the
hardware target, below) uses this same target internally.

## Running on real hardware

```
export SRIG_API_KEY=<your key>
make hil-esp32s3
```

`make hil-esp32s3` runs `hil-preflight` first (see below), then
`hil-esp32s3-build`, then hands the merged binary to `srig run` against
board `esp32-s3`. `SRIG_API_KEY` is your SiliconRig account's API key -
**never commit it**. Keep it in your shell environment (`export
SRIG_API_KEY=...` in a shell profile that is itself not committed) or in
a local `.env`-style file your shell sources before running `make`; this
repository's `.gitignore` does not special-case any particular file for
this, so treat it like any other credential and simply never `git add`
whatever file holds it.

`hil-esp32s3` is deliberately **not** wired into `.github/workflows/ci.yml`
and never will be by default - repository policy is that hardware runs are
manual only. CI only exercises host backends, the cross-compiled NEON
backend, and (via `hil-esp32s3-qemu`, if a workflow chooses to add it) the
emulated gate.

### The `--expect` trap

The `srig run` invocation inside `hil-esp32s3` is:

```
srig run <merged.bin> --board esp32-s3 --expect '##srig-exit:0##' \
    --timeout 120s --retries 3 --retry-delay 20s --log <logfile>
```

`--expect` is **not optional decoration**. In `srig-cli`'s
`runner/eval.go:64`, `Timeout()` returns `ExitCode: 0` when `expect` is
`nil` - so a firmware that hangs, crashes, or never gets far enough to
print its sentinel **times out into a clean pass** if `--expect` is
missing. The sentinel line existing at all is not enough to protect
against this; only telling `srig` what to actually look for is. The
Makefile recipe for `hil-esp32s3` carries a comment saying exactly this
right next to the flag, on purpose - this has already been independently
rediscovered and written down wrong twice before, so the warning lives
where someone editing the recipe will actually see it, not only here.

## Preflight

```
make hil-preflight
```

Fails fast, before any network or hardware call, if `SRIG_API_KEY` is
unset, if `srig` is not on `PATH`, or if the installed `srig` predates the
`run` subcommand (built from rev `dfd7c6d` or later - installed at
`~/.local/bin/srig` in the environment this harness was developed
against). Each failure prints what to do about it. `hil-esp32s3` runs this
automatically; run it standalone to check your setup without touching a
board.

## Backend guard

`test/hil_runner.cc` requires `-DXLSTM_HIL_EXPECT_BACKEND="esp"`
(`main/CMakeLists.txt` sets it, per translation unit, only on
`hil_runner.cc`) and fails loudly before running any suite if the SIMD
backend actually linked disagrees. `main/CMakeLists.txt`'s `SRCS` list
names `src/xlstm_simd_esp.c` explicitly (not the dispatch-by-macro
`Makefile` path the host build uses), so this is not just a runtime
check - the object file linked is also directly verifiable in
`build/xlstm_hil.map`.

## A gate that actually fails

This harness used to end in a `CMD` that ran `idf.py size` and listed the
merged binary - a check that exited 0 having executed zero kernels. It no
longer does. The first real run of the fixed gate against this branch's
own firmware caught something: `mLSTM SweepM17` (H=17) fails on the `esp`
backend under QEMU, `y[0]` expected `14.88931370`, got `14.88927174` (diff
4.20e-05), just outside the mixed absolute+relative tolerance the host
suites use. The same case passes cleanly on every host backend (ref,
sse2) - this is not a logic bug, it is the same floating-point
summation-order sensitivity already diagnosed for the host backends
(different backends group the running sum differently; float addition is
not associative), landing outside tolerance specifically for `esp`'s
grouping (ESP-DSP's `dsps_dotprod_f32_ansi` sums into a fresh `dot`
starting at 0, then the caller does `out[i] += dot`, versus the host
backends' single accumulator seeded from `out[i]` from the start) on this
one case. Nothing in this harness's own code changed that outcome - it is
what running the firmware for real, for the first time, actually showed.
Do not "fix" this by loosening a tolerance or excluding the case; that is
exactly the kind of check-that-does-not-check this harness exists to
prevent. See the W7 report for the full measurement.

## Configuration

Every non-default `sdkconfig.defaults` choice is commented in that file
with its specific reason (large INT8 kernel stack frames, the task
watchdog vs. a multi-second suite run with no yield points,
`reference_data.h`'s size against the default 1 MB app partition, and the
rig's undocumented flash size). `main/hil_esp32s3.c`'s provenance fields
report the runtime-detected flash size and PSRAM presence specifically so
the conservative guesses in `sdkconfig.defaults` can be corrected from a
real run's evidence rather than guessed again.

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

## Notes on driving QEMU directly

`test/hil/qemu_gate.sh` (the emulated gate's `CMD`) and the build-time
artifact generation step in `test/hil/Dockerfile` both call
`qemu-system-xtensa` (or `idf.py qemu`, which wraps it) instead of using
`idf.py monitor`. Two mechanics worth knowing if you touch either file:

- `idf.py monitor` requires a TTY and fails in a container with "Monitor
  requires standard input to be attached to TTY".
- `idf.py qemu` blocks forever once QEMU launches - it exists to run a
  live emulator, not to just generate its input files, and has no flag to
  stop short of that. It also generates the two files a direct
  `qemu-system-xtensa` invocation needs: `qemu_flash.bin` (a 4 MB flash
  image, padded to the full flash size) and `qemu_efuse.bin` (1024 bytes
  of efuse state), both in `build/`.
- On backgrounding `idf.py qemu`: the natural-looking fix - launch it with
  `&`, poll until both files exist, then `pkill` the `qemu-system-xtensa`
  child it spawned - does not survive inside a Docker build step here; the
  whole `RUN` was observed to die (exit 143) within about a second of
  backgrounding, before the poll loop even had a chance to run. The
  working fix `test/hil/Dockerfile` actually uses is simpler: run
  `timeout -k 5 20 idf.py qemu`, throw away its (expected non-zero) exit
  status, and check for the two output files afterward - both are on disk
  within about 2 seconds of startup, so 20 seconds is generous headroom,
  not a tuned figure.
- A literal reading of `idf.py qemu`'s own log line ("Running qemu (fg):
  qemu-system-xtensa ... -nographic ... -serial mon:stdio") suggests
  swapping `mon:stdio` for a plain `-serial stdio` to get non-interactive
  headless output. QEMU rejects that combination outright: `-serial
  stdio: cannot use stdio by multiple character devices` - `-nographic`
  has already claimed stdio for its own default serial+monitor mux, so an
  explicit `-serial` for the same target conflicts. The fix is to drop the
  explicit `-serial` entirely; bare `-nographic` alone reproduces
  `idf.py qemu`'s own working behavior exactly, needs no TTY, and is what
  `qemu_gate.sh` uses.
- The firmware loops forever once it has printed its result, so QEMU never
  exits on its own - whatever drives it (this script, or a human at
  `idf.py qemu`) has to actively kill it after observing (or timing out
  on) the sentinel.
