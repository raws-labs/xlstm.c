# xlstm.c Cortex-M QEMU hardware-in-the-loop (HIL) harness

Runs the four existing golden-data suites (sLSTM/mLSTM f32 and INT8) on three
emulated Cortex-M cores - M4 (`qemu-system-arm -M mps2-an386`), M7 (`-M
mps2-an500`), M33 (`-M mps2-an505`) - and reports the result over the
semihosting console the way `test/hil/qemu_gate.sh` does for the ESP32-S3
precedent: a provenance banner, then a `##srig-exit:N##` sentinel matching
each run's pass/fail status.

**Scope: harness only, no SIMD backend.** All three images link
`src/xlstm_simd_ref.c` and are built with `-DXLSTM_HIL_EXPECT_BACKEND=ref`.
There is no `src/xlstm_simd_cortexm.c` yet (see CLAUDE.md work items 1-2 and
`.docs/plans/2026-08-14-cortexm-backend.md`, work item 7). If a core fails
here, the harness is at fault, not the kernel - `ref` is already verified at
39 assertions across three host backends (`make test-ref`, `test-sse2`,
`test-neon`).

This directory does not reimplement anything - it points at
`test/hil_runner.cc` (the board-independent runner), `test/hil_platform.h`
(the shim it talks to), the four existing suite `.cc` files, and the kernel
sources under `src/`/`include/`, and supplies `hil_cortexm.c` as this board
layer's implementation of that shim (the bare-metal equivalent of
`test/hil_platform_host.c`, structurally the same shape as
`test/hil/main/hil_esp32s3.c`).

## Why QEMU only, no hardware target here

`qemu-system-arm` cannot be installed on the machine this harness was
developed on without a password-gated `sudo`; Docker already is this
repository's own precedent for running QEMU in a container (see
`test/hil/Dockerfile`/`qemu_gate.sh`). A real hardware target (NUCLEO-F446RE,
NUCLEO-H753ZI, RP2350) is a later work item and, per repository policy
(mirrors `hil-esp32s3` vs `hil-esp32s3-qemu`), would be manual-only, never
wired into CI.

## Quick start: the emulated gate

```
make -C ../../.. hil-cortexm-qemu
```

(or, from the repository root: `make hil-cortexm-qemu`). This builds
`test/hil/cortexm/Dockerfile` and runs the resulting image. Exit code 0 means
all four suites passed, on all three cores, under QEMU; anything else is a
real failure - see "A gate that actually fails" below.

Per-core targets exist too (`make hil-cortexm-m4`, `-m7`, `-m33`) if you only
want to check one core; they run the same image with `HIL_CORTEXM_CORES` set
to just that core.

## Layout

- `linker/mps2_flat.ld` - memory map for M4 (`an386`) and M7 (`an500`): code
  at `0x0`, RAM at `0x20000000`.
- `linker/mps2_an505.ld` - memory map for M33 (`an505`): code at
  `0x10000000`, RAM at `0x28000000`. **Not interchangeable** with the flat
  script - linking `an505` flat hard-faults at boot (see "Traps" below).
- `startup.c` - vector table, reset handler, FPU enable. Shared byte-for-byte
  across all three cores; nothing in it is core-specific.
- `hil_cortexm.c` - the board layer: implements `test/hil_platform.h` via
  plain stdio (retargeted through semihosting - see `startup.c`'s call to
  `initialise_monitor_handles()`), reports core/DSP/FPU provenance, and
  calls `xlstm_hil_run()`.
- `Makefile` - builds `build/hil_m4.elf`, `build/hil_m7.elf`,
  `build/hil_m33.elf` from one shared template instantiated per core (so the
  three cores cannot drift out of sync with each other).
- `qemu_gate.sh` - the actual gate: boots each core's ELF headless under
  `qemu-system-arm`, greps its console log for `##srig-exit:N##`, and exits
  non-zero if any core fails, hangs, or never prints a sentinel.
- `Dockerfile` - `debian:bookworm-slim` + `gcc-arm-none-eabi` +
  `qemu-system-arm`; `RUN make all size` builds and reports footprint; `CMD`
  is `qemu_gate.sh`.

## Traps this harness had to solve (do not rediscover)

See `.docs/plans/2026-08-14-cortexm-backend.md` for the full list; the ones
that landed in this directory's code:

1. **`an505`'s memory map is not `an386`/`an500`'s.** Code at `0x10000000`,
   data at `0x28000000`, not the flat `0x0`/`0x20000000` map. Linked flat,
   `an505` dies with `qemu: fatal: Lockup: can't escalate 3 to HardFault`.
   Two linker scripts exist because of this, not by accident.
2. **The FPU must be enabled before any VFP instruction executes.**
   `startup.c`'s `Reset_Handler` sets `CPACR` (`0xE000ED88`) CP10/CP11 full
   access, then `dsb`/`isb`, before touching `.data`/`.bss` or calling
   `main()`. Without this the first float (including inside
   semihosting-retargeted `printf`) hard-faults. Not predicted by
   inspection - found by running and reading the fault.
3. **The linker needs `__exidx_start`/`__exidx_end`, `end`, `_init`/`_fini`**
   even though this firmware never throws and registers no static
   constructors beyond what `__libc_init_array()` already walks - libgcc's
   unwinder and newlib's `_sbrk()` expect the symbols to exist regardless.
   Both linker scripts and `startup.c` supply them.
4. **Semihosting console stream routing is not something to hardcode a
   redirect around.** `qemu_gate.sh` captures both stdout and stderr into
   the same log file it greps, so this is handled once, at the gate script
   level, regardless of which stream a given QEMU build/config actually
   uses.
5. **`DWT->CYCCNT` reads zero forever under QEMU.** This harness does not
   attempt cycle counting - emulation proves correctness and that the build
   actually targets the right core, never timing. That is a later work item
   (`.docs/plans/2026-08-14-cortexm-backend.md`, work item 9), with its own
   guard against exactly this trap.

## A gate that actually fails

Two ways to check the gate itself is a real gate, not a rubber stamp:

- **Perturbed golden data.** Edit `test/reference_data.h` in a scratch copy
  of the tree (e.g. flip `kSweepS1_expected_y` to an obviously wrong value),
  rebuild, and run the gate: the firmware's own tolerance checks fail, it
  prints `##srig-exit:1##`, and `qemu_gate.sh` exits non-zero.
- **Forced hang.** Edit `hil_cortexm.c` in a scratch copy to spin forever
  before calling `xlstm_hil_run()`, rebuild, and run the gate with a short
  `HIL_QEMU_TIMEOUT`: no sentinel ever appears, the poll loop times out, and
  `qemu_gate.sh` kills QEMU and exits non-zero - never a silent pass.

See `.superpowers/sdd/2026-08-14-cortexm/items-1-2-report.md` for the
transcripts from both.
