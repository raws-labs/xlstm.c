# Contributing

Contributions are welcome. This project is Apache-2.0 licensed; by submitting
a pull request you agree that your contribution will be licensed under the same
terms.

## Getting started

```bash
make test              # core sLSTM + mLSTM tests (requires gcc, g++)
make test-docker-ort   # ONNX Runtime integration test
make test-docker-tvm   # Apache TVM integration test
make test-docker-tflm  # TensorFlow Lite Micro integration test
make test-docker-espdl # ESP-DL integration test (compile-only)
```

`make test` is fast (seconds). Docker integration tests are slower and require
Docker. CI runs `make test` under both gcc and clang on every PR; the Docker
integration tests are run locally, not in CI.

## Workflow

1. Fork and create a feature branch
2. Make your changes
3. Run `make test` locally; all core tests must pass
4. Run the relevant `make test-docker-*` if you touched an adapter
5. Open a PR against `main`

## Code style

- Core library: **C99**, no dependencies beyond `math.h`
- Adapters: match the target framework's conventions (C++ for TFLM/ORT/ESP-DL,
  C for microTVM)
- No dynamic allocation in the core; callers provide scratch buffers
- Keep adapters thin: unpack tensors, call core, return

## Regenerating reference data

If you change the core math:

```bash
make reference         # requires Python with torch + xlstm
make test              # verify against new golden values
```

This regenerates both `test/reference_data.h` (C tests) and
`test/reference_data.json` (Python/Docker tests) from the NX-AI/xlstm
reference implementation.

## Testing a backend

```bash
make test-ref          # scalar baseline
make test-sse2         # x86
make test-neon         # cross-compile aarch64, run under QEMU
make test-cortexm      # cross-compile armv7-a, run under QEMU (the DSP path)
make test-esp          # cross-compile xtensa, run under QEMU (system, not user)
```

All five run the same golden vectors. `test-cortexm` gates the Cortex-M
backend without a board: `SXTAB16` and `SMLAD` are ARMv6 DSP instructions that
A-profile also has, so the kernels cross-compile for `armv7-a`. It gates the
arithmetic and only the arithmetic:

- armv7-a Linux permits unaligned word access, so the INT8 matvec's
  `-mno-unaligned-access` load path is never compiled and M-profile alignment
  behaviour is not exercised.
- `XLSTM_FPU_HAS_MINMAX_ROUND` resolves to 0 there, as on Cortex-M4, so the
  FPv5 `vminnm`/`vrinta` path that M7 and M33 declare is not covered.
- Emulated execution says nothing about cycles.

Those three are checked on real parts, from a hardware-in-the-loop harness in
a separate repository.

`test-esp` does the same job for the `esp` backend on an emulated ESP32-S3.
Xtensa has no Linux userspace, so it runs under `qemu-system-xtensa` rather
than `qemu-user`; that is the only way it differs. It needs two tarballs from
Espressif's GitHub releases - the `xtensa-esp-elf` toolchain and a
`qemu-xtensa-softmmu` build - on `PATH`, and nothing else. No ESP-IDF project,
no bootloader, no flash image, no Docker. The toolchain's own `sim.elf.specs`
and `sys.qemu.specs` supply the reset and window vectors, the linker script and
newlib's `_write`/`_exit` on the Xtensa `simcall` instruction, so the four
suites cross-compile unchanged and behave like ordinary test binaries: `printf`
reaches the terminal and `main`'s return value is the exit status.

Read what it covers before quoting it. All four contract functions have an
accelerated body there, and which one a call takes is a property of its shape -
never of where the caller's buffers landed. That is exactly what the suites
cannot check: a dispatch stuck at "always scalar" passes every golden vector.
So a fifth binary, `test/esp_gate.cc`, runs each contract function across the
shapes and alignments that straddle its dispatch rule and fails the run unless
every call took the path its shape dictates and matched the shared scalar body
in `src/xlstm_simd_scalar.h` bit for bit. Emulation still says nothing about
the part's FPU corner cases or about cycles.

## The performance gate

```bash
make perf              # needs valgrind
make perf-baseline     # re-record it, deliberately
```

`make bench` prints wall-clock, which no shared runner reproduces closely
enough to fail a build on. `make perf` counts retired instructions under
callgrind instead, collection toggled on one kernel entry point at a time. The
same binary on the same input gives the same count every run, so a 2% move
against `test/perf_baseline.txt` is a real change in work done, and CI fails on
it. A deliberate change is a `make perf-baseline` and a one-line-per-case diff.

Two limits, worth knowing before trusting a green run:

- **It is a proxy for time, not time.** A change that leaves the instruction
  count alone and worsens cache behaviour passes. Not hypothetical: a change
  with identical instruction counts cost 10% on one Cortex-M part, and a
  smaller binary measured slower on three.
- **Host backends only** (`ref`, `sse2`). `cortexm` and `esp` performance is a
  property of those cores and is measured on hardware.

Counts are specific to the compiler that produced them, so the gate refuses to
compare across a toolchain it did not record.

## Changing a tolerance, a bound, or the generator

```bash
make mutants           # ~35s, edits the working tree and restores it
```

Those changes fail by making a gate quietly stop failing, which a green
`make test` cannot show you. `make mutants` injects the defects the bounds
exist to catch - an activation drift, a zeroed exit state, a state
requantization drift, a dropped zero point, a matvec that skips its SIMD tail,
a single corrupted channel - rebuilds, and asserts the suites fail. A defect
they no longer notice is an escape, and fails the target.

One mutation must **pass**: a 0.1% activation drift. That is the portability
margin the INT8 bounds are derived with, so that a backend whose sigmoid and
tanh are approximations rather than libm - a CMSIS-NN lookup table, say - is
admitted rather than failed. Bounds tight enough to catch it would reject
legitimate backends. The same drift at 0.2% must fail, which is what keeps the
margin a margin rather than a hole.

It covers `ref` and `sse2`; a mutation the running backend does not compile
reports `n/a`, which is distinct from an escape. `neon`, `cortexm` and `esp`
have loop tails and zero-point handling of their own that nothing here mutates.
Not in CI - it edits files in the working tree, which belongs in a run someone
chose to start. It restores them on exit, on failure and on interrupt, and a
run killed outright leaves `.mutants-backup/` for the next run to restore from.
Run it locally, and say in the PR that you did.

## Reporting issues

Open an issue on GitHub. Include:
- What you expected vs what happened
- Minimal reproduction steps
- Compiler/OS/framework versions if relevant
