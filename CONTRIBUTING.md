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
make test-docker-espdl # ESP-DL integration test (runs on an emulated ESP32-S3)
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
make check-tools       # the tools/ examples re-derive from the new data
```

This regenerates both `test/reference_data.h` (C tests) and
`test/reference_data.json` (Python/Docker tests) from the NX-AI/xlstm
reference implementation.

`make check-tools` matters here because the worked examples in `tools/`
reproduce that file's calibration and shapes from its float tensors alone. If
a quantization convention changes and they are not updated with it, they say
so; they are stdlib-only, so this runs in CI alongside the other hygiene
checks.

## Testing a backend

```bash
make test-ref          # scalar baseline
make test-sse2         # x86
make test-neon         # cross-compile aarch64, run under QEMU
make test-cortexm      # cross-compile armv7-a, run under QEMU (the DSP path)
make test-esp          # cross-compile xtensa, run under QEMU (system, not user)
make test-helium       # cross-compile Cortex-M55, run under QEMU (system)
make test-approx       # the approximate gate build, on both host backends
```

Every one of them takes `XLSTM_GATES=exact|approx`, which picks the INT8 cells'
transcendentals (see the block above `xlstm_gate_expf` in
`include/xlstm_util.h`). `exact` is the default and is what all six are gated
on. `approx` is gated by `make test-approx` on `ref` and the host backend, and
by CI on `cortexm`, which is the target it exists for; the arithmetic is plain
C99 in the shared cell code with no SIMD contract behind it, so a fourth
backend would run another instance of the same code rather than another code
path. `test/gate_test.cc` is what actually asserts the accuracy, in ulp against
a double reference - the golden suites quantize to INT8 and would pass whatever
the approximation did. In the default build the same file asserts the opposite:
that each wrapper is bit-identical to libm. Switching variant does not need a
`clean` - an object file carries no record of which one built it, so the
Makefile keeps a stamp named for the variant and makes every rule depend on
it, rather than leaving a stale object to report a green run of a build that
never happened.

All six run the same golden vectors. Five of them also run a fast-path gate as
a fifth binary - `test/simd_gate.cc` for `sse2` and `neon`, and
`test/cortexm_gate.cc`, `test/esp_gate.cc`, `test/helium_gate.cc` for the
cross-compiled ones. They exist because a vector body that is never entered
still produces the right answer through the scalar remainder underneath it, so
every suite stays green with no accelerated instruction executed. That is not
hypothetical: `esp` once reached its accelerated matvec 6 times in 76 suite
calls, by linker accident, with every gate green. So each accelerated backend
counts which body every call took - behind a compile-time flag the test build
sets and the shipping build does not, and referenced unconditionally by the
gate, so a lost define is a link error rather than a check that stopped
checking - and its gate fails unless every call took the body its shape
dictates and matched the scalar bodies in `src/xlstm_simd_scalar.h`.

Those comparisons are bit-exact wherever the backend is: `helium` in all four
kernels, `sse2` and `neon` in the INT8 matvec, the rank-1 update and `vecmat`.
Two are not, and are held to the standard summation error bound instead -
`sse2`/`neon` `matvec_f32`, whose four lane accumulators regroup the sum, and
`cortexm` `matvec_f32`, whose `fmaf` rounds once per term where the scalar body
rounds twice. Both differences are what those bodies are for; a tolerance is
the honest comparison there and is stated as such rather than applied quietly
everywhere.

`test-cortexm` gates the Cortex-M backend without a board: `SXTAB16` and
`SMLAD` are ARMv6 DSP instructions that A-profile also has, so the kernels
cross-compile for `armv7-a`. It gates the arithmetic and only the arithmetic:

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

`test-helium` is the same shape again for the `helium` backend, on an emulated
Cortex-M55 (`qemu-system-arm -M mps3-an547`). Both halves come from the
distribution archive - `gcc-arm-none-eabi` and upstream `qemu-system-arm` - so
there is nothing to download and no version to pin. The bare-metal side is two
small files instead of a specs file, because the Arm toolchain ships no board
support: `test/helium_boot.c` (vector table, the CPACR write that switches the
vector unit on, and a semihosting `SYS_EXIT_EXTENDED` so `main`'s return value
becomes the exit status) and `test/helium.ld` (the AN547 memory map).

`test/helium_gate.cc` asserts three things the golden vectors cannot. That
every call took the vector body its shape dictates, at every alignment of every
operand. That a size which is not a multiple of the vector width ends in a
narrowed vector pass rather than a scalar remainder - each kernel reports that
separately, so a tail creeping back in fails the run. And that no kernel
touches a byte outside its operands: one operand at a time is butted against
the end of mapped memory, where an over-read faults instead of quietly
succeeding, which is the only way to see a load whose lanes are discarded.
Comparisons there are bit-exact and not toleranced, for the f32 kernels as much
as the INT8 one: this backend reassociates nothing and contracts nothing.

QEMU executes MVE architecturally - leave CP10/CP11 at their reset value and
the first vector instruction takes a UsageFault instead - but it models no
timing. Nothing in this gate is a performance claim, and the gap is easy to
misread here: `matvec_f32` buys its bit-exactness with a gather load, which is
several beats on a Cortex-M55 where a contiguous 128-bit load is two.

## The performance gate

```bash
make perf              # needs valgrind
make perf-baseline     # re-record it, deliberately
```

`make bench` prints wall-clock, which no shared runner reproduces closely
enough to fail a build on. `make perf` counts retired instructions under
callgrind instead, collection toggled on one kernel entry point at a time. It
covers both `XLSTM_GATES` builds, including the f32 kernels in both - the
switch reaches all four kernels, so an f32 pair that came back equal would say
it had stopped reaching them. The
same binary on the same input gives the same count every run, so a move against
`test/perf_baseline.txt` is a real change in work done, and CI fails on it. A
deliberate change is a `make perf-baseline` and a one-line-per-case diff.

It fails in **both** directions: a regression beyond +2%, and an improvement
beyond -5%. The second half is not pedantry - an unrecorded win leaves an
over-generous baseline that a later regression can then hide under, so the gate
would quietly loosen itself. The two numbers differ on purpose: a false
regression blocks work that did nothing wrong, whereas a false improvement
costs one `make perf-baseline`, which is the right thing to run whenever the
counts genuinely moved. 5% clears the largest environment effect ever measured
on these loops - the 2.8% the libm implementation choice was worth before the
gate pinned it - and sits far below any real win, `sse2` beating `ref` by 34%
to 59% across this table.

Two limits, worth knowing before trusting a green run:

- **It is a proxy for time, not time.** A change that leaves the instruction
  count alone and worsens cache behaviour passes. Not hypothetical: a change
  with identical instruction counts cost 10% on one Cortex-M part, and a
  smaller binary measured slower on three.
- **Host backends only** (`ref`, `sse2`). `cortexm` and `esp` performance is a
  property of those cores and is measured on hardware.

Counts are specific to the compiler that produced them, so the gate refuses to
compare across a toolchain it did not record.

`make bench` rebuilds from scratch before it measures anything, then refuses to
print a number unless the binary names the backend just built. Objects are
backend-specific; their filenames are not, so `build/xlstm_simd.o` left by an
earlier `make test-ref` is newer than `src/xlstm_simd_sse2.c` and make finds it
up to date - without that rebuild, an auto-detected `make bench` benchmarks
`ref`.

## Hardware results

`bench/results/*.jsonl` is one file per board and gate build, straight from the
boards. Lines are `XLSTM_PROVENANCE` (what was built and at what clock),
`XLSTM_TIMING` (one per kernel and size), `XLSTM_TIMING_ENV` (sampling), and
`XLSTM_XIPDIAG` (weights in flash versus SRAM, on the two boards with room for
it). Nothing needs the harness to read: each timing line carries its own
`macs_per_call`, so a comparison can be checked for equal work, and `exec_from`,
so flash-bound rows cannot be mistaken for compute.

Timings are the minimum over 17 samples of 8 calls. Repeat runs of one build
move by about 1%, so results are quoted to two significant figures. The RP2350
times off a 4 MHz counter, 250 ns a tick, which quantizes its smallest cases;
its rows are SRAM-resident, since executing from XIP flash times flash
bandwidth rather than the kernel.

The CMSIS-NN rows compare `slstm_step_s8` against `arm_lstm_unidirectional_s8`
at identical `macs_per_call`, but they are not the same model: sLSTM carries two
extra states and a log-space stabilizer, and reads 6% to 33% more bytes per
call. Read it as the cost of stabilized exponential gating against a mature
vendor LSTM, not as one implementation of the same thing beating another.

These are auditable, not reproducible: re-deriving them needs the same boards.
The harness that produced them is not part of this repository.

## Changing a tolerance, a bound, or the generator

```bash
make mutants           # ~90s for the host pair, ~3 min for every backend
                       # whose toolchain is installed. Edits the working tree
                       # and restores it.
```

Those changes fail by making a gate quietly stop failing, which a green
`make test` cannot show you. `make mutants` injects the defects the bounds
exist to catch - an activation drift, a zeroed exit state, a state
requantization drift, a dropped zero point, a matvec that skips its SIMD tail,
a single corrupted channel, a vector body no call ever enters - rebuilds, and
asserts the gates fail. A defect they no longer notice is an escape, and fails
the target. Every mutation records WHICH assertion has to catch it, so a check
that quietly stopped firing behind a neighbour that still fires is a failure
too.

The last of those defect classes is the one worth naming: forcing a vector
body unreachable leaves every answer intact, because the scalar remainder
computes the whole row. Fourteen entries inject exactly that, one per
accelerated body across the five accelerated backends, and each is recorded
against the fast-path gate that catches it. One is not: losing the vector body
of `sse2`/`neon` `matvec_f32` also changes the summation order, and the f32
goldens turn out to be tight enough to see that - by 4.6e-05 against their own
bound - so the suites fail first. That is luck rather than design, and it holds
for that one body only.

One mutation must **pass**: a 0.1% activation drift. That is the portability
margin the INT8 bounds are derived with, so that a backend whose sigmoid and
tanh are approximations rather than libm - a CMSIS-NN lookup table, say - is
admitted rather than failed. Bounds tight enough to catch it would reject
legitimate backends. The same drift at 0.2% must fail, which is what keeps the
margin a margin rather than a hole.

Each mutation also records **which** assertion must catch it, and one caught by
a different assertion fails the run as `WRONG CHECK`. Otherwise a check could
quietly stop firing while a neighbour still catches the mutation, and the
battery would report green over a blind check - the very loosening it exists to
detect. The recorded signatures say what actually catches what rather than what
one would like to: the two single-channel mutations are caught by the
exit-state checks, not by the per-channel output bound they were written for,
because a corrupted channel feeds back through `c` and `n` before the output
path sees it.

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
