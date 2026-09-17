# Contributing

This repository follows the RAWS Labs contributing guide for pull requests, commit
subjects, licensing and the code of conduct:
https://github.com/raws-labs/.github/blob/main/CONTRIBUTING.md. Below is only what is
specific to this repository.

## Getting started

```bash
make test              # core sLSTM + mLSTM tests (requires gcc, g++)
make test-docker-ort   # ONNX Runtime integration test
make test-docker-tvm   # Apache TVM integration test
make test-docker-tflm  # TensorFlow Lite Micro integration test
make test-docker-espdl # ESP-DL integration test (runs on an emulated ESP32-S3)
```

`make test` is fast (seconds). Docker integration tests are slower and require
Docker. On every push to `main` and every pull request CI runs `check-refs` and
`check-tools`; `make test`, `test-ref` and `test-approx` under gcc and clang;
the perf gate; `test-neon`, `test-cortexm`, `test-esp` and `test-helium`
under emulation; and the mutation battery on the host pair. The Docker
integration tests are run locally, not in CI.

Those two commands are the whole of the job that runs them, and `check-refs`
asserts that as a property of the workflow file: a step that CI runs and you
cannot is rejected there. So a green `make check-refs` locally is a green refs
job, including the toolchain image pin, which moves whenever the perf baseline
is re-recorded.

## Workflow

- This repository has no `develop` branch: open pull requests against `main`.
- Run `make test` locally; all core tests must pass. If you touched an adapter, run
  the matching `make test-docker-*` as well.

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

Part of what it regenerates is the INT8 output codes and the INT16 exit state,
which the INT8 suites compare as integers rather than through a bound. Those
integers are where the generator has to be bit-identical to the kernel rather
than merely close, so its quantization is float32 throughout, like
`src/xlstm_quant.c` and unlike the float64 the rest of the replica uses. Nothing
there may move to float64 for convenience: a scale rounded differently sends a
value on a .5 boundary to a different integer, and no tolerance can absorb a
branch.

The integer comparisons are also what covers what a bound cannot reach. An
element whose golden is exactly zero, or whose honest measured error already
spans its own dynamic range, has no bound that is both non-vacuous and free of
false failures; 153 of 5456 exit-state elements are in that position, 118 of
them in `SweepM64`'s C matrix, and 5 of 269 output channels sit inside their own
binding bound. An integer comparison does not ask how large an element is, so it
applies to all of them unchanged. `m` is the one state that keeps no integer,
because it stays float32 in the kernel.

Two of those five output channels stay unasserted and always will: `SweepS64`
ch[15] carries 0.37 of one INT8 code and `SweepM64` ch[63] carries 0.20, so
zeroing either moves no integer. That is the INT8 grid of those cases, not a
gap in the gate, and it is what the runners' `unasserted` note now reports.
Closing it would mean a per-channel `y` scale, which is a change to the
quantization contract rather than to a test.

`make check-tools` matters here because the worked examples in `tools/`
reproduce that file's calibration and shapes from its float tensors alone. If
a quantization convention changes and they are not updated with it, they say
so; they are stdlib-only, so this runs in CI alongside the other checks.

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

Every one of them takes `XLSTM_GATES=exact|approx`, which picks all four
kernels' transcendentals (see the block above `xlstm_gate_expf` in
`include/xlstm_util.h`). `exact` is the default and is what all six are gated
on. `approx` is gated by `make test-approx` on `ref` and the host backend, and
by CI on `cortexm`, which is the target it exists for; the arithmetic is plain
C99 in the shared cell code with no SIMD contract behind it, so a fourth
backend would run another instance of the same code rather than another code
path. `test/gate_test.cc` is what actually asserts the accuracy, in ulp against
a double reference - the golden suites quantize to INT8, and while they do
compare the resulting codes as integers, an approximation has to be wrong by
more than half a code before they see it. In the default build the same file
asserts the opposite:
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
callgrind instead, collection toggled on one kernel entry point at a time, at
H=16, 64 and 128. It
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
gate pinned it - and sits far below any real win, `sse2` beating `ref` by 28%
to 71% across this table.

Two limits, worth knowing before trusting a green run:

- **It is a proxy for time, not time.** A change that leaves the instruction
  count alone and worsens cache behaviour passes. Not hypothetical: a change
  with identical instruction counts cost 10% on one Cortex-M part, and a
  smaller binary measured slower on three.
- **Host backends only** (`ref`, `sse2`). `cortexm` and `esp` performance is a
  property of those cores and is measured on hardware.

callgrind can simulate a data cache, and that was tried to close the first
limit. It does not reproduce across machines: the miss counts move with the
size of the environment block, and fixing that with `env -i` still left +4.23%
and +4.96% between this machine and a CI runner at H=128, against a 2%
tolerance, on a run where the instruction counts were +0.00% on every row. The
note above the gate in the Makefile has the numbers.

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
`macs_per_call`, so a comparison can be checked for equal work, and the RP2350
rows carry `exec_from`, so flash-bound rows cannot be mistaken for compute.

All three Cortex-M boards run with the DATA side of their caches off, and every
timing line says so in its own `dcache` field, read back from the register
rather than assumed. That is the M7's L1 D-cache on the H753, the flash ART
accelerator's data cache on the F446, and nothing at all on the RP2350, which
has no data cache for SRAM. So these are uncached-data figures and a
cross-board cycle comparison is not a comparison of the parts. It matters most
on the M7, where enabling the D-cache is worth 2.66x overall and reverses the
CMSIS-NN comparison above a head width of 8; on the F446 the ART data cache is
worth 1.043x and changes nothing.

Timings are the minimum over 17 samples of 8 calls. Repeat runs of one build
move by about 1%, so results are quoted to two significant figures. The RP2350
times off a 4 MHz counter, 250 ns a tick, which quantizes its smallest cases;
its rows are SRAM-resident, since executing from XIP flash times flash
bandwidth rather than the kernel.

The CMSIS-NN rows compare `slstm_step_s8` against `arm_lstm_unidirectional_s8`
at identical `macs_per_call`, but they are not the same model: sLSTM carries two
extra states and a log-space stabilizer, and reads 6% to 71% more bytes per
call. Read it as the cost of stabilized exponential gating against a mature
vendor LSTM, not as one implementation of the same thing beating another.

These are auditable, not reproducible: re-deriving them needs the same boards.
The harness that produced them is not part of this repository.

## Changing a tolerance, a bound, or the generator

```bash
make mutants           # 71 s for the host pair, 345 s for the five whose
                       # toolchains were installed when that was measured.
                       # Edits the working tree and restores it.
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
computes the whole row. Eighteen entries inject exactly that, one per
accelerated body across the five accelerated backends, and each is recorded
against the fast-path gate that catches it. One is not: losing the vector body
of `sse2`/`neon` `matvec_f32` also changes the summation order, and the f32
goldens turn out to be tight enough to see that - by 4.6e-05 against their own
bound - so the suites fail first. That is luck rather than design, and it holds
for that one body only.

One mutation is about the **bounds not firing**: a 0.1% activation drift, which
is the portability margin the INT8 bounds are derived with, so that a backend
whose sigmoid and tanh are approximations rather than libm - a CMSIS-NN lookup
table, say - is admitted rather than failed. Bounds tight enough to catch it
would reject legitimate backends, so that mutation forbids every bound from
firing. It still fails the run, because the INT8 output codes are also compared
as integers and a 0.1% drift moves integers; both halves are recorded. The same
drift at 0.2% must trip a bound, which is what keeps the margin a margin rather
than a hole.

Each mutation also records **which** assertion must catch it, and one whose own
assertion never fires fails the run as `WRONG CHECK`. Otherwise a check could
quietly stop firing while a neighbour still catches the mutation, and the
battery would report green over a blind check - the very loosening it exists to
detect. The recorded signatures say what actually catches what rather than what
one would like to: the two single-channel mutations are caught by the
exit-state checks, not by the per-channel output bound they were written for,
because a corrupted channel feeds back through `c` and `n` before the output
path sees it.

It covers all six backends: 50 entries, including the loop tails, zero-point
folding, lane order and alignment instances only `neon`, `cortexm`, `esp` and
`helium` compile. A mutation the running backend does not compile reports
`n/a`, and a missing toolchain reports `NOT COVERED`; neither is an escape.
CI runs the host pair on every push and pull request. Those two carry every
entry that exercises a tolerance, a bound or a state comparison, which is the
part that has to be gated on each change; the cross entries exercise the
fast-path counters, and the four emulated jobs already run those gates
unmutated. Run all six locally when you touch a cross backend, and say in the
PR that you did.

The battery edits files in the working tree, which is why it is not part of
`make test`. It restores them on exit, on failure and on interrupt, and a run
killed outright leaves `.mutants-backup/` for the next run to restore from.
