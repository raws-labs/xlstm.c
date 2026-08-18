CC      := gcc
CXX     := g++
CFLAGS  := -std=c99 -O2 -Wall -Wextra
CXXFLAGS:= -std=c++17 -O2 -Wall -Wextra

BUILD   := build
VENV    := .venv/bin/python3

# --- SIMD backend selection ---
# XLSTM_SIMD: auto (default), ref, sse2, neon, esp, cortexm
#
# auto never picks esp or cortexm: both need a cross toolchain and neither can
# run on the build host, so they are opt-in rather than part of `make test`.
# Both are gated under emulation instead - cortexm by test-cortexm, esp by
# test-esp, both below.
XLSTM_SIMD ?= auto

ifeq ($(XLSTM_SIMD),auto)
  SIMD_PROBE := $(shell $(CC) -dM -E - < /dev/null)
  ifneq (,$(findstring __SSE2__,$(SIMD_PROBE)))
    XLSTM_SIMD_IMPL := sse2
    SIMD_CFLAGS := -msse2
  else ifneq (,$(findstring __ARM_NEON,$(SIMD_PROBE)))
    XLSTM_SIMD_IMPL := neon
    SIMD_CFLAGS :=
  else
    XLSTM_SIMD_IMPL := ref
    SIMD_CFLAGS :=
  endif
else
  XLSTM_SIMD_IMPL := $(XLSTM_SIMD)
  ifeq ($(XLSTM_SIMD),sse2)
    SIMD_CFLAGS := -msse2
  else
    SIMD_CFLAGS :=
  endif
endif

.PHONY: all test simd-info test-ref test-sse2 test-neon test-cortexm test-esp reference clean \
        test-docker-ort test-docker-tvm test-docker-tflm test-docker-espdl \
        bench bench-ref bench-sse2 perf perf-baseline mutants \
        check-internal-refs

all: $(BUILD)/slstm.o $(BUILD)/mlstm.o \
     $(BUILD)/xlstm_quant.o $(BUILD)/slstm_s8.o $(BUILD)/mlstm_s8.o \
     $(BUILD)/xlstm_simd.o

$(BUILD):
	@mkdir -p $@

# --- SIMD kernel object ---

$(BUILD)/xlstm_simd.o: src/xlstm_simd_$(XLSTM_SIMD_IMPL).c include/xlstm_simd.h \
    src/xlstm_simd_scalar.h | $(BUILD)
	@$(CC) $(CFLAGS) $(SIMD_CFLAGS) -Iinclude -c $< -o $@

# --- Core objects ---

$(BUILD)/slstm.o: src/slstm.c include/slstm.h include/xlstm_util.h include/xlstm_simd.h | $(BUILD)
	@$(CC) $(CFLAGS) -Iinclude -c $< -o $@

$(BUILD)/mlstm.o: src/mlstm.c include/mlstm.h include/xlstm_util.h include/xlstm_simd.h | $(BUILD)
	@$(CC) $(CFLAGS) -Iinclude -c $< -o $@

# --- Quantized objects ---

$(BUILD)/xlstm_quant.o: src/xlstm_quant.c include/xlstm_quant.h | $(BUILD)
	@$(CC) $(CFLAGS) -Iinclude -c $< -o $@

$(BUILD)/slstm_s8.o: src/slstm_s8.c include/slstm_s8.h include/xlstm_quant.h include/xlstm_util.h include/xlstm_simd.h | $(BUILD)
	@$(CC) $(CFLAGS) -Iinclude -c $< -o $@

$(BUILD)/mlstm_s8.o: src/mlstm_s8.c include/mlstm_s8.h include/xlstm_quant.h include/xlstm_util.h include/xlstm_simd.h | $(BUILD)
	@$(CC) $(CFLAGS) -Iinclude -c $< -o $@

# --- Core tests ---

$(BUILD)/slstm_test: test/slstm_test.cc $(BUILD)/slstm.o $(BUILD)/xlstm_simd.o include/slstm.h test/reference_data.h | $(BUILD)
	@$(CXX) $(CXXFLAGS) -Iinclude -Itest -o $@ $< $(BUILD)/slstm.o $(BUILD)/xlstm_simd.o -lm

$(BUILD)/mlstm_test: test/mlstm_test.cc $(BUILD)/mlstm.o $(BUILD)/xlstm_simd.o include/mlstm.h test/reference_data.h | $(BUILD)
	@$(CXX) $(CXXFLAGS) -Iinclude -Itest -o $@ $< $(BUILD)/mlstm.o $(BUILD)/xlstm_simd.o -lm

# --- Quantized tests ---

$(BUILD)/slstm_s8_test: test/slstm_s8_test.cc $(BUILD)/slstm_s8.o $(BUILD)/xlstm_quant.o $(BUILD)/xlstm_simd.o include/slstm_s8.h test/reference_data.h | $(BUILD)
	@$(CXX) $(CXXFLAGS) -Iinclude -Itest -o $@ $< $(BUILD)/slstm_s8.o $(BUILD)/xlstm_quant.o $(BUILD)/xlstm_simd.o -lm

$(BUILD)/mlstm_s8_test: test/mlstm_s8_test.cc $(BUILD)/mlstm_s8.o $(BUILD)/xlstm_quant.o $(BUILD)/xlstm_simd.o include/mlstm_s8.h test/reference_data.h | $(BUILD)
	@$(CXX) $(CXXFLAGS) -Iinclude -Itest -o $@ $< $(BUILD)/mlstm_s8.o $(BUILD)/xlstm_quant.o $(BUILD)/xlstm_simd.o -lm

TEST_BINS := $(BUILD)/slstm_test $(BUILD)/mlstm_test \
             $(BUILD)/slstm_s8_test $(BUILD)/mlstm_s8_test

# The esp backend's fast-path checks - the fifth binary of test-esp below.
# -Isrc so it compares the accelerated bodies against src/xlstm_simd_scalar.h
# itself rather than a copy. Buildable only with the xtensa toolchain.
$(BUILD)/esp_gate: test/esp_gate.cc $(BUILD)/xlstm_simd.o src/xlstm_simd_scalar.h | $(BUILD)
	@$(CXX) $(CXXFLAGS) -Iinclude -Isrc -o $@ $< $(BUILD)/xlstm_simd.o -lm

test: $(TEST_BINS)
	@$(BUILD)/slstm_test
	@$(BUILD)/mlstm_test
	@$(BUILD)/slstm_s8_test
	@$(BUILD)/mlstm_s8_test

# --- SIMD convenience targets ---

simd-info:
	@echo "SIMD backend: $(XLSTM_SIMD_IMPL)"

test-ref:
	@$(MAKE) clean
	@$(MAKE) test XLSTM_SIMD=ref

test-sse2:
	@$(MAKE) clean
	@$(MAKE) test XLSTM_SIMD=sse2


# Build only, then run under the emulator explicitly. Invoking the `test`
# target here would execute aarch64 binaries on the build host, which appears
# to work wherever binfmt_misc is registered and fails everywhere else.
test-neon:
	@$(MAKE) clean
	@$(MAKE) $(TEST_BINS) XLSTM_SIMD=neon \
		CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ \
		CFLAGS="-std=c99 -O2 -Wall -Wextra -static" \
		CXXFLAGS="-std=c++17 -O2 -Wall -Wextra -static"
	@qemu-aarch64 $(BUILD)/slstm_test
	@qemu-aarch64 $(BUILD)/mlstm_test
	@qemu-aarch64 $(BUILD)/slstm_s8_test
	@qemu-aarch64 $(BUILD)/mlstm_s8_test
	@# Leave no foreign binaries behind. Where binfmt_misc is registered, a
	@# later host `make test` finds these up to date and RUNS them under the
	@# emulator, reporting a pass for a backend it never built.
	@$(MAKE) clean

# The cortexm backend's DSP arithmetic, on armv7-a under emulation. SXTAB16 and
# SMLAD are ARMv6 DSP instructions that A-profile has too, so `armv7-a+fp`
# satisfies the backend's __ARM_FEATURE_SIMD32 guard and runs the whole suite
# with no board and no bare-metal harness. (+fp because a bare -march=armv7-a
# drops the FPU this hard-float toolchain needs.)
#
# It gates the arithmetic - the SXTAB16 + SMLAD INT8 path and the blocked f32
# path, against the same golden vectors as every other backend - and nothing
# else. Green here is NOT a Cortex-M gate:
#
#   - Alignment. armv7-a Linux permits unaligned word access, so this build
#     defines __ARM_FEATURE_UNALIGNED and takes the memcpy spelling of the INT8
#     matvec's unaligned group load. The byte-assembly spelling that firmware
#     built -mno-unaligned-access takes is never compiled here, and M-profile
#     alignment behaviour is not exercised at all. Genuine blind spot.
#   - Numerics on FPv5. XLSTM_FPU_HAS_MINMAX_ROUND resolves to 0 here (gcc
#     defines no __ARM_FEATURE_NUMERIC_MAXMIN for armv7-a), so this runs the
#     portable explicit min/max/round path, as a Cortex-M4 does. The vminnm /
#     vrinta path M7 and M33 declare is not covered.
#   - Timing. Emulated instruction execution says nothing about cycles.
#
# Build only, then invoke the emulator explicitly - same reason as test-neon.
test-cortexm:
	@$(MAKE) clean
	@$(MAKE) $(TEST_BINS) XLSTM_SIMD=cortexm \
		CC=arm-linux-gnueabihf-gcc CXX=arm-linux-gnueabihf-g++ \
		CFLAGS="-std=c99 -O2 -Wall -Wextra -static -march=armv7-a+fp" \
		CXXFLAGS="-std=c++17 -O2 -Wall -Wextra -static -march=armv7-a+fp"
	@qemu-arm $(BUILD)/slstm_test
	@qemu-arm $(BUILD)/mlstm_test
	@qemu-arm $(BUILD)/slstm_s8_test
	@qemu-arm $(BUILD)/mlstm_s8_test
	@# Same reason as test-neon above: do not leave armhf binaries in build/.
	@$(MAKE) clean

# The esp backend, on an emulated ESP32-S3. Xtensa has no Linux userspace, so
# this is system emulation rather than qemu-user - the only way it differs in
# shape from the two gates above. It cross-compiles the same four suites,
# unchanged, and runs each as its own image, plus a fifth for the fast-path
# checks in test/esp_gate.cc.
#
# It carries no startup code, no linker script and no semihosting shim: two
# specs files that ship with the xtensa toolchain supply all of it.
#
#   sim.elf.specs   crt1-sim.o and the esp32s3 memory-map linker scripts -
#                   reset and window vectors, .bss, .init_array, main, exit,
#                   and a handler that prints PANIC and exits non-zero on an
#                   unhandled exception instead of spinning.
#   sys.qemu.specs  libsys_qemu - newlib's _write and _exit on the Xtensa
#                   simcall instruction. That is what makes these ordinary
#                   test binaries: printf reaches the host, and main's return
#                   value becomes the emulator's process exit status.
#
# --defsym=entire_dram_seg=1 widens the linker script's default 64 KB data
# region to the S3's full 448 KB, which the ~170 KB of golden vectors each
# suite pulls in needs. XLSTM_TEST_MAX_H=64 bounds the runners' own static
# buffers (test/test_config.h): the 256 default does not fit either, and 64
# covers every case in test/reference_data.h. Neither touches XLSTM_MAX_HIDDEN,
# so the kernels under test are the shipping ones.
#
# Needs xtensa-esp32s3-elf-g++ and Espressif's qemu-system-xtensa on PATH,
# both single tarballs from GitHub releases - no ESP-IDF and no Docker.
# .github/workflows/ci.yml installs them and pins the versions.
#
# WHAT IT EXERCISES, precisely - a green run here is a narrow claim:
#
#   - xlstm_matvec_f32, both of its paths. Its blocked body - four rows at a
#     time, four columns per EE.LDF.128.IP - is entered on rows and cols
#     alone: 7 or more columns and at least one whole block of rows.
#     test/esp_gate.cc runs 15 shapes at all four alignments of each operand
#     and fails the run unless each took the path its shape dictates and
#     matched xlstm_scalar_matvec_f32 bit for bit.
#   - xlstm_matvec_s8, both of its paths. Its vector body - 16 columns per
#     EE.VMULAS.S8.ACCX, each group assembled from the two aligned blocks
#     holding it - is entered on cols and the zero point alone, never on
#     alignment, so a vector-body call runs every column of every row there.
#     The gate runs 23 cases at all 256 pairings of the two operands'
#     alignments and demands bit-identical int32 output, which for integers
#     is the whole of correctness - there is no tolerance to hide in.
#   - xlstm_rank1_update_f32, both of its paths. Four rows at a time through
#     EE.LDF.128.IP and EE.STF.128.IP, entered on H alone: seven columns, no
#     alignment condition, because a row a four-row block cannot reach takes
#     its own prefix as a block of one. The gate runs 15 shapes at 64
#     alignment triples and two gate pairs and demands bit-identical output.
#     MADD.S does not round the product it adds, so equality here is what
#     detects the vector body and the scalar body contracting differently -
#     a tolerance would swallow precisely that.
#   - xlstm_vecmat_f32, both of its spellings. Four columns of out[] held in
#     registers across the row loop either way; the 128-bit load of M on top
#     wherever one column boundary can be aligned for every row at once,
#     which is a cols divisible by four. 19 shapes at 64 alignment triples,
#     bit-exact, with out[] seeded non-zero so that dropping the accumulator
#     seed - the one change that moved a golden here before - cannot pass.
#   - The four golden-vector suites, 39 assertions, against this backend on
#     this core. They are the same binaries the other gates run, built from
#     the same sources with no test-side change - one image each, because
#     each pulls in its own copy of the golden vectors and four of those do
#     not fit in the S3's memory at once. The gate binary reports its own
#     four fast-path splits rather than asserting them; the four checks above
#     are the assertion.
#
# What it does NOT cover: real silicon. QEMU executes the S3's SIMD
# instructions but does not model their timing, its Xtensa FPU is not
# guaranteed bit-identical to the part, and nothing here says anything about
# cycles.
#
# -ffp-contract=off is load-bearing and belongs on BOTH dialects. The gate
# compares the accelerated bodies (C) against the scalar bodies it inlines from
# src/xlstm_simd_scalar.h (C++), bit for bit. gcc 14 defaults the two dialects
# differently - strict-ISO C does not contract, C++ does - so leaving it
# implicit compiles one side with MADD.S and the other with MUL.S + ADD.S and
# the gate fails on a last-bit difference that is the flags' doing and not the
# kernel's. Off, not fast, because that is what the arithmetic the goldens were
# gated against does; a contracted build also passes, and would be the setting
# to move to if this ever needs to gate a -O2 GNU-dialect firmware.
ESP_DEFS := -DXLSTM_TEST_MAX_H=64 -DXLSTM_ESP_FASTPATH_COUNTERS \
            -ffp-contract=off
ESP_LINK := -specs=sim.elf.specs -specs=sys.qemu.specs \
            -Wl,--defsym=entire_dram_seg=1
# timeout because an infinite loop in a system emulator never returns, unlike
# the qemu-user gates above; a hang has to be red rather than a stuck runner.
ESP_RUN  := timeout 300 qemu-system-xtensa -M esp32s3 -semihosting \
            -display none -serial none -monitor none -kernel

test-esp:
	@$(MAKE) clean
	@$(MAKE) $(TEST_BINS) $(BUILD)/esp_gate XLSTM_SIMD=esp \
		CC=xtensa-esp32s3-elf-gcc CXX=xtensa-esp32s3-elf-g++ \
		CFLAGS="-std=c99 -O2 -Wall -Wextra $(ESP_DEFS)" \
		CXXFLAGS="-std=c++17 -O2 -Wall -Wextra $(ESP_DEFS) $(ESP_LINK)"
	@$(ESP_RUN) $(BUILD)/esp_gate
	@$(ESP_RUN) $(BUILD)/slstm_test
	@$(ESP_RUN) $(BUILD)/mlstm_test
	@$(ESP_RUN) $(BUILD)/slstm_s8_test
	@$(ESP_RUN) $(BUILD)/mlstm_s8_test
	@# Same reason as test-neon above: do not leave foreign binaries in build/.
	@$(MAKE) clean

# --- Docker integration tests ---

test-docker-ort:
	docker build -f test/adapters/onnxruntime/Dockerfile -t xlstm-test-ort .
	docker run --rm xlstm-test-ort

test-docker-tvm:
	docker build -f test/adapters/microtvm/Dockerfile -t xlstm-test-tvm .
	docker run --rm xlstm-test-tvm

test-docker-tflm:
	docker build -f test/adapters/tflm/Dockerfile -t xlstm-test-tflm .
	docker run --rm xlstm-test-tflm

test-docker-espdl:
	docker build -f test/adapters/esp-dl/Dockerfile -t xlstm-test-espdl .
	docker run --rm xlstm-test-espdl

# --- Benchmark ---

$(BUILD)/xlstm_bench: test/xlstm_bench.cc $(BUILD)/slstm.o $(BUILD)/mlstm.o \
    $(BUILD)/slstm_s8.o $(BUILD)/mlstm_s8.o \
    $(BUILD)/xlstm_quant.o $(BUILD)/xlstm_simd.o | $(BUILD)
	@$(CXX) $(CXXFLAGS) -Iinclude -o $@ $< \
		$(BUILD)/slstm.o $(BUILD)/mlstm.o \
		$(BUILD)/slstm_s8.o $(BUILD)/mlstm_s8.o \
		$(BUILD)/xlstm_quant.o $(BUILD)/xlstm_simd.o -lm

# Objects are backend-specific; their names are not. build/xlstm_simd.o is
# compiled from src/xlstm_simd_$(impl).c, so one left behind by `make test-ref`
# is newer than src/xlstm_simd_sse2.c and make finds it up to date: an
# auto-detected `make bench` then links ref and reports ref while everything
# else in this file says sse2. Measured, not suspected. perf-measure already
# rebuilds from scratch for exactly this reason - do the same here, since bench
# is the target a human runs when they want a number.
#
# Then check it rather than trust it: the binary names its own backend, so a
# number is only printed once that name matches the backend just built. Via a
# file rather than a pipe, so the run's own exit status survives - `| tee` would
# report success for a benchmark that died halfway with its header already out.
bench: | $(BUILD)
	@rm -f $(BUILD)/*.o $(BUILD)/xlstm_bench
	@$(MAKE) --no-print-directory $(BUILD)/xlstm_bench XLSTM_SIMD=$(XLSTM_SIMD_IMPL)
	@$(BUILD)/xlstm_bench > $(BUILD)/bench.txt; rc=$$?; \
	cat $(BUILD)/bench.txt; \
	[ $$rc -eq 0 ] || { echo "bench: the benchmark exited $$rc" >&2; exit $$rc; }; \
	grep -q 'backend: $(XLSTM_SIMD_IMPL)$$' $(BUILD)/bench.txt || { \
		echo "bench: built $(XLSTM_SIMD_IMPL), but the binary reports:" >&2; \
		head -1 $(BUILD)/bench.txt >&2; \
		echo "Refusing to report a number for a backend that did not run." >&2; \
		exit 1; }

bench-ref:
	@$(MAKE) bench XLSTM_SIMD=ref

bench-sse2:
	@$(MAKE) bench XLSTM_SIMD=sse2

# --- Performance gate ---
#
# `bench` prints wall-clock, which a shared CI runner cannot reproduce closely
# enough to fail a build on. This gate counts retired instructions instead:
# callgrind with collection toggled on the kernel entry point, so the number is
# that kernel's inclusive cost and nothing else - no process startup, no setup,
# no timing calls. The same binary on the same input yields the same count on
# every run, so a 2% move is signal.
#
# Limits, which are real and which a green run here does NOT cover:
#   - Instructions are a proxy for time, not time. A change that leaves the
#     count alone and worsens cache or memory behaviour passes this gate. That
#     is not hypothetical: a change with identical instruction counts has cost
#     10% on a Cortex-M part, and a smaller binary has measured slower.
#   - Host backends only (ref, sse2). cortexm and esp performance is a property
#     of those cores and is measured on hardware, not here.
#
# The kernels call expf/tanhf/logf, and glibc binds those to an FMA or a plain
# SSE implementation depending on the CPU it finds. Measured, that choice alone
# moves a count by up to 2.8%, which is enough to fail a tight gate on nothing
# but a change of runner. Pinning the tunable to the implementations every
# x86-64 has makes the count independent of which machine picked up the job.
# The counts stay inclusive of libm, so trading a libm call for hand-rolled
# arithmetic still scores as the win or loss it actually is.
PERF_ENV := GLIBC_TUNABLES=glibc.cpu.hwcaps=-FMA,-AVX2,-AVX

# Two bounds, and the gate fails on either. Instruction counts are exact for a
# given binary, so neither covers measurement noise - there is none. What is
# left to absorb is drift the toolchain check above cannot see: it compares
# `gcc --version` only, so a glibc update that reshapes expf/tanhf moves these
# counts, inclusive of libm as they are, with the recorded line still matching.
#
# PERF_TOL, on the regression side, stays at 2%.
#
# PERF_TOL_FAST bounds the other direction, because a gate that only fails
# upward loosens itself: an unrecorded win leaves an over-generous baseline
# that a later regression can hide under. 5% is chosen against the one
# environment effect ever measured on these loops - the libm implementation
# choice PERF_ENV now pins, worth up to 2.8% - so no drift of that scale can
# fire it on its own, while every real win here is far larger: sse2 against ref
# is 34% to 59% across this table, and nothing smaller than 5% is worth the
# re-record it asks for. Asymmetric on purpose: a false regression blocks work
# that did nothing wrong, whereas a false improvement costs one
# `make perf-baseline`, which is the right action whenever counts truly moved.

PERF_BASELINE := test/perf_baseline.txt
PERF_BACKENDS ?= ref sse2
PERF_KERNELS  ?= slstm_f32 mlstm_f32 slstm_s8 mlstm_s8
PERF_WIDTHS   ?= 16 64
PERF_STEPS    ?= 200
PERF_TOL      ?= 2.0
PERF_TOL_FAST ?= 5.0
VALGRIND      ?= valgrind

# Emit "backend kernel H steps instructions" for every case. Rebuilds the bench
# once per backend; only xlstm_simd.o actually differs, but the objects are
# cheap and a stale one would silently measure the wrong backend.
define perf-measure
	command -v $(VALGRIND) >/dev/null 2>&1 || { \
		echo "perf: $(VALGRIND) not found - apt-get install valgrind" >&2; exit 1; }; \
	for b in $(PERF_BACKENDS); do \
		rm -f $(BUILD)/*.o $(BUILD)/xlstm_bench; \
		$(MAKE) --no-print-directory $(BUILD)/xlstm_bench XLSTM_SIMD=$$b >/dev/null; \
		for k in $(PERF_KERNELS); do \
			sym=$${k%%_*}_step_$${k#*_}; \
			for h in $(PERF_WIDTHS); do \
				ir=$$($(PERF_ENV) $(VALGRIND) --tool=callgrind \
					--callgrind-out-file=/dev/null \
					--collect-atstart=no --toggle-collect=$$sym \
					$(BUILD)/xlstm_bench $$k $$h $(PERF_STEPS) 2>&1 \
					| sed -n 's/.*Collected *: *//p'); \
				test -n "$$ir" || { echo "perf: no count for $$b $$k $$h" >&2; exit 1; }; \
				echo "$$b $$k $$h $(PERF_STEPS) $$ir"; \
			done; \
		done; \
	done
endef

# The recorded toolchain is load-bearing, not decorative. gcc and clang differ
# by up to 50% on these loops, so comparing across them measures the compiler
# and not the change. Worse, clang against a gcc baseline reads as "faster
# everywhere" and passes - a gate that cannot fail. Refuse rather than mislead.
perf:
	@mkdir -p $(BUILD)
	@rec=$$(sed -n 's/^# toolchain: //p' $(PERF_BASELINE)); \
	cur=$$($(CC) --version | head -1); \
	if [ "$$rec" != "$$cur" ]; then \
		echo "perf: toolchain differs from the one the baseline was recorded with."; \
		echo "  baseline: $$rec"; \
		echo "  current:  $$cur"; \
		echo "Counts are compiler-specific, so this comparison would not mean anything."; \
		echo "Build with the recorded compiler, or re-record with: make perf-baseline"; \
		exit 1; \
	fi
	@$(perf-measure) > $(BUILD)/perf.txt
	@awk -v tol=$(PERF_TOL) -v fast=$(PERF_TOL_FAST) ' \
	BEGIN { \
	  printf "%-7s %-10s %4s %6s %14s %14s %9s\n", \
	    "backend","kernel","H","steps","baseline","current","delta"; \
	  printf "%s\n", "---------------------------------------------------------------------------"; \
	} \
	NR == FNR { if (NF == 5 && $$1 !~ /^#/) base[$$1" "$$2" "$$3" "$$4] = $$5; next } \
	{ \
	  key = $$1" "$$2" "$$3" "$$4; \
	  if (!(key in base)) { \
	    printf "%-7s %-10s %4s %6s %14s %14d    NO BASELINE\n", $$1,$$2,$$3,$$4,"-",$$5; \
	    fail = 1; next; \
	  } \
	  seen[key] = 1; d = 100.0 * ($$5 - base[key]) / base[key]; \
	  tag = (d > tol) ? "  REGRESSED" \
	      : ((d < -fast) ? "  IMPROVED - RE-RECORD" \
	      : ((d < -tol) ? "  faster" : "")); \
	  if (d > tol || d < -fast) fail = 1; \
	  printf "%-7s %-10s %4s %6s %14d %14d %+8.2f%%%s\n", \
	    $$1,$$2,$$3,$$4,base[key],$$5,d,tag; \
	} \
	END { \
	  for (k in base) if (!(k in seen)) { printf "not measured: %s\n", k; fail = 1 } \
	  printf "\ninstructions (callgrind Ir), tolerance +%s%% / -%s%%\n", tol, fast; \
	  printf "proxy for time, blind to cache behaviour; host backends only,\n"; \
	  printf "not cortexm or esp - those are measured on hardware.\n"; \
	  if (fail) { \
	    printf "\nperf: FAILED - moved beyond tolerance, or baseline out of sync.\n"; \
	    printf "A large improvement fails too, deliberately: an unrecorded win leaves\n"; \
	    printf "an over-generous baseline for a later regression to hide under.\n"; \
	    printf "If the change is deliberate, re-record with: make perf-baseline\n"; \
	    exit 1; \
	  } \
	  printf "\nperf: OK\n"; \
	}' $(PERF_BASELINE) $(BUILD)/perf.txt

# Deliberate, reviewable baseline update: one line per case, so a legitimate
# change shows up as a readable diff rather than an opaque blob.
perf-baseline:
	@mkdir -p $(BUILD)
	@$(perf-measure) > $(BUILD)/perf.txt
	@{ \
	  echo "# xlstm.c performance baseline - retired instruction counts (callgrind Ir)."; \
	  echo "#"; \
	  echo "# Regenerate deliberately with:  make perf-baseline"; \
	  echo "# Checked by:                    make perf   (tolerance +$(PERF_TOL)% / -$(PERF_TOL_FAST)%)"; \
	  echo "#"; \
	  echo "# Counts are the inclusive cost of one kernel entry point over $(PERF_STEPS) steps,"; \
	  echo "# collection toggled on that symbol alone. Exact and reproducible for a given"; \
	  echo "# binary, so any movement here is a real change in work done, not noise."; \
	  echo "# That cuts both ways: make perf fails on a large improvement too, because an"; \
	  echo "# unrecorded win leaves a baseline generous enough to hide a later regression."; \
	  echo "#"; \
	  echo "# These numbers describe the HOST backends only. They say nothing about cortexm"; \
	  echo "# or esp, whose performance is a property of those cores and is measured on"; \
	  echo "# hardware. They are also a proxy for time and not time itself: a change that"; \
	  echo "# holds the instruction count and worsens cache locality does not appear here."; \
	  echo "#"; \
	  echo "# Counts are specific to the compiler that produced them - gcc and clang differ"; \
	  echo "# by up to 50% on these loops - so make perf refuses to compare across a change"; \
	  echo "# of toolchain. The line below is what it checks."; \
	  echo "#"; \
	  echo "# toolchain: $$($(CC) --version | head -1)"; \
	  echo "# valgrind:  $$($(VALGRIND) --version 2>/dev/null)"; \
	  echo "# libm:      $(PERF_ENV)"; \
	  echo "#"; \
	  echo "# backend kernel     H steps    instructions"; \
	  cat $(BUILD)/perf.txt; \
	} > $(PERF_BASELINE)
	@echo "wrote $(PERF_BASELINE):"
	@grep -v '^#' $(PERF_BASELINE)

# --- Mutation battery ---
#
# Injects known defects into the kernels and asserts the suites fail. The
# failure mode a loosened tolerance produces is a gate that silently stops
# failing, and this is the only thing here that detects it - so run it after
# any change to tolerances, bounds, or generate_reference.py.
#
# Not in CI, deliberately: it edits files in the working tree, which belongs
# in a run someone chose to start. It restores them on exit, on failure and
# on interrupt; a run killed outright leaves .mutants-backup/, and the next
# run restores from that before doing anything else.
#
# Plain python3, stdlib only - no .venv, unlike `reference` above.
# MUTANT_BACKENDS=ref narrows it to one backend; the default runs ref and sse2.
mutants:
	@python3 test/mutants.py $(MUTANT_BACKENDS)

# --- Reference data ---

test/reference_data.h: test/generate_reference.py
	@$(VENV) $<

reference: test/generate_reference.py
	@$(VENV) $<

# --- Public-repo hygiene ---
#
# This repository is public. Tracked files must not reference paths that only
# exist on a maintainer's machine: they leak internal structure and are broken
# references for anyone who clones. Write the fact, not a pointer.

# Two rules, both stated generically so this file names nothing it is looking
# for. A checker that must spell out the paths it forbids defeats its purpose.
#
#   1. No absolute paths into a user's home directory.
#   2. Every .md a tracked file cites must itself be tracked. Citing a document
#      that is not in the repository is either a broken reference for whoever
#      clones it, or a pointer to something that was never meant to ship.

check-internal-refs:
	@fail=0; \
	abs=$$(git ls-files -z | xargs -0 grep -nE '/(home|Users)/[A-Za-z0-9_.-]+/' 2>/dev/null \
	       | grep -v '^Makefile:.*home|Users'); \
	if [ -n "$$abs" ]; then \
		echo "check-internal-refs: absolute path into a home directory:"; \
		echo "$$abs"; fail=1; \
	fi; \
	for ref in $$(git ls-files -z | xargs -0 grep -hoE '\.?[A-Za-z0-9_][A-Za-z0-9_./-]*\.md' 2>/dev/null | sort -u); do \
		git ls-files --error-unmatch "$$ref" >/dev/null 2>&1 || { \
			echo "check-internal-refs: cites an untracked document: $$ref"; \
			git ls-files -z | xargs -0 grep -n "$$ref" 2>/dev/null | head -3; \
			fail=1; }; \
	done; \
	if [ $$fail -ne 0 ]; then \
		echo "Write the fact inline instead of citing a document that is not in the repo."; \
		exit 1; \
	fi; \
	echo "check-internal-refs: OK"

# --- Cleanup ---

clean:
	@rm -rf $(BUILD)
