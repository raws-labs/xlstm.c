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

# The esp backend, on an emulated ESP32-S3. Unlike every other backend gate
# this one needs a full ESP-IDF toolchain and a chip-specific QEMU, so it is
# a container rather than a cross-compile: test/esp/ is an IDF project that
# builds the same four suites into one firmware image, and the image's CMD
# boots it under qemu-system-xtensa and exits with the firmware's verdict.
# ESP-IDF v5.4, because v5.3's bundled QEMU has no esp32s3 machine.
#
# WHAT IT EXERCISES, precisely - a green run here is a narrow claim:
#
#   - xlstm_matvec_f32, both of its paths. Its blocked body - four rows at a
#     time, four columns per EE.LDF.128.IP - is the ONLY accelerated code in
#     src/xlstm_simd_esp.c, and it is entered on rows and cols alone: 7 or
#     more columns and at least one whole block of rows.
#     test/esp/main/esp_gate.cc runs 15 shapes at all four alignments of each
#     operand and fails the run unless each took the path its shape dictates
#     and matched xlstm_scalar_matvec_f32 bit for bit.
#   - xlstm_matvec_s8, xlstm_rank1_update_f32 and xlstm_vecmat_f32 against
#     the golden vectors - but those three are scalar C in this backend, so
#     what is gated there is the arithmetic, not any Xtensa instruction.
#   - The suites' own f32 matvecs: 36 of their 76 xlstm_matvec_f32 calls run
#     blocked, the other 40 being the cases whose H and I are 1 to 4, under
#     one 128-bit group wide. The firmware prints that split every run rather
#     than leaving it to be assumed, and does not assert on it - it is a
#     property of the case list, and pinning it here would gate
#     reference_data.h rather than the kernel.
#
# What it does NOT cover: real silicon. QEMU executes the S3's SIMD
# instructions but does not model their timing, its Xtensa FPU is not
# guaranteed bit-identical to the part, and nothing here says anything about
# cycles.
test-esp:
	docker build -f test/esp/Dockerfile -t xlstm-test-esp .
	docker run --rm xlstm-test-esp

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

bench: $(BUILD)/xlstm_bench
	@$(BUILD)/xlstm_bench

bench-ref:
	@$(MAKE) clean && $(MAKE) bench XLSTM_SIMD=ref

bench-sse2:
	@$(MAKE) clean && $(MAKE) bench XLSTM_SIMD=sse2

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

# Tolerance: instruction counts are exact for a given binary, so this covers
# toolchain drift, not measurement noise. 2% is far tighter than any wall-clock
# gate could hold, and still wide enough to survive a compiler point release.

PERF_BASELINE := test/perf_baseline.txt
PERF_BACKENDS ?= ref sse2
PERF_KERNELS  ?= slstm_f32 mlstm_f32 slstm_s8 mlstm_s8
PERF_WIDTHS   ?= 16 64
PERF_STEPS    ?= 200
PERF_TOL      ?= 2.0
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
	@awk -v tol=$(PERF_TOL) ' \
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
	  tag = (d > tol) ? "  REGRESSED" : ((d < -tol) ? "  faster" : ""); \
	  if (d > tol) fail = 1; \
	  printf "%-7s %-10s %4s %6s %14d %14d %+8.2f%%%s\n", \
	    $$1,$$2,$$3,$$4,base[key],$$5,d,tag; \
	} \
	END { \
	  for (k in base) if (!(k in seen)) { printf "not measured: %s\n", k; fail = 1 } \
	  printf "\ninstructions (callgrind Ir), tolerance +%s%%\n", tol; \
	  printf "proxy for time, blind to cache behaviour; host backends only,\n"; \
	  printf "not cortexm or esp - those are measured on hardware.\n"; \
	  if (fail) { \
	    printf "\nperf: FAILED - regression beyond tolerance, or baseline out of sync.\n"; \
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
	  echo "# Checked by:                    make perf   (tolerance +$(PERF_TOL)%)"; \
	  echo "#"; \
	  echo "# Counts are the inclusive cost of one kernel entry point over $(PERF_STEPS) steps,"; \
	  echo "# collection toggled on that symbol alone. Exact and reproducible for a given"; \
	  echo "# binary, so any movement here is a real change in work done, not noise."; \
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
	for ref in $$(git ls-files -z | xargs -0 grep -hoE '[A-Za-z0-9_][A-Za-z0-9_./-]*\.md' 2>/dev/null | sort -u); do \
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
