CC      := gcc
CXX     := g++
CFLAGS  := -std=c99 -O2 -Wall -Wextra
CXXFLAGS:= -std=c++17 -O2 -Wall -Wextra

BUILD   := build
VENV    := .venv/bin/python3

# --- SIMD backend selection ---
# XLSTM_SIMD: auto (default), ref, sse2, neon, esp
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

.PHONY: all test simd-info test-ref test-sse2 test-neon reference clean \
        test-docker-ort test-docker-tvm test-docker-tflm test-docker-espdl \
        bench bench-ref bench-sse2 \
        hil-esp32s3-build hil-esp32s3-qemu hil-esp32s3 hil-preflight \
        hil-cortexm-build hil-cortexm-qemu hil-cortexm-m4 hil-cortexm-m7 hil-cortexm-m33 \
        check-internal-refs

all: $(BUILD)/slstm.o $(BUILD)/mlstm.o \
     $(BUILD)/xlstm_quant.o $(BUILD)/slstm_s8.o $(BUILD)/mlstm_s8.o \
     $(BUILD)/xlstm_simd.o

$(BUILD):
	@mkdir -p $@

# --- SIMD kernel object ---

$(BUILD)/xlstm_simd.o: src/xlstm_simd_$(XLSTM_SIMD_IMPL).c include/xlstm_simd.h | $(BUILD)
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

# --- ESP32-S3 hardware-in-the-loop (HIL) ---
# See test/hil/README.md. hil-esp32s3-qemu is the emulated gate: `docker
# run`'s exit code IS the firmware's ##srig-exit:N## verdict, so this is
# CI-safe (a hang or crash fails it, never a silent pass). hil-esp32s3 is
# the real hardware run via SiliconRig and is deliberately NOT wired into
# CI - repository policy is that hardware runs are manual only.

HIL_IMAGE      := xlstm-hil
HIL_BOARD      := esp32-s3
HIL_MERGED_BIN := $(BUILD)/xlstm_hil_merged.bin
HIL_LOG        := $(BUILD)/hil-esp32s3.log

hil-esp32s3-build: | $(BUILD)
	docker build -f test/hil/Dockerfile -t $(HIL_IMAGE) .
	-docker rm -f xlstm-hil-extract >/dev/null 2>&1
	docker create --name xlstm-hil-extract $(HIL_IMAGE)
	docker cp xlstm-hil-extract:/workspace/test/hil/build/xlstm_hil_merged.bin $(HIL_MERGED_BIN)
	docker rm xlstm-hil-extract

# The emulated gate a human runs to check the firmware works before
# touching hardware. `docker run`'s own exit code is the firmware's -
# test/hil/qemu_gate.sh (this image's CMD) boots the firmware headless
# under QEMU and exits with whatever ##srig-exit:N## it observes, or
# non-zero on a hang/crash/timeout. Never a silent pass.
hil-esp32s3-qemu:
	docker build -f test/hil/Dockerfile -t $(HIL_IMAGE) .
	docker run --rm $(HIL_IMAGE)

hil-esp32s3: hil-preflight hil-esp32s3-build | $(BUILD)
	@echo "hil-esp32s3: running $(HIL_MERGED_BIN) on board $(HIL_BOARD) via srig"
	srig run $(HIL_MERGED_BIN) --board $(HIL_BOARD) \
		--expect '##srig-exit:0##' \
		--timeout 120s --retries 3 --retry-delay 20s \
		--log $(HIL_LOG)
	@# --expect is NON-NEGOTIABLE: DO NOT remove it. srig-cli's
	@# runner/eval.go:64 Timeout() returns ExitCode: 0 when `expect` is
	@# nil, so without --expect, firmware that hangs or crashes before
	@# printing anything times out into a CLEAN PASS, not a failure. The
	@# sentinel line alone is not enough to protect against that - this
	@# flag is what makes a timeout actually mean "fail".

# Fails fast, before any hardware or network call, if this shell cannot
# actually run `srig run`: no API key, no srig binary, or an srig build
# too old to have the `run` subcommand (added after this project started
# depending on it - see test/hil/README.md).
hil-preflight:
	@if [ -z "$$SRIG_API_KEY" ]; then \
		echo "hil-preflight: SRIG_API_KEY is not set."; \
		echo "  Hardware runs need a SiliconRig API key. Export it in your"; \
		echo "  shell before running 'make hil-esp32s3' - never commit it:"; \
		echo "    export SRIG_API_KEY=<your key>"; \
		echo "  See test/hil/README.md for where to keep it."; \
		exit 1; \
	fi
	@if ! command -v srig >/dev/null 2>&1; then \
		echo "hil-preflight: the 'srig' CLI is not on PATH."; \
		echo "  Install it (e.g. to ~/.local/bin/srig) and make sure that"; \
		echo "  directory is on PATH. See test/hil/README.md."; \
		exit 1; \
	fi
	@if ! srig --help 2>&1 | grep -qE '^[[:space:]]*run[[:space:]]'; then \
		echo "hil-preflight: this srig build has no 'run' subcommand."; \
		echo "  hil-esp32s3 needs 'srig run ... --expect ...'. Upgrade srig"; \
		echo "  to a version that has it (built from rev dfd7c6d or later)."; \
		exit 1; \
	fi
	@echo "hil-preflight: OK (SRIG_API_KEY set, srig on PATH, 'run' subcommand present)"

# --- Cortex-M hardware-in-the-loop (HIL), QEMU only ---
# See test/hil/cortexm/README.md. No real board target yet - QEMU inside
# Docker is the whole harness (test/hil/cortexm/Dockerfile's CMD is the
# gate); CI runs this same target, emulated only.

HIL_CORTEXM_IMAGE := xlstm-hil-cortexm

hil-cortexm-build:
	docker build -f test/hil/cortexm/Dockerfile -t $(HIL_CORTEXM_IMAGE) .

hil-cortexm-qemu: hil-cortexm-build
	docker run --rm $(HIL_CORTEXM_IMAGE)

# Per-core convenience targets - same image and gate, restricted (via
# HIL_CORTEXM_CORES) to one core's ELF/QEMU machine.
hil-cortexm-m4: hil-cortexm-build
	docker run --rm -e HIL_CORTEXM_CORES=m4 $(HIL_CORTEXM_IMAGE)

hil-cortexm-m7: hil-cortexm-build
	docker run --rm -e HIL_CORTEXM_CORES=m7 $(HIL_CORTEXM_IMAGE)

hil-cortexm-m33: hil-cortexm-build
	docker run --rm -e HIL_CORTEXM_CORES=m33 $(HIL_CORTEXM_IMAGE)

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
