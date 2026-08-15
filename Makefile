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
