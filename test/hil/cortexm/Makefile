# Builds the Cortex-M QEMU hardware-in-the-loop (HIL) firmware: the
# board-independent HIL runner (../hil_runner.cc, ../hil_platform.h), this
# directory's board layer (hil_cortexm.c, startup.c), and the four
# existing suite entry points - fused into one ELF per emulated core.
#
# Scope (see CLAUDE.md work items 1-2): harness only. Every core links
# src/xlstm_simd_ref.c and is built with -DXLSTM_HIL_EXPECT_BACKEND=ref -
# there is no src/xlstm_simd_cortexm.c yet and this Makefile must not grow
# one. XLSTM_SIMD=ref is the scalar kernel already verified at 39
# assertions across three host backends (see test-ref/test-sse2/test-neon
# in the repository root Makefile); if a core fails here, the harness is
# at fault, not the kernel.
#
# Invoked directly (arm-none-eabi-gcc/g++, qemu-system-arm must be on
# PATH) or, more commonly, from inside test/hil/cortexm/Dockerfile, which
# is what the repository root Makefile's hil-cortexm-* targets and CI
# actually run.
#
#   make            - build build/hil_m4.elf, build/hil_m7.elf,
#                      build/hil_m33.elf
#   make size       - print flash/SRAM footprint for all three
#   make m4         - build only build/hil_m4.elf (also: make m7, make m33)
#   make clean

CC   := arm-none-eabi-gcc
CXX  := arm-none-eabi-g++
SIZE := arm-none-eabi-size

ROOT    := ../../..
INCDIRS := -I$(ROOT)/include -I$(ROOT)/test

# Matches test/hil/main/CMakeLists.txt's XLSTM_TEST_MAX_H=64 for the
# ESP32-S3 image: shrinks the four suites' static state/scratch/output
# buffers from the 256 default (test/test_config.h) to fit these MCU-class
# targets. 64 comfortably covers every case in test/reference_data.h
# today (largest: H=64, I=64 - SweepS64/SweepM64).
TEST_MAX_H := 64

BUILD := build

WARN_CFLAGS := -Wall -Wextra
COMMON_CFLAGS   := -O2 $(WARN_CFLAGS) -std=c99 \
                    -DXLSTM_TEST_MAX_H=$(TEST_MAX_H) $(INCDIRS)
# -fno-exceptions -fno-rtti: this firmware never throws and links no
# unwind runtime beyond the .ARM.exidx table the linker script already
# provides symbols for (trap 3) - keeps flash footprint down and keeps
# the C++ ABI surface this build depends on small and auditable.
COMMON_CXXFLAGS := -O2 $(WARN_CFLAGS) -std=c++17 -fno-exceptions -fno-rtti \
                    -DXLSTM_TEST_MAX_H=$(TEST_MAX_H) -DXLSTM_HIL_EXPECT_BACKEND=ref \
                    $(INCDIRS)

# -specs=rdimon.specs: ARM semihosting newlib - retargets stdio through
# the semihosting console (startup.c calls initialise_monitor_handles()).
# -nostartfiles: this project supplies its own Reset_Handler/vector table
# (startup.c), not newlib's/CMSIS's.
LDFLAGS_COMMON := -nostartfiles -specs=rdimon.specs -Wl,--gc-sections -lm

CORES := m4 m7 m33

# Per-core CPU/FPU target. an386 (M4) and an500 (M7) are both Armv7E-M
# with the DSP extension; an505 (M33) is Armv8-M Mainline, also built with
# the DSP extension (the GCC cortex-m33 default - see hil_cortexm.c's
# provenance report of __ARM_FEATURE_DSP, which records what actually got
# compiled rather than asserting it here as a comment that could drift).
cpuflags-m4  := -mcpu=cortex-m4  -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16
cpuflags-m7  := -mcpu=cortex-m7  -mthumb -mfloat-abi=hard -mfpu=fpv5-d16
cpuflags-m33 := -mcpu=cortex-m33 -mthumb -mfloat-abi=hard -mfpu=fpv5-sp-d16

# See linker/mps2_flat.ld and linker/mps2_an505.ld - trap 1, an505 needs a
# different memory map than an386/an500.
ldscript-m4  := linker/mps2_flat.ld
ldscript-m7  := linker/mps2_flat.ld
ldscript-m33 := linker/mps2_an505.ld

qemumach-m4  := mps2-an386
qemumach-m7  := mps2-an500
qemumach-m33 := mps2-an505

fpuname-m4  := fpv4-sp-d16
fpuname-m7  := fpv5-d16
fpuname-m33 := fpv5-sp-d16

# Renamed entry point per suite translation unit - test/hil_runner.cc
# calls all four by fixed name (see that file's header comment).
TESTMAIN_slstm_test    := slstm_test_main
TESTMAIN_mlstm_test    := mlstm_test_main
TESTMAIN_slstm_s8_test := slstm_s8_test_main
TESTMAIN_mlstm_s8_test := mlstm_s8_test_main

.PHONY: all clean size $(CORES)

all: $(foreach c,$(CORES),$(BUILD)/hil_$(c).elf)

size: all
	@echo "=== flash (text+data) / SRAM (data+bss) per core ==="
	@$(SIZE) $(foreach c,$(CORES),$(BUILD)/hil_$(c).elf)

# One template, instantiated once per core via $(eval $(call ...)) below -
# not three hand-copied blocks, so the three cores cannot drift out of
# sync with each other by editing one and forgetting the others.
define CORE_TEMPLATE

$(BUILD)/$(1):
	@mkdir -p $(BUILD)/$(1)

$(BUILD)/$(1)/startup.o: startup.c | $(BUILD)/$(1)
	$(CC) $(cpuflags-$(1)) $(COMMON_CFLAGS) -c $$< -o $$@

$(BUILD)/$(1)/hil_cortexm.o: hil_cortexm.c | $(BUILD)/$(1)
	$(CC) $(cpuflags-$(1)) $(COMMON_CFLAGS) \
	  -DXLSTM_CORTEXM_CORE=$(1) -DXLSTM_CORTEXM_FPU=$(fpuname-$(1)) \
	  -c $$< -o $$@

$(BUILD)/$(1)/%.o: $(ROOT)/src/%.c | $(BUILD)/$(1)
	$(CC) $(cpuflags-$(1)) $(COMMON_CFLAGS) -c $$< -o $$@

$(BUILD)/$(1)/%.o: $(ROOT)/test/%.cc | $(BUILD)/$(1)
	$(CXX) $(cpuflags-$(1)) $(COMMON_CXXFLAGS) -DXLSTM_TEST_MAIN=$$(TESTMAIN_$$*) -c $$< -o $$@

CORE_OBJS_$(1) := $$(addprefix $(BUILD)/$(1)/,\
  startup.o hil_cortexm.o hil_runner.o \
  slstm.o mlstm.o xlstm_quant.o slstm_s8.o mlstm_s8.o xlstm_simd_ref.o \
  slstm_test.o mlstm_test.o slstm_s8_test.o mlstm_s8_test.o)

$(BUILD)/hil_$(1).elf: $$(CORE_OBJS_$(1)) $$(ldscript-$(1))
	$(CXX) $(cpuflags-$(1)) -T $$(ldscript-$(1)) $(LDFLAGS_COMMON) \
	  -o $$@ $$(CORE_OBJS_$(1))

$(1): $(BUILD)/hil_$(1).elf

endef

$(foreach c,$(CORES),$(eval $(call CORE_TEMPLATE,$(c))))

clean:
	@rm -rf $(BUILD)
