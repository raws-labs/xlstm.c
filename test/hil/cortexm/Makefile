# Builds one ELF per emulated Cortex-M core (M4, M7, M33), normally run
# from inside test/hil/cortexm/Dockerfile.
#   make | make m4 (also m7, m33) | make size
CC   := arm-none-eabi-gcc
CXX  := arm-none-eabi-g++
SIZE := arm-none-eabi-size
ROOT    := ../../..
INCDIRS := -I$(ROOT)/include -I$(ROOT)/test
BUILD   := build
# 64 covers every case in test/reference_data.h; shrinks static buffers
# from the 256 default to fit these MCU-class targets.
TEST_MAX_H := 64
WARN_CFLAGS := -Wall -Wextra
COMMON_CFLAGS   := -O2 $(WARN_CFLAGS) -std=c99 \
                    -DXLSTM_TEST_MAX_H=$(TEST_MAX_H) $(INCDIRS)
COMMON_CXXFLAGS := -O2 $(WARN_CFLAGS) -std=c++17 -fno-exceptions -fno-rtti \
                    -DXLSTM_TEST_MAX_H=$(TEST_MAX_H) -DXLSTM_HIL_EXPECT_BACKEND=ref \
                    $(INCDIRS)
# -nostartfiles: this project supplies its own Reset_Handler/vector table.
LDFLAGS_COMMON := -nostartfiles -specs=rdimon.specs -Wl,--gc-sections -lm
CORES := m4 m7 m33
# XLSTM_FPU_HAS_MINMAX_ROUND (include/xlstm_util.h) MUST be defined
# before cpuflags-* below, which use it via `:=` (expanded immediately) -
# defined after, it silently never reaches the compiler. m4 undeclared
# on purpose: default (0) is the portable path.
capflags-m7  := -DXLSTM_FPU_HAS_MINMAX_ROUND=1
capflags-m33 := -DXLSTM_FPU_HAS_MINMAX_ROUND=1
# an386 (M4) and an500 (M7) are Armv7E-M; an505 (M33) is Armv8-M Mainline.
cpuflags-m4  := -mcpu=cortex-m4  -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 $(capflags-m4)
cpuflags-m7  := -mcpu=cortex-m7  -mthumb -mfloat-abi=hard -mfpu=fpv5-d16 $(capflags-m7)
cpuflags-m33 := -mcpu=cortex-m33 -mthumb -mfloat-abi=hard -mfpu=fpv5-sp-d16 $(capflags-m33)
fpuname-m4  := fpv4-sp-d16
fpuname-m7  := fpv5-d16
fpuname-m33 := fpv5-sp-d16
# an505 needs a different memory map than an386/an500 - mps2.ld picks it
# via --defsym=AN505=1, passed only for m33.
LDSCRIPT    := mps2.ld
ldflags-m33 := -Xlinker --defsym=AN505=1
# test/hil_runner.cc calls each suite's renamed entry point by fixed name.
TESTMAIN_slstm_test    := slstm_test_main
TESTMAIN_mlstm_test    := mlstm_test_main
TESTMAIN_slstm_s8_test := slstm_s8_test_main
TESTMAIN_mlstm_s8_test := mlstm_s8_test_main
.PHONY: all clean size $(CORES)
all: $(foreach c,$(CORES),$(BUILD)/hil_$(c).elf)
size: all
	@$(SIZE) $(foreach c,$(CORES),$(BUILD)/hil_$(c).elf)
# One template, instantiated once per core, so the three cannot drift
# out of sync by editing one and forgetting the others.
define CORE_TEMPLATE
$(BUILD)/$(1):
	@mkdir -p $(BUILD)/$(1)
$(BUILD)/$(1)/hil_cortexm.o: hil_cortexm.c | $(BUILD)/$(1)
	$(CC) $(cpuflags-$(1)) $(COMMON_CFLAGS) \
	  -DXLSTM_CORTEXM_CORE=$(1) -DXLSTM_CORTEXM_FPU=$(fpuname-$(1)) -c $$< -o $$@
$(BUILD)/$(1)/%.o: $(ROOT)/src/%.c | $(BUILD)/$(1)
	$(CC) $(cpuflags-$(1)) $(COMMON_CFLAGS) -c $$< -o $$@
$(BUILD)/$(1)/%.o: $(ROOT)/test/%.cc | $(BUILD)/$(1)
	$(CXX) $(cpuflags-$(1)) $(COMMON_CXXFLAGS) -DXLSTM_TEST_MAIN=$$(TESTMAIN_$$*) -c $$< -o $$@
CORE_OBJS_$(1) := $$(addprefix $(BUILD)/$(1)/,\
  hil_cortexm.o hil_runner.o \
  slstm.o mlstm.o xlstm_quant.o slstm_s8.o mlstm_s8.o xlstm_simd_ref.o \
  slstm_test.o mlstm_test.o slstm_s8_test.o mlstm_s8_test.o)
$(BUILD)/hil_$(1).elf: $$(CORE_OBJS_$(1)) $(LDSCRIPT)
	$(CXX) $(cpuflags-$(1)) $(ldflags-$(1)) -T $(LDSCRIPT) $(LDFLAGS_COMMON) -o $$@ $$(CORE_OBJS_$(1))
$(1): $(BUILD)/hil_$(1).elf
endef
$(foreach c,$(CORES),$(eval $(call CORE_TEMPLATE,$(c))))
clean:
	@rm -rf $(BUILD)
