/* Cortex-M board layer for the QEMU hardware-in-the-loop (HIL) test
 * firmware. Implements test/hil_platform.h (the board shim
 * test/hil_runner.cc talks to) and main() - the bare-metal equivalent of
 * test/hil_platform_host.c's main(), and structurally the same shape as
 * test/hil/main/hil_esp32s3.c's app_main() for the ESP32-S3 precedent.
 *
 * Output goes through plain stdio (fputs/fflush), exactly like
 * test/hil_platform_host.c. On real hardware that would need a UART
 * bring-up step the way hil_esp32s3.c has one; here it needs nothing
 * extra because this firmware links with -specs=rdimon.specs (ARM
 * semihosting newlib), which retargets stdio through the semihosting
 * console once startup.c's Reset_Handler calls
 * initialise_monitor_handles() - see startup.c. QEMU's mps2-* machines
 * implement that console directly, no serial bring-up needed.
 *
 * Trap 4 (.docs/plans/2026-08-14-cortexm-backend.md): the semihosting
 * console is not guaranteed to land on the same host stream every QEMU
 * build/version - qemu_gate.sh redirects both stdout and stderr into the
 * same log file it greps, so this is handled at the gate script level,
 * not here.
 *
 * Deliberately does NOT attempt cycle counting (DWT->CYCCNT reads zero
 * forever under QEMU - trap 5). Timing is a later work item, not this
 * one.
 */
#include "hil_platform.h"
#include "test_config.h"

#include <stdio.h>

#ifndef XLSTM_CORTEXM_CORE
#error "XLSTM_CORTEXM_CORE must be defined at build time (m4, m7, or m33) -" \
       " see test/hil/cortexm/Makefile."
#endif
#ifndef XLSTM_CORTEXM_FPU
#error "XLSTM_CORTEXM_FPU must be defined at build time (e.g. fpv4-sp-d16) -" \
       " see test/hil/cortexm/Makefile."
#endif

#define XLSTM_CM_STR_(x) #x
#define XLSTM_CM_STR(x) XLSTM_CM_STR_(x)

void hil_platform_println(const char *line) {
    fputs(line, stdout);
    fputc('\n', stdout);
    fflush(stdout);
}

const char *hil_platform_provenance_fields(void) {
    static char buf[192];
    /* "core" identifies which of the three emulated Cortex-M targets this
     * image is; "dsp" is __ARM_FEATURE_DSP, the compiler's own record of
     * whether the DSP extension was targeted for this build (Armv7E-M
     * always has it; Armv8-M Mainline has it unless built with a
     * "+nodsp" cpu variant, which this project never does); "fpu" is the
     * FPU variant this core was compiled for (see the Makefile's
     * cpuflags-* / fpuname-* per core - FPv4-SP-D16 on M4 vs FPv5-D16 on
     * M7 is exactly the newlib-in-hot-loops gap the plan's Decision A
     * targets). */
    snprintf(buf, sizeof(buf),
        "\"platform\":\"qemu-mps2\",\"core\":\"%s\",\"dsp\":%s,\"fpu\":\"%s\"",
        XLSTM_CM_STR(XLSTM_CORTEXM_CORE),
#if defined(__ARM_FEATURE_DSP)
        "true",
#else
        "false",
#endif
        XLSTM_CM_STR(XLSTM_CORTEXM_FPU));
    return buf;
}

/* Defined in test/hil_runner.cc as extern "C" - same cross-language call
 * test/hil_platform_host.c and test/hil/main/hil_esp32s3.c both make. */
extern int xlstm_hil_run(void);

int main(void) {
    return xlstm_hil_run();
}
