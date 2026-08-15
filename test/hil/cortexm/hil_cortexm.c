/* Cortex-M board layer: startup (vector table, reset handler, FPU
 * enable, shared byte-for-byte across M4/M7/M33) plus the board shim
 * implementing test/hil_platform.h and main(). Plain stdio, retargeted
 * through semihosting once Reset_Handler calls initialise_monitor_handles(). */
#include "hil_platform.h"
#include "test_config.h"
#include "xlstm_util.h" /* XLSTM_FPU_HAS_MINMAX_ROUND, reported below */
#include <stdint.h>
#include <stdio.h>
#if !defined(XLSTM_CORTEXM_CORE) || !defined(XLSTM_CORTEXM_FPU)
#error "XLSTM_CORTEXM_CORE (m4/m7/m33) and XLSTM_CORTEXM_FPU (e.g. fpv4-sp-d16) must be defined at build time"
#endif
extern unsigned _etext, _sdata, _edata, _sbss, _ebss, _estack;
extern int main(void);
extern void __libc_init_array(void);
extern void initialise_monitor_handles(void);
void Reset_Handler(void) {
    /* CPACR CP10/CP11 full access + dsb/isb before any VFP instruction,
     * or the first float hard-faults (found by running, not by inspection). */
    *(volatile uint32_t *)0xE000ED88 |= (0xFu << 20);
    __asm__ volatile("dsb" ::: "memory");
    __asm__ volatile("isb" ::: "memory");
    /* Copy .data from flash to its RAM VMA, then zero .bss. */
    unsigned *src = &_etext, *dst = &_sdata;
    while (dst < &_edata) {
        *dst++ = *src++;
    }
    for (dst = &_sbss; dst < &_ebss; ) {
        *dst++ = 0;
    }
    initialise_monitor_handles();
    __libc_init_array();
    main();
    for (;;) {} /* nothing to return to; a hung gate treats this as a fault */
}
/* An untriggered fault lands here and spins - indistinguishable from a
 * hang, never a silent pass. */
void Default_Handler(void) { for (;;) {} }
/* -specs=rdimon.specs expects these even though nothing here throws. */
void _init(void) {} void _fini(void) {}
/* Vector table (ARMv7-M/ARMv8-M layout): SP_main, Reset, then
 * Default_Handler for the 14 exceptions this firmware never triggers. */
__attribute__((section(".isr_vector"), used))
void (* const g_vectors[16])(void) = {
    (void (*)(void))&_estack, Reset_Handler, Default_Handler, Default_Handler,
    Default_Handler, Default_Handler, Default_Handler, Default_Handler,
    Default_Handler, Default_Handler, Default_Handler, Default_Handler,
    Default_Handler, Default_Handler, Default_Handler, Default_Handler,
};
#define XLSTM_CM_STR_(x) #x
#define XLSTM_CM_STR(x) XLSTM_CM_STR_(x)
void hil_platform_println(const char *line) {
    fputs(line, stdout);
    fputc('\n', stdout);
    fflush(stdout);
}
const char *hil_platform_provenance_fields(void) {
    static char buf[256];
    /* fpu_minmax_round: what the preprocessor resolved, not what the
     * Makefile meant - a Makefile ordering slip once dropped the -D flag. */
    snprintf(buf, sizeof(buf),
        "\"platform\":\"qemu-mps2\",\"core\":\"%s\",\"dsp\":%s,\"fpu\":\"%s\","
        "\"fpu_minmax_round\":%s",
        XLSTM_CM_STR(XLSTM_CORTEXM_CORE),
#if defined(__ARM_FEATURE_DSP)
        "true",
#else
        "false",
#endif
        XLSTM_CM_STR(XLSTM_CORTEXM_FPU),
        XLSTM_FPU_HAS_MINMAX_ROUND ? "true" : "false");
    return buf;
}
extern int xlstm_hil_run(void); /* test/hil_runner.cc, extern "C" */
int main(void) {
    return xlstm_hil_run();
}
