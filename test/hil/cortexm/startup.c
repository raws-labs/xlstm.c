/* Minimal Cortex-M startup: vector table, reset handler, FPU enable.
 *
 * Shared byte-for-byte across all three emulated cores (M4/M7/M33) - the
 * System Control Space register this touches (CPACR at 0xE000ED88) sits
 * at the same address on ARMv7E-M and ARMv8-M Mainline, so nothing here
 * needs to be core-specific. Per-core differences (memory map, CPU/FPU
 * variant) live in the linker scripts under linker/ and the Makefile's
 * per-core flags instead.
 *
 * FPU enable (trap 2, .docs/plans/2026-08-14-cortexm-backend.md): CPACR
 * CP10/CP11 must be set to full access BEFORE any VFP instruction
 * executes, followed by dsb+isb so the write is visible before anything
 * that depends on it runs. Without this, the first float touched by
 * libc's semihosting-retargeted printf (or by the f32 suites themselves)
 * hard-faults. This was not predicted by inspection - it was found by
 * running and reading the fault. Do not remove it or reorder it after
 * __libc_init_array()/main().
 *
 * initialise_monitor_handles() is librdimon's (semihosting newlib) call to
 * wire stdio through the semihosting console before any stdio call is
 * made - this project links with -specs=rdimon.specs precisely so plain
 * fputs/printf (as test/hil_platform_host.c and hil_cortexm.c both use)
 * work unmodified on target, matching the host build.
 */
#include <stdint.h>

extern unsigned _etext, _sdata, _edata, _sbss, _ebss, _estack;
extern int main(void);
extern void __libc_init_array(void);
extern void initialise_monitor_handles(void);

void Reset_Handler(void) {
    /* CPACR |= (0xF << 20): full access to CP10 (single-precision FPU)
     * and CP11 (the coprocessor pair VFP instructions decode through). */
    *(volatile uint32_t *)0xE000ED88 |= (0xFu << 20);
    __asm__ volatile("dsb" ::: "memory");
    __asm__ volatile("isb" ::: "memory");

    /* Copy .data from flash (LMA, right after .text) to its RAM VMA, then
     * zero .bss - the usual C runtime bring-up a vendor's crt0 normally
     * does for you. */
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

    /* main() (via test/hil_platform_host.c's pattern, mirrored here by
     * hil_cortexm.c) already returns after printing the sentinel. There is
     * nothing further to run and nowhere to return to - spin. A QEMU gate
     * with a timeout treats this identically to any other hang if the
     * sentinel was never seen; if it WAS seen, the gate has already made
     * its decision and just needs the process killed. */
    for (;;) {}
}

/* Any exception this vector table doesn't give a real handler to (bus
 * fault, usage fault, hard fault, ...) lands here and spins forever. A
 * spin is indistinguishable from a hang to the QEMU gate script, which is
 * exactly the point (see .docs/plans/2026-08-14-cortexm-backend.md,
 * trap 6, and the CLAUDE.md requirement that a hang must fail, never pass:
 * a silently-eaten fault must not look like a clean run). */
void Default_Handler(void) {
    for (;;) {}
}

/* trap 3: the linker (via -specs=rdimon.specs' use of the standard
 * crti/crtn convention) expects these two to exist even though this
 * firmware registers no static C++ constructors/destructors of its own
 * beyond what __libc_init_array() already walks. */
void _init(void) {}
void _fini(void) {}

/* Vector table: SP_main, Reset, then just enough real vs. default fault
 * handlers to keep every faulting path landing in Default_Handler rather
 * than falling off the end of an undersized table. Index layout matches
 * the ARMv7-M/ARMv8-M exception table (NMI, HardFault, MemManage,
 * BusFault, UsageFault, four reserved slots, SVCall, two reserved,
 * PendSV, SysTick) - entries this firmware never triggers (SVCall,
 * PendSV, SysTick) still point at Default_Handler rather than being left
 * at 0, since a 0 entry taken as a branch target is its own fault. */
__attribute__((section(".isr_vector"), used))
void (* const g_vectors[16])(void) = {
    (void (*)(void))&_estack,  /* initial SP */
    Reset_Handler,              /* Reset */
    Default_Handler,            /* NMI */
    Default_Handler,            /* HardFault */
    Default_Handler,            /* MemManage */
    Default_Handler,            /* BusFault */
    Default_Handler,            /* UsageFault */
    Default_Handler,            /* reserved (was 0) */
    Default_Handler,            /* reserved (was 0) */
    Default_Handler,            /* reserved (was 0) */
    Default_Handler,            /* reserved (was 0) */
    Default_Handler,            /* SVCall */
    Default_Handler,            /* reserved (debug monitor) */
    Default_Handler,            /* reserved (was 0) */
    Default_Handler,            /* PendSV */
    Default_Handler,            /* SysTick */
};
