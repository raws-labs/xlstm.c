/* Bare-metal bring-up for the `helium` gate: Cortex-M55 on an MPS3 AN547.
 *
 * Linked into all five binaries of `make test-helium`. Everything else those
 * images need - printf, malloc, exit - comes from the toolchain's rdimon
 * semihosting library, so what is left here is the four things a library
 * cannot know: where the stack starts, that the vector unit has to be
 * switched on, that .bss must be zero, and how a return value from main
 * becomes the emulator's exit status.
 *
 * Built with -nostartfiles, so this replaces crt0 rather than sitting beside
 * it. crt0 would work, but on M-profile it takes its stack from a semihosting
 * SYS_HEAPINFO call, and the vector table below has to exist either way - the
 * core loads the initial stack pointer and the reset address from it before
 * it executes anything. Doing both here is one file instead of two mechanisms.
 * =========================================================================*/

#include <stdint.h>

extern unsigned __bss_start__, __bss_end__, __stack_top;
extern void (*__init_array_start[])(void);
extern void (*__init_array_end[])(void);
extern void initialise_monitor_handles(void); /* rdimon: opens stdout */
extern int main(void);

/* Semihosting SYS_EXIT_EXTENDED. The plain SYS_EXIT that newlib's exit()
 * reaches passes only a reason code, so a non-zero verdict from main would
 * arrive at the host as a clean exit; the extended call takes the status as a
 * second word and QEMU exits with it. That is what makes `make test-helium`
 * able to fail: the firmware's return value IS the target's exit status. */
static void semi_exit(int code)
{
    volatile uint32_t block[2];

    block[0] = 0x20026u; /* ADP_Stopped_ApplicationExit */
    block[1] = (uint32_t)code;
    __asm__ volatile("mov r0, %0\n\tmov r1, %1\n\tbkpt 0xAB\n"
                     :
                     : "r"(0x20u), "r"(block)
                     : "r0", "r1", "memory");
    for (;;) {
    }
}

void Reset_Handler(void)
{
    unsigned* p;
    void (**f)(void);

    /* CPACR CP10/CP11, full access. MVE and the FPU share these bits, and at
     * reset they deny both: without this the first vector instruction in the
     * kernels raises a UsageFault (NOCP) instead of executing. It is also the
     * negative control that proves an emulator is really running MVE rather
     * than ignoring it - deny the coprocessors and a green run turns into a
     * fault. */
    *(volatile uint32_t*)0xE000ED88u |= (0xFu << 20);
    __asm__ volatile("dsb\n\tisb");

    for (p = &__bss_start__; p < &__bss_end__; ++p) *p = 0u;
    initialise_monitor_handles();
    for (f = __init_array_start; f != __init_array_end; ++f) (*f)();
    semi_exit(main());
}

/* Semihosting SYS_WRITE0: one instruction and a pointer to a NUL-terminated
 * string, touching no library state at all. */
static void semi_write0(const char* s)
{
    __asm__ volatile("mov r0, %0\n\tmov r1, %1\n\tbkpt 0xAB\n"
                     :
                     : "r"(0x04u), "r"(s)
                     : "r0", "r1", "memory");
}

static void Fault_Handler(void)
{
    /* Still no printf: a fault may have arrived through a wrecked C
     * environment, and printf from here can hang instead of reporting. But an
     * exit status alone cannot say WHICH check died, and the out-of-bounds
     * check is meant to fail by faulting - so the one line below goes out
     * through semihosting directly, which is as safe as the exit call under
     * it. Any non-zero status still fails the gate. */
    semi_write0("FAIL fault: the image faulted. The check it was running is "
                "the last [ RUN ] line above.\n");
    semi_exit(70);
}

/* The core fetches the initial stack pointer and the reset address from this
 * table at the boot vector address, which on this part is 0 - the linker
 * script puts the table there. Only the entries that can fire before main
 * are listed; anything past them is unreachable in a bare compute image with
 * no interrupt sources enabled. */
__attribute__((section(".isr_vector"), used))
void (* const g_vectors[])(void) = {
    (void (*)(void)) & __stack_top,
    Reset_Handler,
    Fault_Handler, /* NMI */
    Fault_Handler, /* HardFault */
    Fault_Handler, /* MemManage */
    Fault_Handler, /* BusFault */
    Fault_Handler, /* UsageFault - what a disabled vector unit raises */
    Fault_Handler, /* SecureFault */
};
