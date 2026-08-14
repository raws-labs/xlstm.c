/* Linker script for QEMU's "flat" MPS2 memory map: code and vector table
 * at 0x00000000, RAM (data/bss/stack) at 0x20000000.
 *
 * Used for TWO of the three emulated cores this harness targets:
 *   - mps2-an386 (Cortex-M4)
 *   - mps2-an500 (Cortex-M7)
 *
 * Do NOT use this for mps2-an505 (Cortex-M33) - that FPGA image maps code
 * at 0x10000000 and RAM at 0x28000000 instead. Linking an505 flat at
 * 0x0/0x20000000 boots into "qemu: fatal: Lockup: can't escalate 3 to
 * HardFault" (see .docs/plans/2026-08-14-cortexm-backend.md, trap 1, and
 * linker/mps2_an505.ld for the an505-specific script).
 */
MEMORY {
  CODE (rx)  : ORIGIN = 0x00000000, LENGTH = 4M
  DATA (rwx) : ORIGIN = 0x20000000, LENGTH = 4M
}

ENTRY(Reset_Handler)

SECTIONS {
  .text : {
    KEEP(*(.isr_vector))
    *(.text*)
    *(.rodata*)
    *(.glue_7*)
    . = ALIGN(4);
  } > CODE

  /* libgcc's unwinder (__exidx_start/__exidx_end) expects these symbols
   * to exist even though this firmware never throws - trap 3. */
  .ARM.exidx : {
    __exidx_start = .;
    *(.ARM.exidx*)
    __exidx_end = .;
    . = ALIGN(4);
  } > CODE

  _etext = .;

  .data : AT(_etext) {
    _sdata = .;
    *(.data*)
    . = ALIGN(4);
    _edata = .;
  } > DATA

  .bss : {
    _sbss = .;
    *(.bss*)
    *(COMMON)
    . = ALIGN(8);
    _ebss = .;
  } > DATA

  . = ALIGN(8);
  end = .;
  PROVIDE(_end = .);      /* newlib's _sbrk() looks for this - trap 3. */

  _estack = ORIGIN(DATA) + LENGTH(DATA);
}
