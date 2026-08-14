/* Linker script for the mps2-an505 (Cortex-M33) QEMU FPGA image.
 *
 * an505 uses a DIFFERENT memory map than an386/an500: code and vector
 * table at 0x10000000, RAM (data/bss/stack) at 0x28000000 - not the flat
 * 0x0/0x20000000 map linker/mps2_flat.ld uses for M4/M7. This is not a
 * cosmetic difference: linking this core with the flat script produces a
 * binary that boots into "qemu: fatal: Lockup: can't escalate 3 to
 * HardFault" (see .docs/plans/2026-08-14-cortexm-backend.md, trap 1).
 *
 * This firmware runs entirely in the M33's Secure state (no TrustZone
 * partitioning) - these addresses are an505's Secure aliases.
 */
MEMORY {
  CODE (rx)  : ORIGIN = 0x10000000, LENGTH = 2M
  DATA (rwx) : ORIGIN = 0x28000000, LENGTH = 2M
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
