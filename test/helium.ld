/* Memory layout for the `helium` gate images: Cortex-M55 on an MPS3 AN547.
 *
 * The board's RAM regions, of which this uses two:
 *   0x00000000  512K  ITCM      - and the address the core fetches its
 *                                 vector table from at reset, which is why
 *                                 the table is pinned here and nothing else
 *                                 is.
 *   0x20000000  512K  DTCM
 *   0x21000000    4M  SRAM      - everything else. The golden vectors alone
 *                                 are around 170K per image, which is why
 *                                 the roomy region rather than either TCM.
 *
 * Load addresses equal run addresses throughout, so there is no .data copy
 * for the startup code to do: the emulator's ELF loader writes every section
 * straight into RAM.
 */

ENTRY(Reset_Handler)

MEMORY
{
  ITCM (rx)  : ORIGIN = 0x00000000, LENGTH = 512K
  SRAM (rwx) : ORIGIN = 0x21000000, LENGTH = 4M
}

SECTIONS
{
  .vectors : { KEEP(*(.isr_vector)) } > ITCM

  .text : {
    *(.text*)
    *(.rodata*)
    . = ALIGN(4);
    PROVIDE_HIDDEN(__preinit_array_start = .);
    KEEP(*(.preinit_array*))
    PROVIDE_HIDDEN(__preinit_array_end = .);
    PROVIDE_HIDDEN(__init_array_start = .);
    KEEP(*(SORT(.init_array.*)))
    KEEP(*(.init_array*))
    PROVIDE_HIDDEN(__init_array_end = .);
    . = ALIGN(8);
  } > SRAM

  /* C++ unwind tables. Not optional even for suites that raise nothing:
   * libgcc's unwinder references the two symbols unconditionally, so without
   * this section the link fails rather than the image growing. */
  .ARM.extab : { *(.ARM.extab* .gnu.linkonce.armextab.*) } > SRAM
  .ARM.exidx : {
    __exidx_start = .;
    *(.ARM.exidx* .gnu.linkonce.armexidx.*)
    __exidx_end = .;
    . = ALIGN(8);
  } > SRAM

  .data : { *(.data*) . = ALIGN(8); } > SRAM

  .bss : {
    __bss_start__ = .;
    *(.bss*)
    *(COMMON)
    . = ALIGN(8);
    __bss_end__ = .;
  } > SRAM

  /* newlib's _sbrk grows the heap from `end`. printf needs it for its stream
   * buffer, so an image with no heap prints nothing at all. */
  .heap (NOLOAD) : {
    . = ALIGN(8);
    __end__ = .;
    end = .;
    . = . + 1M;
    __HeapLimit = .;
  } > SRAM

  .stack (NOLOAD) : {
    . = ALIGN(8);
    . = . + 64K;
    __stack_top = .;
  } > SRAM

  /* Operand scratch positioned so that its LAST byte is the last byte of
   * mapped SRAM. An access one element past an operand placed here reaches
   * unmapped address space and faults, which is the only way a gate can see
   * an out-of-bounds READ whose lanes are then discarded - no value ever
   * changes, so no comparison can catch it. test/helium_gate.cc runs each
   * kernel with each of its operands in turn butted against this edge. */
  .oob_tail ORIGIN(SRAM) + LENGTH(SRAM) - 8K (NOLOAD) : {
    KEEP(*(.oob_tail))
  } > SRAM
}
