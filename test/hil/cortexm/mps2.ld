/* QEMU MPS2 memory map, shared by all three cores. an386/an500 (M4/M7)
 * use the flat map (code 0x0, RAM 0x20000000); an505 (M33) hard-faults
 * on that map and needs code 0x10000000, RAM 0x28000000 instead - pass
 * --defsym=AN505=1 (see the Makefile) to select it. */
CODE_ORIGIN = DEFINED(AN505) ? 0x10000000 : 0x00000000;
DATA_ORIGIN = DEFINED(AN505) ? 0x28000000 : 0x20000000;
CODE_LEN    = DEFINED(AN505) ? 2M : 4M;
DATA_LEN    = DEFINED(AN505) ? 2M : 4M;
MEMORY {
  CODE (rx)  : ORIGIN = CODE_ORIGIN, LENGTH = CODE_LEN
  DATA (rwx) : ORIGIN = DATA_ORIGIN, LENGTH = DATA_LEN
}
ENTRY(Reset_Handler)
SECTIONS {
  .text : {
    KEEP(*(.isr_vector))
    *(.text*) *(.rodata*) *(.glue_7*)
    . = ALIGN(4);
  } > CODE
  .ARM.exidx : {  /* libgcc's unwinder expects these to exist */
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
    *(.bss*) *(COMMON)
    . = ALIGN(8);
    _ebss = .;
  } > DATA
  . = ALIGN(8); end = .; PROVIDE(_end = .); /* newlib's _sbrk() wants this */
  _estack = ORIGIN(DATA) + LENGTH(DATA);
}
