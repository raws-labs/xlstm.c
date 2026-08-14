/* Host-side implementation of test/hil_platform.h.
 *
 * Two jobs:
 *   1. Proves test/hil_runner.cc actually works before any hardware is
 *      involved - see the W3 report for the acceptance transcripts this
 *      produced.
 *   2. A permanent, zero-cost regression check that hil_platform.h stays
 *      genuinely implementable from plain C99 (deliberate: it is guarded
 *      with `extern "C"` precisely so a board's C BSP can implement it,
 *      and this is what actually exercises that end to end, including the
 *      C -> C++ call into xlstm_hil_run() across the language boundary).
 *
 * Stdout-backed: not a board, just what "print a line" means on a host.
 *
 * Not wired into the Makefile yet - the make target that fuses this with
 * test/hil_runner.cc and the four suites is a later task's job. Until
 * then, build it by hand (see the W3 report for the exact commands) or
 * treat it as the reference a board layer's hil_platform.c should follow.
 */
#include "hil_platform.h"

#include <stdio.h>

void hil_platform_println(const char *line) {
    /* fputs + explicit '\n', not puts(), so behavior doesn't depend on
     * puts()'s own newline-appending convention. fflush keeps the sentinel
     * line from being lost if this ever runs somewhere with non-line-
     * buffered stdout (e.g. redirected to a file). */
    fputs(line, stdout);
    fputc('\n', stdout);
    fflush(stdout);
}

const char *hil_platform_provenance_fields(void) {
    return "\"platform\":\"host-acceptance-build\"";
}

/* Defined in test/hil_runner.cc as extern "C", so this plain C declaration
 * links against it directly. */
extern int xlstm_hil_run(void);

int main(void) {
    return xlstm_hil_run();
}
