/* Board platform shim for the hardware-in-the-loop (HIL) test runner
 * (test/hil_runner.cc).
 *
 * test/hil_runner.cc is board-independent: it talks to the outside world
 * only through the two functions declared here. Each board under a
 * (future) boards/<board>/ directory implements them; the host acceptance
 * build implements them too (stdout-backed), proving the runner works
 * before any hardware is involved.
 *
 * Deliberately narrow: this is not a general platform/BSP abstraction (no
 * clock setup, no timing, no halt). The suites under test print their own
 * PASS/FAIL detail via the C library's normal stdio (already true before
 * this file existed), which on a real board is retargeted to a UART by
 * that board's own bring-up code - a board-specific concern that belongs
 * in the board layer's main(), not here. What actually needs a board-
 * independent hook is narrower: one reliable way to emit a line the rig
 * can capture even if stdio retargeting is broken or absent, and one way
 * for the board to say what it is. Add a function here only when a caller
 * in test/hil_runner.cc actually needs it - do not grow this file
 * speculatively for future boards.
 * =========================================================================*/

#ifndef XLSTM_HIL_PLATFORM_H_
#define XLSTM_HIL_PLATFORM_H_

#ifdef __cplusplus
extern "C" {
#endif

/* Emit one line of text over the rig's serial transport (or stdout, for the
 * host build). `line` is a NUL-terminated string with no trailing newline;
 * the implementation appends whatever line ending its transport needs.
 *
 * This is the ONLY output path test/hil_runner.cc relies on for anything
 * the rig must be able to see: the provenance banner, the backend-guard
 * failure message, the per-suite progress lines, and the final
 * "##srig-exit:N##" sentinel all go through this one function. The four
 * suites under test still print their own pass/fail detail straight to
 * stdio, independently of this shim, exactly as they did before this file
 * existed.
 *
 * Must block until the line is actually on the wire (or hand off to a
 * buffer the transport itself drains synchronously) - hil_runner.cc calls
 * this in strict sequence and relies on lines not being dropped or
 * reordered, in particular right before a fatal early return. */
void hil_platform_println(const char *line);

/* Zero or more comma-separated JSON object members - `"key":value` pairs,
 * no surrounding braces, no leading or trailing comma, e.g.
 * `"platform":"nucleo_h753zi","clock_hz":480000000` - that the board layer
 * contributes to the provenance banner. At minimum this should identify
 * the platform (e.g. `"platform":"host-linux-x86_64"`); a board may add
 * whatever else is worth recording for provenance (chip revision, clock
 * speed, firmware build id, ...). test/hil_runner.cc splices this fragment
 * in after its own fixed fields (active SIMD backend, expected backend,
 * XLSTM_TEST_MAX_H), so it composes regardless of how many members the
 * board contributes.
 *
 * Must return a non-NULL, NUL-terminated string. A board layer with
 * nothing to add returns "" (the empty string) - never NULL, and never a
 * fragment starting or ending with a comma. */
const char *hil_platform_provenance_fields(void);

#ifdef __cplusplus
}
#endif

#endif /* XLSTM_HIL_PLATFORM_H_ */
