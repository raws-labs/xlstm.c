/* Board-independent hardware-in-the-loop (HIL) test runner.
 *
 * Fuses the four existing suite entry points (test/slstm_test.cc,
 * test/mlstm_test.cc, test/slstm_s8_test.cc, test/mlstm_s8_test.cc) into
 * one pass/fail run and reports it the way the SiliconRig rig expects: a
 * machine-parseable provenance banner, then a sentinel line matching
 * `##srig-exit:N##` that the rig maps directly to the run's exit code.
 *
 * Nothing here is board-specific. All device I/O goes through
 * test/hil_platform.h, which a board layer (or, for the host acceptance
 * build, a small stdout-backed shim) implements.
 *
 * The four suites are linked into the same binary as separate translation
 * units, each of the four test runners below compiled with its entry point
 * renamed via `-DXLSTM_TEST_MAIN=<name>` (test/test_config.h documents the
 * knob these runners share; the rename hook itself is defined in each one):
 *
 *   test/slstm_test.cc     -DXLSTM_TEST_MAIN=slstm_test_main
 *   test/mlstm_test.cc     -DXLSTM_TEST_MAIN=mlstm_test_main
 *   test/slstm_s8_test.cc  -DXLSTM_TEST_MAIN=slstm_s8_test_main
 *   test/mlstm_s8_test.cc  -DXLSTM_TEST_MAIN=mlstm_s8_test_main
 *
 * This file and those four are all C++17, so the plain (non-`extern "C"`)
 * declarations below link against those renamed definitions directly - no
 * separate header needed. The actual build wiring (make target / firmware
 * project) that passes those defines is a later task's job; this file only
 * assumes the names above exist by the time it is linked.
 *
 * Backend guard: this build must be told which SIMD backend it expects to
 * have linked, via -DXLSTM_HIL_EXPECT_BACKEND=<name> (e.g. `esp`, `sse2`,
 * `ref`). If the backend actually linked (xlstm_simd_backend()) disagrees,
 * that is not a warning - it means this run cannot prove what it claims to
 * (e.g. a silent fallback to the ref backend would make every suite pass
 * without ever exercising the accelerated backend under test), so the
 * runner fails loudly before running anything.
 * =========================================================================*/

#include "hil_platform.h"
#include "test_config.h"
#include "xlstm_simd.h"

#include <cstdio>
#include <cstring>

#ifndef XLSTM_HIL_EXPECT_BACKEND
#error "XLSTM_HIL_EXPECT_BACKEND must be defined at build time to the SIMD" \
       " backend this build is expected to link, e.g." \
       " -DXLSTM_HIL_EXPECT_BACKEND=sse2 - see xlstm_simd_backend() in" \
       " include/xlstm_simd.h for the valid names (ref, sse2, neon, esp)."
#endif

#define XLSTM_HIL_STRINGIFY_(x) #x
#define XLSTM_HIL_STRINGIFY(x) XLSTM_HIL_STRINGIFY_(x)

/* The four suite entry points, renamed per translation unit as described
 * above. Each returns 0 on full pass, non-zero if any case in that suite
 * failed - the same convention plain `main()` used before W2 renamed them. */
extern int slstm_test_main(void);
extern int mlstm_test_main(void);
extern int slstm_s8_test_main(void);
extern int mlstm_s8_test_main(void);

namespace {

struct HilSuite {
    const char *name;
    int (*run)(void);
};

const HilSuite kSuites[] = {
    {"slstm_f32", slstm_test_main},
    {"mlstm_f32", mlstm_test_main},
    {"slstm_s8",  slstm_s8_test_main},
    {"mlstm_s8",  mlstm_s8_test_main},
};
const int kSuiteCount = sizeof(kSuites) / sizeof(kSuites[0]);

/* Generous but fixed - no dynamic allocation, matching the rest of the
 * test harness. The board-contributed provenance fragment is the only
 * unbounded-ish input; snprintf truncates safely if a board ever
 * contributes something implausibly long instead of overflowing. */
const int kLineBufSize = 512;

} /* namespace */

/* extern "C" so a plain C board main() (typical for firmware entry points,
 * e.g. an ESP-IDF app_main.c or a Cortex-M startup file) can call this with
 * an ordinary `extern int xlstm_hil_run(void);` declaration, with no C++
 * name mangling to match. */
extern "C" int xlstm_hil_run(void) {
    char line[kLineBufSize];

    const char *backend = xlstm_simd_backend();
    const char *expected = XLSTM_HIL_STRINGIFY(XLSTM_HIL_EXPECT_BACKEND);
    const char *extra = hil_platform_provenance_fields();
    if (extra == NULL) {
        extra = ""; /* be forgiving of a misbehaving board layer */
    }

    /* Provenance banner FIRST, before the backend guard below - so even a
     * build that is about to fail the guard still reports what it actually
     * linked, and a truncated log (crash/hang before the guard even
     * evaluates) still shows this line. */
    std::snprintf(line, sizeof(line),
        "XLSTM_PROVENANCE:{\"backend\":\"%s\",\"expected_backend\":\"%s\","
        "\"test_max_h\":%d%s%s}",
        backend, expected, XLSTM_TEST_MAX_H,
        (extra[0] != '\0') ? "," : "", extra);
    hil_platform_println(line);

    /* Hazard 1 (see task brief): must fail loudly and BEFORE running
     * anything if the linked backend is not the one this run claims to
     * verify. Not a warning - a silent fallback to `ref` here would let
     * every suite pass while proving nothing about the intended backend. */
    if (std::strcmp(backend, expected) != 0) {
        std::snprintf(line, sizeof(line),
            "HIL_FATAL: linked SIMD backend \"%s\" does not match the "
            "backend this build expects (\"%s\", from "
            "XLSTM_HIL_EXPECT_BACKEND) - refusing to run any suite. This "
            "would otherwise look like a passing hardware run of the "
            "wrong backend.", backend, expected);
        hil_platform_println(line);
        hil_platform_println("##srig-exit:1##");
        return 1;
    }

    int failed = 0;
    for (int i = 0; i < kSuiteCount; ++i) {
        std::snprintf(line, sizeof(line), "HIL_SUITE_BEGIN:%s",
                     kSuites[i].name);
        hil_platform_println(line);

        int rc = kSuites[i].run();
        failed |= rc;

        std::snprintf(line, sizeof(line), "HIL_SUITE_END:%s rc=%d",
                     kSuites[i].name, rc);
        hil_platform_println(line);
    }

    std::snprintf(line, sizeof(line), "##srig-exit:%d##", failed ? 1 : 0);
    hil_platform_println(line);

    return failed ? 1 : 0;
}
