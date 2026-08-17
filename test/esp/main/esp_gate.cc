/* Entry point for the emulated ESP32-S3 gate (`make test-esp`).
 *
 * Runs the four golden-vector suites against the `esp` SIMD backend on a
 * QEMU-emulated ESP32-S3, plus the fast-path check below, and prints one
 * sentinel line that ../qemu_gate.sh turns into the container's exit code.
 *
 * Deliberately thin: no UART driver takeover, no trigger-byte handshake, no
 * timing pass, no clock guard, no provenance banner. Those belong to a rig
 * driving real silicon and measuring it. Nothing here needs them - QEMU's
 * serial output is captured from the first byte, and an emulated core has
 * no cycle count worth printing.
 * =========================================================================*/

#include "test_config.h"
#include "xlstm_simd.h"

#include <cmath>
#include <cstdio>
#include <cstring>

/* The four suite entry points, renamed per translation unit by
 * main/CMakeLists.txt. All five files are C++, so these link straight
 * against the renamed definitions with no header in between. */
extern int slstm_test_main(void);
extern int mlstm_test_main(void);
extern int slstm_s8_test_main(void);
extern int mlstm_s8_test_main(void);

/* Defined in src/xlstm_simd_esp.c under XLSTM_ESP_FASTPATH_COUNTERS, which
 * main/CMakeLists.txt sets. Referenced unconditionally so that losing the
 * define is a link error rather than a gate that stops checking. */
extern "C" unsigned long xlstm_esp_matvec_f32_fast;
extern "C" unsigned long xlstm_esp_matvec_f32_scalar;

namespace {

/* --- Fast-path check ----------------------------------------------------
 *
 * Everything this backend accelerates is one call - ESP-DSP's dot product
 * inside xlstm_matvec_f32 - taken only when cols is a multiple of 4 and both
 * operands are 16-byte aligned. That is a property of the buffers a caller
 * happens to pass, so the four suites can pass in full while the accelerated
 * body never executes, and very nearly do: their state and scratch arrays
 * are not 16-byte aligned, so every recurrent matvec runs scalar, and the
 * few input matvecs that do reach the fast path reach it because the linker
 * put two arrays on a 16-byte boundary. A relink moves that, and a gate
 * resting on it would go on passing with no accelerated coverage left.
 *
 * So this does not rest on the suites. It calls the kernel twice with
 * buffers it aligns itself and fails the run unless all three hold:
 *
 *   1. the aligned call takes the fast path,
 *   2. the deliberately misaligned call takes the scalar one - without this,
 *      a guard stuck at "always fast" would look identical to a correct one,
 *   3. both agree with a dot product computed here, so the accelerated
 *      result is checked and not merely counted.
 */

const int kRows = 8;
const int kCols = 64; /* multiple of 4: the length half of the guard */

/* +4 floats of headroom so the misaligned view (g_M + 1) still ends in
 * bounds. alignas gives the aligned view its half of the guard; the
 * misaligned view is then exactly 4 bytes off, which fails it. */
alignas(16) float g_M[kRows * kCols + 4];
alignas(16) float g_v[kCols];
float g_out[kRows];

/* Seed for out[], so the check also covers the contract's accumulate
 * semantics (out[i] += dot) rather than only the product. */
float OutSeed(int i) { return 0.25f * (float)i; }

bool CheckPath(const char* what, const float* M, const float* v,
               bool expect_fast) {
    const unsigned long fast0 = xlstm_esp_matvec_f32_fast;
    const unsigned long scalar0 = xlstm_esp_matvec_f32_scalar;
    bool ok = true;

    for (int i = 0; i < kRows; ++i) g_out[i] = OutSeed(i);
    xlstm_matvec_f32(M, v, g_out, kRows, kCols);

    const unsigned long d_fast = xlstm_esp_matvec_f32_fast - fast0;
    const unsigned long d_scalar = xlstm_esp_matvec_f32_scalar - scalar0;
    const unsigned long want_fast = expect_fast ? 1ul : 0ul;
    if (d_fast != want_fast || d_scalar != 1ul - want_fast) {
        std::printf("  FAIL %s: expected the %s path, got fast+%lu scalar+%lu. "
                    "The accelerated dot product is the only thing this "
                    "backend accelerates; a gate that cannot prove it ran "
                    "proves nothing.\n",
                    what, expect_fast ? "accelerated" : "scalar",
                    d_fast, d_scalar);
        ok = false;
    }

    for (int i = 0; i < kRows; ++i) {
        float ref = OutSeed(i);
        for (int j = 0; j < kCols; ++j) ref += M[i * kCols + j] * v[j];
        /* The accelerated body sums into four partial accumulators and the
         * reference above sums in order, so this is a float-grouping
         * tolerance, not a correctness allowance. */
        const float diff = std::fabs(ref - g_out[i]);
        if (diff > 1e-5f + 1e-5f * std::fabs(ref)) {
            std::printf("  FAIL %s[%d]: expected %.8f, got %.8f (diff %.2e)\n",
                        what, i, ref, g_out[i], diff);
            ok = false;
        }
    }
    return ok;
}

bool TestFastPath(void) {
    /* Deterministic, and neither constant nor symmetric - a lane-ordering
     * defect in the 4-wide body has to be able to show up. */
    for (int i = 0; i < kRows * kCols + 4; ++i)
        g_M[i] = 0.5f - (float)((i * 37) % 71) / 70.0f;
    for (int j = 0; j < kCols; ++j)
        g_v[j] = (float)((j * 53) % 31) / 15.0f - 1.0f;

    bool ok = CheckPath("aligned", g_M, g_v, true);
    ok &= CheckPath("misaligned", g_M + 1, g_v, false);
    return ok;
}

} /* namespace */

extern "C" void app_main(void) {
    const char* backend = xlstm_simd_backend();
    int rc = 0;

    std::printf("XLSTM_ESP_GATE: backend=%s test_max_h=%d\n",
                backend, XLSTM_TEST_MAX_H);

    /* An image that had silently linked src/xlstm_simd_ref.c would pass
     * every suite below and prove nothing about this backend. Refuse before
     * running anything, so that failure can never read as a green run. */
    if (std::strcmp(backend, "esp") != 0) {
        std::printf("FATAL: linked SIMD backend is \"%s\", not \"esp\" - "
                    "refusing to run. A pass here would be a pass for the "
                    "wrong backend.\n", backend);
        rc = 1;
    } else {
        std::printf("[ RUN      ] esp fast path\n");
        if (TestFastPath()) {
            std::printf("[       OK ] esp fast path\n");
        } else {
            std::printf("[  FAILED  ] esp fast path\n");
            rc = 1;
        }

        const unsigned long fast0 = xlstm_esp_matvec_f32_fast;
        const unsigned long scalar0 = xlstm_esp_matvec_f32_scalar;

        rc |= slstm_test_main();
        rc |= mlstm_test_main();
        rc |= slstm_s8_test_main();
        rc |= mlstm_s8_test_main();

        /* Reported, not asserted. Which suite matvecs clear the alignment
         * guard is decided by where the linker placed the runners' arrays,
         * so failing on this number would make the gate flake on an
         * unrelated relink. TestFastPath above is the assertion; this is
         * here so a reader of a green log can see how little of the run was
         * actually accelerated. */
        const unsigned long fast = xlstm_esp_matvec_f32_fast - fast0;
        const unsigned long scalar = xlstm_esp_matvec_f32_scalar - scalar0;
        std::printf("XLSTM_ESP_FASTPATH: the suites called xlstm_matvec_f32 "
                    "%lu times, %lu of them accelerated. Every other kernel "
                    "in this backend is scalar C.\n",
                    fast + scalar, fast);
    }

    std::printf("##xlstm-esp-gate:%d##\n", rc ? 1 : 0);
    std::fflush(stdout);
}
