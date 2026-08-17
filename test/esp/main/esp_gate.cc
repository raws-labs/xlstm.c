/* Entry point for the emulated ESP32-S3 gate (`make test-esp`).
 *
 * Runs the four golden-vector suites against the `esp` SIMD backend on a
 * QEMU-emulated ESP32-S3, plus the two fast-path checks below, and prints
 * one sentinel line that ../qemu_gate.sh turns into the container's exit
 * code.
 *
 * Deliberately thin: no UART driver takeover, no trigger-byte handshake, no
 * timing pass, no clock guard, no provenance banner. Those belong to a rig
 * driving real silicon and measuring it. Nothing here needs them - QEMU's
 * serial output is captured from the first byte, and an emulated core has
 * no cycle count worth printing.
 * =========================================================================*/

#include "test_config.h"
#include "xlstm_simd.h"
/* The scalar bodies themselves, not a copy of them: the check below compares
 * the accelerated matvec against the same text every backend is defined
 * against. */
#include "xlstm_simd_scalar.h"

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
extern "C" unsigned long xlstm_esp_matvec_s8_fast;
extern "C" unsigned long xlstm_esp_matvec_s8_scalar;

namespace {

/* --- f32 fast-path check -------------------------------------------------
 *
 * The first of the two bodies this backend accelerates: the four-row,
 * 128-bit load blocked matvec inside xlstm_matvec_f32. Which calls reach it
 * is decided by rows and cols alone: at least 7 columns (a scalar prefix of up
 * to 3, then one whole 16-byte group) and at least one whole block of four
 * rows spaced 4 / gcd(cols, 4) apart. Where M and v landed does not enter
 * into it, and that is the property being checked here. The guard this
 * replaced asked for two 16-byte-aligned operands instead, and got them 6
 * times in 76 suite calls - by linker accident, so a relink could have taken
 * even those away and the suites would have gone on passing with no
 * accelerated coverage at all.
 *
 * So this does not rest on the suites. It runs every shape at all four
 * alignments of M and of v and fails the run unless all three hold:
 *
 *   1. the shapes the rule blocks took the blocked path, at every alignment,
 *   2. the shapes it cannot took the scalar body - without this, a guard
 *      stuck at "always fast" would look identical to a correct one,
 *   3. every result matched xlstm_scalar_matvec_f32 BIT FOR BIT. The blocked
 *      body reorders loads, not additions, so this is an equality and not a
 *      tolerance; a tolerance here would hide the one thing the body is
 *      written to avoid, and the f32 goldens have no room for it.
 */

const int kMaxRows = 20;
const int kMaxCols = 64;

/* 16-byte aligned, so +1, +2 and +3 floats are exactly the other three
 * alignments, and 4 floats longer than the largest shape so those views
 * still end in bounds. */
alignas(16) float g_M[kMaxRows * kMaxCols + 4];
alignas(16) float g_v[kMaxCols + 4];
float g_out[kMaxRows];
float g_ref[kMaxRows];

/* Seed for out[], so the check also covers the contract's accumulate
 * semantics (out[i] += row . v) rather than only the product. */
float OutSeed(int i) { return 0.25f * (float)i; }

bool CheckShape(int rows, int cols, int moff, int voff, bool expect_fast) {
    const float* M = g_M + moff;
    const float* v = g_v + voff;
    const unsigned long fast0 = xlstm_esp_matvec_f32_fast;
    const unsigned long scalar0 = xlstm_esp_matvec_f32_scalar;
    bool ok = true;

    for (int i = 0; i < rows; ++i) g_out[i] = g_ref[i] = OutSeed(i);
    xlstm_matvec_f32(M, v, g_out, rows, cols);
    xlstm_scalar_matvec_f32(M, v, g_ref, rows, cols);

    const unsigned long d_fast = xlstm_esp_matvec_f32_fast - fast0;
    const unsigned long d_scalar = xlstm_esp_matvec_f32_scalar - scalar0;
    const unsigned long want_fast = expect_fast ? 1ul : 0ul;
    if (d_fast != want_fast || d_scalar != 1ul - want_fast) {
        std::printf("  FAIL rows=%d cols=%d M+%d v+%d: expected the %s path, "
                    "got fast+%lu scalar+%lu. Which path a call takes is a "
                    "property of its shape; a gate that cannot prove the "
                    "128-bit load ran proves nothing.\n",
                    rows, cols, moff, voff,
                    expect_fast ? "blocked" : "scalar", d_fast, d_scalar);
        ok = false;
    }

    for (int i = 0; i < rows; ++i) {
        if (g_out[i] != g_ref[i]) {
            std::printf("  FAIL rows=%d cols=%d M+%d v+%d out[%d]: got %.9g, "
                        "reference %.9g (diff %.2e). The blocked body reorders "
                        "loads, not additions - this has to be exact.\n",
                        rows, cols, moff, voff, i, (double)g_out[i],
                        (double)g_ref[i],
                        (double)std::fabs(g_out[i] - g_ref[i]));
            ok = false;
            break;
        }
    }
    return ok;
}

bool TestFastPath(void) {
    /* Deterministic, and neither constant nor symmetric - a lane-ordering
     * defect in the 4-wide load has to be able to show up. */
    for (int i = 0; i < kMaxRows * kMaxCols + 4; ++i)
        g_M[i] = 0.5f - (float)((i * 37) % 71) / 70.0f;
    for (int j = 0; j < kMaxCols + 4; ++j)
        g_v[j] = (float)((j * 53) % 31) / 15.0f - 1.0f;

    /* Spelled out rather than recomputed from the kernel's own formula: a
     * check that derives the rule the same way the kernel does cannot fail
     * when the rule changes. cols straddles the 4-column group both ways,
     * and the last four entries vary the row count rather than the width. */
    static const struct { int rows, cols; bool fast; } kShapes[] = {
        {20, 1, false}, {20, 2, false}, {20, 3, false}, /* under one group */
        {20, 4, false}, {20, 6, false},  /* a group only at some alignments */
        {20, 7, true},  {20, 8, true},  {20, 9, true},
        {20, 16, true}, {20, 17, true}, {20, 64, true},
        {8, 17, false}, /* odd cols blocks 4 rows apart: 16 rows minimum */
        {8, 16, true},  {4, 15, false}, {4, 16, true},
    };
    const int kShapeCount = (int)(sizeof kShapes / sizeof kShapes[0]);
    bool ok = true;

    for (int s = 0; s < kShapeCount; ++s) {
        for (int moff = 0; moff < 4; ++moff) {
            for (int voff = 0; voff < 4; ++voff) {
                ok &= CheckShape(kShapes[s].rows, kShapes[s].cols, moff, voff,
                                 kShapes[s].fast);
            }
        }
    }
    std::printf("  %d shapes x 16 alignments, all bit-exact against "
                "xlstm_scalar_matvec_f32\n", kShapeCount);
    return ok;
}

/* --- INT8 fast-path check ------------------------------------------------
 *
 * Same three assertions as above, for the kernel that assembles each
 * 16-column group out of the two aligned blocks holding it and multiplies it
 * with EE.VMULAS.S8.ACCX. Two things differ from the f32 case and both are
 * worth being explicit about:
 *
 *   - Alignment does not enter the dispatch at all, so the expectation is
 *     the same at all 256 pairings of M and v. The f32 body has to block
 *     rows to share one scalar prefix; this one never seeks alignment, so a
 *     16-byte group does not cost it the odd sizes. Every column of every
 *     row of a vector-body call runs on the 16-lane MAC, H = 17 included.
 *   - What leaves the vector body is a cols of 0 or less, or a zero point
 *     two int8 lanes cannot carry (|v_zp| > 254). The zero point is folded
 *     into a constant vector rather than subtracted from v, so it is part of
 *     the dispatch here in a way it never is for f32.
 *
 * Bit-exactness is again an equality and not a tolerance, and here there is
 * not even a rounding argument to have: these are integers. The vector body
 * regroups an exact sum of exact products, so any difference at all is a
 * defect.
 */

alignas(16) int8_t g_Mi[kMaxRows * kMaxCols + 16];
alignas(16) int8_t g_vi[kMaxCols + 16];
int32_t g_outi[kMaxRows];
int32_t g_refi[kMaxRows];

bool CheckShapeS8(int rows, int cols, int32_t zp, int moff, int voff,
                  bool expect_fast) {
    const int8_t* M = g_Mi + moff;
    const int8_t* v = g_vi + voff;
    const unsigned long fast0 = xlstm_esp_matvec_s8_fast;
    const unsigned long scalar0 = xlstm_esp_matvec_s8_scalar;
    bool ok = true;

    /* A sentinel rather than zero: this contract overwrites out[] instead of
     * accumulating into it, so a row the kernel never wrote has to show up
     * as a mismatch and not as a plausible-looking 0. */
    for (int i = 0; i < rows; ++i) g_outi[i] = g_refi[i] = 0x5A5A5A5A;
    xlstm_matvec_s8(M, v, g_outi, rows, cols, zp);
    xlstm_scalar_matvec_s8(M, v, g_refi, rows, cols, zp);

    const unsigned long d_fast = xlstm_esp_matvec_s8_fast - fast0;
    const unsigned long d_scalar = xlstm_esp_matvec_s8_scalar - scalar0;
    const unsigned long want_fast = expect_fast ? 1ul : 0ul;
    if (d_fast != want_fast || d_scalar != 1ul - want_fast) {
        std::printf("  FAIL s8 rows=%d cols=%d zp=%ld M+%d v+%d: expected the "
                    "%s path, got fast+%lu scalar+%lu. Which path a call takes "
                    "is a property of its cols and zero point; a gate that "
                    "cannot prove EE.VMULAS.S8.ACCX ran proves nothing.\n",
                    rows, cols, (long)zp, moff, voff,
                    expect_fast ? "vector" : "scalar", d_fast, d_scalar);
        ok = false;
    }

    for (int i = 0; i < rows; ++i) {
        if (g_outi[i] != g_refi[i]) {
            std::printf("  FAIL s8 rows=%d cols=%d zp=%ld M+%d v+%d out[%d]: "
                        "got %ld, reference %ld. These are integers - the "
                        "vector body regroups an exact sum of exact products, "
                        "so any difference at all is a defect.\n",
                        rows, cols, (long)zp, moff, voff, i,
                        (long)g_outi[i], (long)g_refi[i]);
            ok = false;
            break;
        }
    }
    return ok;
}

bool TestFastPathS8(void) {
    /* Deterministic, asymmetric, and reaching both int8 extremes: -128 has
     * no positive counterpart, and a lane-ordering or sign defect has to be
     * able to show up. */
    for (int i = 0; i < kMaxRows * kMaxCols + 16; ++i)
        g_Mi[i] = (int8_t)(((i * 37) % 255) - 128);
    for (int j = 0; j < kMaxCols + 16; ++j)
        g_vi[j] = (int8_t)(((j * 53) % 255) - 128);

    /* Spelled out rather than recomputed from the kernel's own rule. cols
     * straddles the 16-column group both ways; the zero points straddle the
     * two-lane fold bound and include -128, which is what a tensor with no
     * negative values calibrates to and the one int8 value needing the
     * second lane. */
    static const struct { int rows, cols; long zp; bool fast; } kCases[] = {
        {20, 0, 0, false},           /* no columns at all */
        {20, 1, 0, true},   {20, 2, 0, true},   {20, 8, 0, true},
        {20, 15, 0, true},  {20, 16, 0, true},  {20, 17, 0, true},
        {20, 31, 0, true},  {20, 32, 0, true},  {20, 64, 0, true},
        {1, 17, 0, true},   {3, 17, 0, true},   /* fewer rows than a block */
        {20, 17, -128, true},        /* the split zero point */
        {20, 17, 127, true},  {20, 17, -127, true},
        {20, 64, -128, true}, {20, 16, -128, true}, {20, 1, -128, true},
        {20, 17, 254, true},  {20, 17, -254, true},  /* the fold bound */
        {20, 17, 255, false}, {20, 17, -255, false}, /* just outside it */
        {20, 17, 1000, false},
    };
    const int kCaseCount = (int)(sizeof kCases / sizeof kCases[0]);
    bool ok = true;

    for (int s = 0; s < kCaseCount; ++s) {
        for (int moff = 0; moff < 16; ++moff) {
            for (int voff = 0; voff < 16; ++voff) {
                ok &= CheckShapeS8(kCases[s].rows, kCases[s].cols,
                                   (int32_t)kCases[s].zp, moff, voff,
                                   kCases[s].fast);
            }
        }
    }
    std::printf("  %d cases x 256 alignment pairings, all bit-exact against "
                "xlstm_scalar_matvec_s8\n", kCaseCount);
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

        std::printf("[ RUN      ] esp fast path (int8)\n");
        if (TestFastPathS8()) {
            std::printf("[       OK ] esp fast path (int8)\n");
        } else {
            std::printf("[  FAILED  ] esp fast path (int8)\n");
            rc = 1;
        }

        const unsigned long fast0 = xlstm_esp_matvec_f32_fast;
        const unsigned long scalar0 = xlstm_esp_matvec_f32_scalar;
        const unsigned long qfast0 = xlstm_esp_matvec_s8_fast;
        const unsigned long qscalar0 = xlstm_esp_matvec_s8_scalar;

        rc |= slstm_test_main();
        rc |= mlstm_test_main();
        rc |= slstm_s8_test_main();
        rc |= mlstm_s8_test_main();

        /* Reported, not asserted. This number is now a property of the case
         * list - every call with 7 or more columns is blocked, and the rest
         * are the H and I of 1 to 4 - so asserting it would only pin down
         * reference_data.h, which is not what this gate is for. TestFastPath
         * above is the assertion; this is here so a reader of a green log
         * can see how much of the run was accelerated, and what the calls
         * that were not have in common. */
        const unsigned long fast = xlstm_esp_matvec_f32_fast - fast0;
        const unsigned long scalar = xlstm_esp_matvec_f32_scalar - scalar0;
        const unsigned long qfast = xlstm_esp_matvec_s8_fast - qfast0;
        const unsigned long qscalar = xlstm_esp_matvec_s8_scalar - qscalar0;
        std::printf("XLSTM_ESP_FASTPATH: the suites called xlstm_matvec_f32 "
                    "%lu times, %lu of them blocked (the rest are under 7 "
                    "columns wide), and xlstm_matvec_s8 %lu times, %lu of "
                    "them on EE.VMULAS.S8.ACCX - and a vector-body INT8 call "
                    "runs every column of every row there, at any alignment. "
                    "xlstm_vecmat_f32 and xlstm_rank1_update_f32 are scalar C "
                    "in this backend.\n",
                    fast + scalar, fast, qfast + qscalar, qfast);
    }

    std::printf("##xlstm-esp-gate:%d##\n", rc ? 1 : 0);
    std::fflush(stdout);
}
