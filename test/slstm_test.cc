/* sLSTM kernel unit tests - standalone (no TFLM dependency)
 *
 * Tests the core sLSTM cell computation against reference values
 * generated from the NX-AI/xlstm PyTorch reference (vanilla backend).
 *
 * Build:
 *   make test
 * =========================================================================*/

#include "slstm.h"
#include "test_config.h"
#include "test_util.h"
#include "reference_data.h"

#include <cstdio>

/* Static buffers sized for the largest case. See test_config.h for what
 * XLSTM_TEST_MAX_H bounds and how it differs from XLSTM_MAX_HIDDEN. */
static float g_y[XLSTM_TEST_MAX_H], g_c[XLSTM_TEST_MAX_H], g_n[XLSTM_TEST_MAX_H], g_m[XLSTM_TEST_MAX_H];
static float g_output[3 * XLSTM_TEST_MAX_H];
static float g_scratch[4 * XLSTM_TEST_MAX_H];

static bool RunSlstmCase(const XlstmRefCase* tc) {
    const int H = tc->H, T = tc->T;

    /* g_output holds T*H, not B*T*H, and the state buffers hold one batch;
     * the output assertion below checks batch 0's slice only. Every case in
     * reference_data.h is B=1, but a future B=2 case at H=256 would overrun
     * g_output rather than fail. Fail loudly instead. */
    if (tc->B != 1) {
        std::printf("  FAIL %s: B=%d, but this runner is written for B=1 only\n",
                    tc->name, tc->B);
        return false;
    }

    /* The same argument for the other dimension the static buffers are sized
     * by. XLSTM_TEST_MAX_H is 256 by default and forced to 64 for the board
     * images, so a case wider than the knob overruns every buffer above
     * rather than failing. */
    if (H > XLSTM_TEST_MAX_H) {
        std::printf("  FAIL %s: H=%d exceeds XLSTM_TEST_MAX_H=%d, which sizes "
                    "this runner's buffers\n", tc->name, H, XLSTM_TEST_MAX_H);
        return false;
    }

    for (int i = 0; i < H; ++i) { g_y[i] = 0; g_c[i] = 0; g_n[i] = 0; g_m[i] = 0; }
    for (int i = 0; i < T * H; ++i) g_output[i] = 0;
    for (int i = 0; i < 4 * H; ++i) g_scratch[i] = 0;

    SlstmParams params = {0.0f};
    slstm_eval_f32(tc->input, tc->W, tc->R, tc->b,
                   g_y, g_c, g_n, g_m, g_output, g_scratch,
                   tc->B, T, tc->I, H, &params);

    bool ok = true;
    ok &= ExpectFinite("y", g_y, H);
    ok &= ExpectFinite("c", g_c, H);
    ok &= ExpectFinite("n", g_n, H);
    ok &= ExpectFinite("m", g_m, H);
    ok &= ExpectNear("y", tc->expected_y, g_y, H, tc->tol_f32);
    if (tc->expected_state)
        ok &= ExpectNear("c", tc->expected_state, g_c, H, tc->tol_f32);
    ok &= ExpectNear("n", tc->expected_n, g_n, H, tc->tol_f32);
    ok &= ExpectNear("m", tc->expected_m, g_m, H, tc->tol_f32);
    /* Cases with no stored expected_output (Test1, Test3) have T=1, where
     * output == y, so fall back to expected_y rather than leaving output
     * unchecked. A future T>1 case without expected_output would correctly
     * skip this (the fallback only fires when T == 1). */
    const float* out_ref = tc->expected_output ? tc->expected_output : tc->expected_y;
    if (tc->expected_output || T == 1)
        ok &= ExpectNear("output", out_ref, g_output, T * H, tc->tol_f32);
    return ok;
}

/* Multi-head contract: hidden_size is the PER-HEAD width, and multi-head is
 * the caller's outer loop over head-sliced weights.
 *
 * This is the only test that is not vacuous on that point - every other case
 * is num_heads=1, where per-head width and model width coincide.
 *
 * The reference tensors here are the FUSED ones from a real num_heads=2 cell,
 * and the reference outputs are that cell's own. The slicing happens below,
 * in C, deliberately: it is the thing under test. The reference packs the
 * fused weight rows GATE-major, so head h's four gate blocks are strided
 * across the matrix rather than contiguous - slicing rows
 * [h*4*DH, (h+1)*4*DH) instead, which is the obvious guess, silently yields a
 * different model. The rule was established empirically (not read off the
 * reference source, whose declared parameter shapes are rewritten at
 * construction) by test/derive_multihead_layout.py.
 *
 * Head2/Head2b in the table above are these same weights pre-sliced in
 * Python; this test does not use them, so that a wrong slicing here cannot be
 * masked by a matching wrong slicing in the generator. */
static bool TestHeadComposition() {
    const int B = 1, T = kHead2_T, I = kHead2_I;
    const int DH = kHead2_DH, NH = kHead2_NH, Hf = kHead2_Hf;

    float joined_y[kHead2_Hf] = {0};
    float joined_output[kHead2_NH * kHead2_T * kHead2_DH] = {0};

    for (int h = 0; h < NH; ++h) {
        /* Per-head weight slice, in this library's flat [4*DH, I] packing. */
        float Wh[4 * kHead2_DH * kHead2_I];
        float bh[4 * kHead2_DH];
        for (int g = 0; g < 4; ++g) {
            for (int j = 0; j < DH; ++j) {
                const int src = g * Hf + h * DH + j;  /* fused row */
                const int dst = g * DH + j;           /* per-head row */
                for (int k = 0; k < I; ++k)
                    Wh[dst * I + k] = kHead2Fused_W[src * I + k];
                bh[dst] = kHead2Fused_b[src];
            }
        }
        /* The reference carries no cross-head recurrence, so head h's
         * recurrent matrix is already a contiguous [4*DH, DH] block. */
        const float* Rh = kHead2Fused_R + h * (4 * DH * DH);

        float y[kHead2_DH] = {0}, c[kHead2_DH] = {0};
        float n[kHead2_DH] = {0}, m[kHead2_DH] = {0};
        float out[kHead2_T * kHead2_DH] = {0};
        float scratch[4 * kHead2_DH] = {0};
        SlstmParams params = {0.0f};

        slstm_eval_f32(kHead2_input, Wh, Rh, bh,
                       y, c, n, m, out, scratch, B, T, I, DH, &params);

        for (int j = 0; j < DH; ++j) joined_y[h * DH + j] = y[j];
        for (int j = 0; j < T * DH; ++j) joined_output[h * T * DH + j] = out[j];
    }

    bool ok = ExpectFinite("joined_y", joined_y, Hf);
    ok &= ExpectNear("joined_y", kHead2_expected_y_joined, joined_y, Hf, 1e-5f);
    ok &= ExpectNear("joined_output", kHead2_expected_output_joined,
                     joined_output, NH * T * DH, 1e-5f);
    return ok;
}

#ifndef XLSTM_TEST_MAIN
#define XLSTM_TEST_MAIN main
#endif

/* include/slstm.h states that params may be NULL and that it is equivalent to
 * an all-zero SlstmParams. The kernel spells that as a `params &&` guard
 * (src/slstm.c:87); dropping the guard turns a documented call into a null
 * dereference, and nothing else here passes NULL. Drive the first golden case
 * both ways and require bit-identical state and output. */
static bool TestNullParamsEqualsZeroStruct() {
    const XlstmRefCase* tc = &kSlstmCases[0];
    const int H = tc->H, T = tc->T;
    if (tc->B != 1 || H > XLSTM_TEST_MAX_H) return true;

    static float y_z[XLSTM_TEST_MAX_H], c_z[XLSTM_TEST_MAX_H];
    static float n_z[XLSTM_TEST_MAX_H], m_z[XLSTM_TEST_MAX_H];
    static float out_z[3 * XLSTM_TEST_MAX_H], scratch_z[4 * XLSTM_TEST_MAX_H];
    SlstmParams zero = {0};

    for (int i = 0; i < H; ++i) { g_y[i] = c_z[i] = n_z[i] = m_z[i] = 0; }
    for (int i = 0; i < H; ++i) { g_c[i] = g_n[i] = g_m[i] = 0; y_z[i] = 0; }
    for (int i = 0; i < T * H; ++i) { g_output[i] = out_z[i] = 0; }

    slstm_eval_f32(tc->input, tc->W, tc->R, tc->b, g_y, g_c, g_n, g_m,
                   g_output, g_scratch, 1, T, tc->I, H, NULL);
    slstm_eval_f32(tc->input, tc->W, tc->R, tc->b, y_z, c_z, n_z, m_z,
                   out_z, scratch_z, 1, T, tc->I, H, &zero);

    bool ok = true;
    for (int i = 0; i < T * H; ++i) {
        if (g_output[i] != out_z[i]) {
            std::printf("  FAIL: output[%d] NULL params %.9g, zero struct %.9g\n",
                        i, (double)g_output[i], (double)out_z[i]);
            ok = false;
        }
    }
    const float* a[4] = { g_y, g_c, g_n, g_m };
    const float* b[4] = { y_z, c_z, n_z, m_z };
    const char* nm[4] = { "y", "c", "n", "m" };
    for (int k = 0; k < 4; ++k) {
        for (int i = 0; i < H; ++i) {
            if (a[k][i] != b[k][i]) {
                std::printf("  FAIL: %s[%d] NULL params %.9g, zero struct %.9g\n",
                            nm[k], i, (double)a[k][i], (double)b[k][i]);
                ok = false;
            }
        }
    }
    if (ok) std::printf("  NULL params is bit-identical to an all-zero SlstmParams\n");
    return ok;
}

/* cell_clip is pinned to 0 by every golden case, so the clamp at
 * src/slstm.c:86-88 never runs under the rest of this suite: a clamp that
 * clamped the wrong way, or to the wrong bound, would ship green. Drive one
 * case unclipped to learn its true |c| range, then re-drive it with a clip
 * strictly inside that range and require every element to equal the clamp of
 * the unclipped value. Exact equality rather than a bound: a clamp that pins
 * every element to +clip satisfies |c| <= clip and is still wrong. The case
 * must be T=1, because at T>1 a clamp at one step moves the next step's
 * trajectory and the elementwise relation no longer holds. */
static bool TestCellClipBinds() {
    const XlstmRefCase* tc = &kSlstmCases[0];
    const int H = tc->H, T = tc->T;
    if (tc->B != 1 || H > XLSTM_TEST_MAX_H) return true;
    if (T != 1) {
        std::printf("  FAIL: %s has T=%d; this test needs T=1\n", tc->name, T);
        return false;
    }

    for (int i = 0; i < H; ++i) { g_y[i] = g_c[i] = g_n[i] = g_m[i] = 0; }
    for (int i = 0; i < T * H; ++i) g_output[i] = 0;
    slstm_eval_f32(tc->input, tc->W, tc->R, tc->b, g_y, g_c, g_n, g_m,
                   g_output, g_scratch, 1, T, tc->I, H, NULL);

    float peak = 0.0f;
    for (int i = 0; i < H; ++i) {
        float a = g_c[i] < 0 ? -g_c[i] : g_c[i];
        if (a > peak) peak = a;
    }
    if (peak <= 0.0f) {
        std::printf("  FAIL: unclipped |c| peak is 0, nothing to clamp against\n");
        return false;
    }

    SlstmParams clipped = {0};
    clipped.cell_clip = peak * 0.5f;
    static float y2[XLSTM_TEST_MAX_H], c2[XLSTM_TEST_MAX_H];
    static float n2[XLSTM_TEST_MAX_H], m2[XLSTM_TEST_MAX_H];
    static float out2[3 * XLSTM_TEST_MAX_H], scratch2[4 * XLSTM_TEST_MAX_H];
    for (int i = 0; i < H; ++i) { y2[i] = c2[i] = n2[i] = m2[i] = 0; }
    for (int i = 0; i < T * H; ++i) out2[i] = 0;
    slstm_eval_f32(tc->input, tc->W, tc->R, tc->b, y2, c2, n2, m2,
                   out2, scratch2, 1, T, tc->I, H, &clipped);

    bool ok = true, any_clamped = false;
    const float k = clipped.cell_clip;
    for (int i = 0; i < H; ++i) {
        float want = g_c[i] < -k ? -k : (g_c[i] > k ? k : g_c[i]);
        if (c2[i] != want) {
            std::printf("  FAIL: c[%d] = %.9g, clamp of %.9g to +/-%.9g is %.9g\n",
                        i, (double)c2[i], (double)g_c[i], (double)k, (double)want);
            ok = false;
        }
        float u = g_c[i] < 0 ? -g_c[i] : g_c[i];
        if (u > k) any_clamped = true;
    }
    if (!any_clamped) {
        std::printf("  FAIL: no element exceeded the clip, so nothing was tested\n");
        ok = false;
    }
    /* peak*0.5 only binds on whichever side the large elements sit, so an
     * error in the other bound survives it. Re-drive with a clip below the
     * smallest magnitude, which forces every element to clamp and therefore
     * exercises both bounds whenever the case carries both signs. */
    float smallest = peak;
    for (int i = 0; i < H; ++i) {
        float a = g_c[i] < 0 ? -g_c[i] : g_c[i];
        if (a < smallest) smallest = a;
    }
    SlstmParams tight = {0};
    tight.cell_clip = smallest * 0.5f;
    bool saw_pos = false, saw_neg = false;
    if (tight.cell_clip > 0.0f) {
        for (int i = 0; i < H; ++i) { y2[i] = c2[i] = n2[i] = m2[i] = 0; }
        for (int i = 0; i < T * H; ++i) out2[i] = 0;
        slstm_eval_f32(tc->input, tc->W, tc->R, tc->b, y2, c2, n2, m2,
                       out2, scratch2, 1, T, tc->I, H, &tight);
        const float t = tight.cell_clip;
        for (int i = 0; i < H; ++i) {
            float want = g_c[i] < -t ? -t : (g_c[i] > t ? t : g_c[i]);
            if (c2[i] != want) {
                std::printf("  FAIL: tight clip, c[%d] = %.9g, expected %.9g\n",
                            i, (double)c2[i], (double)want);
                ok = false;
            }
            if (g_c[i] > t) saw_pos = true;
            if (g_c[i] < -t) saw_neg = true;
        }
        if (!saw_pos || !saw_neg) {
            std::printf("  NOTE: %s clamps on one side only (pos=%d neg=%d), "
                        "so the other bound is untested here\n",
                        tc->name, (int)saw_pos, (int)saw_neg);
        }
    }

    if (ok) std::printf("  cell_clip binds: |c| in [%.6g, %.6g], both bounds "
                        "exercised: %s\n", (double)smallest, (double)peak,
                        (saw_pos && saw_neg) ? "yes" : "no");
    return ok;
}

int XLSTM_TEST_MAIN(void) {
    std::printf("[==========] Running sLSTM kernel tests\n");

    for (int i = 0; i < kSlstmCasesCount; ++i) {
        const XlstmRefCase* tc = &kSlstmCases[i];
        g_tests_run++;
        std::printf("[ RUN      ] sLSTM %s (H=%d, T=%d)\n", tc->name, tc->H, tc->T);
        if (RunSlstmCase(tc)) {
            g_tests_passed++;
            std::printf("[       OK ] sLSTM %s\n", tc->name);
        } else {
            std::printf("[  FAILED  ] sLSTM %s\n", tc->name);
        }
    }

    RUN_TEST(TestHeadComposition);
    RUN_TEST(TestNullParamsEqualsZeroStruct);
    RUN_TEST(TestCellClipBinds);

    std::printf("[==========] %d/%d tests passed\n", g_tests_passed, g_tests_run);
    return g_tests_passed == g_tests_run ? 0 : 1;
}
