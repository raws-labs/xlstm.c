/* sLSTM kernel unit tests - standalone (no TFLM dependency)
 *
 * Tests the core sLSTM cell computation against reference values
 * generated from the NX-AI/xlstm PyTorch reference (vanilla backend).
 *
 * Build:
 *   make test
 * =========================================================================*/

#include "slstm.h"
#include "test_util.h"
#include "reference_data.h"

#include <cstdio>

/* Static buffers sized for the largest case. XLSTM_MAX_HIDDEN is 256. */
static float g_y[256], g_c[256], g_n[256], g_m[256];
static float g_output[3 * 256];
static float g_scratch[4 * 256];

static bool RunSlstmCase(const XlstmRefCase* tc) {
    const int H = tc->H, T = tc->T;

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
 * the caller's outer loop over head-sliced weights (.docs/SCOPE.md section 6).
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

int main() {
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

    std::printf("[==========] %d/%d tests passed\n", g_tests_passed, g_tests_run);
    return g_tests_passed == g_tests_run ? 0 : 1;
}
