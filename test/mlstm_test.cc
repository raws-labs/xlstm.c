/* mLSTM kernel unit tests - standalone (no TFLM dependency)
 *
 * Tests the core mLSTM cell computation against reference values
 * generated from the NX-AI/xlstm reference (recurrent_step_stabilized_simple).
 *
 * Build:
 *   make test
 * =========================================================================*/

#include "mlstm.h"
#include "test_config.h"
#include "test_util.h"
#include "reference_data.h"

#include <cstdio>

/* Static buffers sized for the largest case. See test_config.h for what
 * XLSTM_TEST_MAX_H bounds and how it differs from XLSTM_MAX_HIDDEN. */
static float g_y[XLSTM_TEST_MAX_H], g_n[XLSTM_TEST_MAX_H], g_m[XLSTM_TEST_MAX_H];
static float g_C[XLSTM_TEST_MAX_H * XLSTM_TEST_MAX_H];
static float g_output[3 * XLSTM_TEST_MAX_H];
static float g_scratch[4 * XLSTM_TEST_MAX_H + 2];

static bool RunMlstmCase(const XlstmRefCase* tc) {
    /* DQ is the query/key width and DV the value width; H equals DV, and the
     * two differ only for the rectangular cases. n is sized by DQ, y and the
     * output by DV, and C by their product - sizing any of them from one
     * number is the mistake these cases exist to catch. */
    const int DQ = tc->DQ, DV = tc->DV, T = tc->T;

    /* g_output holds T*DV, not B*T*DV, and the state buffers hold one batch;
     * the output assertion below checks batch 0's slice only. Every case in
     * reference_data.h is B=1, but a future B=2 case at H=256 would overrun
     * g_output rather than fail. Fail loudly instead. */
    if (tc->B != 1) {
        std::printf("  FAIL %s: B=%d, but this runner is written for B=1 only\n",
                    tc->name, tc->B);
        return false;
    }

    for (int i = 0; i < DV; ++i) { g_y[i] = 0; g_m[i] = 0; }
    for (int i = 0; i < DQ; ++i) g_n[i] = 0;
    for (int i = 0; i < DQ * DV; ++i) g_C[i] = 0;
    for (int i = 0; i < T * DV; ++i) g_output[i] = 0;
    for (int i = 0; i < 2 * DQ + 2 * DV + 2; ++i) g_scratch[i] = 0;

    MlstmParams params = {0.0f};
    mlstm_eval_f32(tc->input, tc->W, tc->b,
                   g_y, g_C, g_n, g_m, g_output, g_scratch,
                   tc->B, T, tc->I, DQ, DV, &params);

    bool ok = true;
    ok &= ExpectFinite("y", g_y, DV);
    ok &= ExpectFinite("C", g_C, DQ * DV);
    ok &= ExpectFinite("n", g_n, DQ);
    ok &= ExpectFinite("m", g_m, 1);
    ok &= ExpectNear("y", tc->expected_y, g_y, DV, tc->tol_f32);
    if (tc->expected_state)
        ok &= ExpectNear("C", tc->expected_state, g_C, DQ * DV, tc->tol_f32);
    ok &= ExpectNear("n", tc->expected_n, g_n, DQ, tc->tol_f32);
    ok &= ExpectNear("m", tc->expected_m, g_m, 1, tc->tol_f32);
    /* Cases with no stored expected_output (MTest1, MTest3) have T=1, where
     * output == y, so fall back to expected_y rather than leaving output
     * unchecked. A future T>1 case without expected_output would correctly
     * skip this (the fallback only fires when T == 1). */
    const float* out_ref = tc->expected_output ? tc->expected_output : tc->expected_y;
    if (tc->expected_output || T == 1)
        ok &= ExpectNear("output", out_ref, g_output, T * DV, tc->tol_f32);
    return ok;
}

#ifndef XLSTM_TEST_MAIN
#define XLSTM_TEST_MAIN main
#endif

int XLSTM_TEST_MAIN(void) {
    std::printf("[==========] Running mLSTM kernel tests\n");

    for (int i = 0; i < kMlstmCasesCount; ++i) {
        const XlstmRefCase* tc = &kMlstmCases[i];
        g_tests_run++;
        std::printf("[ RUN      ] mLSTM %s (%dx%d, T=%d)\n", tc->name, tc->DQ,
                tc->DV, tc->T);
        if (RunMlstmCase(tc)) {
            g_tests_passed++;
            std::printf("[       OK ] mLSTM %s\n", tc->name);
        } else {
            std::printf("[  FAILED  ] mLSTM %s\n", tc->name);
        }
    }

    std::printf("[==========] %d/%d tests passed\n", g_tests_passed, g_tests_run);
    return g_tests_passed == g_tests_run ? 0 : 1;
}
