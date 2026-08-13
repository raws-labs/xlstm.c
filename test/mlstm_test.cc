/* mLSTM kernel unit tests - standalone (no TFLM dependency)
 *
 * Tests the core mLSTM cell computation against reference values
 * generated from the NX-AI/xlstm reference (recurrent_step_stabilized_simple).
 *
 * Build:
 *   make test
 * =========================================================================*/

#include "mlstm.h"
#include "test_util.h"
#include "reference_data.h"

#include <cstdio>

/* Static buffers sized for the largest case. XLSTM_MAX_HIDDEN is 256. */
static float g_y[256], g_n[256], g_m[256];
static float g_C[256 * 256];
static float g_output[3 * 256];
static float g_scratch[4 * 256 + 2];

static bool RunMlstmCase(const XlstmRefCase* tc) {
    const int H = tc->H, T = tc->T;

    for (int i = 0; i < H; ++i) { g_y[i] = 0; g_n[i] = 0; g_m[i] = 0; }
    for (int i = 0; i < H * H; ++i) g_C[i] = 0;
    for (int i = 0; i < T * H; ++i) g_output[i] = 0;
    for (int i = 0; i < 4 * H + 2; ++i) g_scratch[i] = 0;

    MlstmParams params = {0.0f};
    mlstm_eval_f32(tc->input, tc->W, tc->b,
                   g_y, g_C, g_n, g_m, g_output, g_scratch,
                   tc->B, T, tc->I, H, &params);

    bool ok = true;
    ok &= ExpectFinite("y", g_y, H);
    ok &= ExpectFinite("C", g_C, H * H);
    ok &= ExpectFinite("n", g_n, H);
    ok &= ExpectFinite("m", g_m, 1);
    ok &= ExpectNear("y", tc->expected_y, g_y, H, tc->tol_f32);
    if (tc->expected_state)
        ok &= ExpectNear("C", tc->expected_state, g_C, H * H, tc->tol_f32);
    ok &= ExpectNear("n", tc->expected_n, g_n, H, tc->tol_f32);
    ok &= ExpectNear("m", tc->expected_m, g_m, 1, tc->tol_f32);
    /* Cases with no stored expected_output (MTest1, MTest3) have T=1, where
     * output == y, so fall back to expected_y rather than leaving output
     * unchecked. A future T>1 case without expected_output would correctly
     * skip this (the fallback only fires when T == 1). */
    const float* out_ref = tc->expected_output ? tc->expected_output : tc->expected_y;
    if (tc->expected_output || T == 1)
        ok &= ExpectNear("output", out_ref, g_output, T * H, tc->tol_f32);
    return ok;
}

int main() {
    std::printf("[==========] Running mLSTM kernel tests\n");

    for (int i = 0; i < kMlstmCasesCount; ++i) {
        const XlstmRefCase* tc = &kMlstmCases[i];
        g_tests_run++;
        std::printf("[ RUN      ] mLSTM %s (H=%d, T=%d)\n", tc->name, tc->H, tc->T);
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
