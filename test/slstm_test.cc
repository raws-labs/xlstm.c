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
    if (tc->expected_output)
        ok &= ExpectNear("output", tc->expected_output, g_output, T * H, tc->tol_f32);
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

    std::printf("[==========] %d/%d tests passed\n", g_tests_passed, g_tests_run);
    return g_tests_passed == g_tests_run ? 0 : 1;
}
