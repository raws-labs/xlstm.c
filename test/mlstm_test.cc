/* mLSTM kernel unit tests — standalone (no TFLM dependency)
 *
 * Tests the core mLSTM cell computation against reference values
 * generated from the NX-AI/xlstm reference (recurrent_step_stabilized_simple).
 *
 * Build:
 *   make test
 * =========================================================================*/

#include "mlstm.h"
#include "test_util.h"

// ============================================================================
// Reference test data — generated from NX-AI/xlstm reference
// Regenerate: make reference
// ============================================================================

#include "reference_data.h"

// ============================================================================
// Test cases
// ============================================================================

constexpr float kTolerance = 1e-5f;

bool TestMlstmSingleTimestepZeroState() {
    const int B = 1, T = 1, I = 3, H = 2;

    float y[H] = {0};
    float C[H * H] = {0};
    float n[H] = {0};
    float m_state[1] = {0};
    float output[T * H] = {0};
    float scratch[4 * H + 2] = {0};
    MlstmParams params = {0.0f};

    mlstm_eval_f32(kMTest1_input, kMTest1_W, kMTest1_b,
                   y, C, n, m_state, output, scratch, B, T, I, H, &params);

    bool ok = true;
    ok &= ExpectNear("y", kMTest1_expected_y, y, H, kTolerance);
    ok &= ExpectNear("C", kMTest1_expected_C, C, H * H, kTolerance);
    ok &= ExpectNear("n", kMTest1_expected_n, n, H, kTolerance);
    ok &= ExpectNear("m", kMTest1_expected_m, m_state, 1, kTolerance);
    ok &= ExpectNear("output", kMTest1_expected_y, output, H, kTolerance);
    return ok;
}

bool TestMlstmMultipleTimesteps() {
    const int B = 1, T = 3, I = 3, H = 2;

    float y[H] = {0};
    float C[H * H] = {0};
    float n[H] = {0};
    float m_state[1] = {0};
    float output[T * H] = {0};
    float scratch[4 * H + 2] = {0};
    MlstmParams params = {0.0f};

    mlstm_eval_f32(kMTest2_input, kMTest1_W, kMTest1_b,
                   y, C, n, m_state, output, scratch, B, T, I, H, &params);

    bool ok = true;
    ok &= ExpectNear("y_final", kMTest2_expected_y, y, H, kTolerance);
    ok &= ExpectNear("C_final", kMTest2_expected_C, C, H * H, kTolerance);
    ok &= ExpectNear("n_final", kMTest2_expected_n, n, H, kTolerance);
    ok &= ExpectNear("m_final", kMTest2_expected_m, m_state, 1, kTolerance);
    ok &= ExpectNear("output_all", kMTest2_expected_output, output, T * H, kTolerance);
    return ok;
}

bool TestMlstmOverflowPrevention() {
    const int B = 1, T = 1, I = 3, H = 2;

    float y[H] = {0};
    float C[H * H] = {0};
    float n[H] = {0};
    float m_state[1] = {0};
    float output[T * H] = {0};
    float scratch[4 * H + 2] = {0};
    MlstmParams params = {0.0f};

    mlstm_eval_f32(kMTest3_input, kMTest3_W, kMTest3_b,
                   y, C, n, m_state, output, scratch, B, T, I, H, &params);

    bool ok = true;
    ok &= ExpectFinite("y", y, H);
    ok &= ExpectFinite("C", C, H * H);
    ok &= ExpectFinite("n", n, H);
    ok &= ExpectFinite("m", m_state, 1);

    ok &= ExpectNear("y", kMTest3_expected_y, y, H, kTolerance);
    ok &= ExpectNear("C", kMTest3_expected_C, C, H * H, kTolerance);
    ok &= ExpectNear("n", kMTest3_expected_n, n, H, kTolerance);
    ok &= ExpectNear("m", kMTest3_expected_m, m_state, 1, kTolerance);
    return ok;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::printf("[==========] Running mLSTM kernel tests\n");

    RUN_TEST(TestMlstmSingleTimestepZeroState);
    RUN_TEST(TestMlstmMultipleTimesteps);
    RUN_TEST(TestMlstmOverflowPrevention);

    std::printf("[==========] %d/%d tests passed\n", g_tests_passed, g_tests_run);
    return g_tests_passed == g_tests_run ? 0 : 1;
}
