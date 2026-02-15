/* sLSTM kernel unit tests — standalone (no TFLM dependency)
 *
 * Tests the core sLSTM cell computation against reference values
 * generated from the NX-AI/xlstm PyTorch reference (vanilla backend).
 *
 * Build:
 *   make test
 * =========================================================================*/

#include "slstm.h"
#include "test_util.h"

// ============================================================================
// Reference test data — generated from NX-AI/xlstm reference (vanilla backend)
// Regenerate: make reference
// ============================================================================

#include "reference_data.h"

// ============================================================================
// Test cases
// ============================================================================

constexpr float kTolerance = 1e-5f;

bool TestSingleTimestepZeroState() {
    const int B = 1, T = 1, I = 2, H = 2;

    float y[H] = {0}, c[H] = {0}, n[H] = {0}, m_state[H] = {0};
    float output[T * H] = {0};
    float scratch[4 * H] = {0};
    SlstmParams params = {0.0f};

    slstm_eval_f32(kTest1_input, kTest1_W, kTest1_R, kTest1_b,
                   y, c, n, m_state, output, scratch, B, T, I, H, &params);

    bool ok = true;
    ok &= ExpectNear("y", kTest1_expected_y, y, H, kTolerance);
    ok &= ExpectNear("c", kTest1_expected_c, c, H, kTolerance);
    ok &= ExpectNear("n", kTest1_expected_n, n, H, kTolerance);
    ok &= ExpectNear("m", kTest1_expected_m, m_state, H, kTolerance);
    ok &= ExpectNear("output", kTest1_expected_y, output, H, kTolerance);
    return ok;
}

bool TestMultipleTimesteps() {
    const int B = 1, T = 3, I = 2, H = 2;

    float y[H] = {0}, c[H] = {0}, n[H] = {0}, m_state[H] = {0};
    float output[T * H] = {0};
    float scratch[4 * H] = {0};
    SlstmParams params = {0.0f};

    slstm_eval_f32(kTest2_input, kTest1_W, kTest1_R, kTest1_b,
                   y, c, n, m_state, output, scratch, B, T, I, H, &params);

    bool ok = true;
    ok &= ExpectNear("y_final", kTest2_expected_y, y, H, kTolerance);
    ok &= ExpectNear("c_final", kTest2_expected_c, c, H, kTolerance);
    ok &= ExpectNear("n_final", kTest2_expected_n, n, H, kTolerance);
    ok &= ExpectNear("m_final", kTest2_expected_m, m_state, H, kTolerance);
    ok &= ExpectNear("output_all", kTest2_expected_output, output, T * H, kTolerance);
    return ok;
}

bool TestOverflowPrevention() {
    const int B = 1, T = 1, I = 2, H = 2;

    float y[H] = {0}, c[H] = {0}, n[H] = {0}, m_state[H] = {0};
    float output[T * H] = {0};
    float scratch[4 * H] = {0};
    SlstmParams params = {0.0f};

    slstm_eval_f32(kTest3_input, kTest3_W, kTest3_R, kTest3_b,
                   y, c, n, m_state, output, scratch, B, T, I, H, &params);

    bool ok = true;
    ok &= ExpectFinite("y", y, H);
    ok &= ExpectFinite("c", c, H);
    ok &= ExpectFinite("n", n, H);
    ok &= ExpectFinite("m", m_state, H);

    ok &= ExpectNear("y", kTest3_expected_y, y, H, kTolerance);
    ok &= ExpectNear("c", kTest3_expected_c, c, H, kTolerance);
    ok &= ExpectNear("n", kTest3_expected_n, n, H, kTolerance);
    ok &= ExpectNear("m", kTest3_expected_m, m_state, H, kTolerance);
    return ok;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::printf("[==========] Running sLSTM kernel tests\n");

    RUN_TEST(TestSingleTimestepZeroState);
    RUN_TEST(TestMultipleTimesteps);
    RUN_TEST(TestOverflowPrevention);

    std::printf("[==========] %d/%d tests passed\n", g_tests_passed, g_tests_run);
    return g_tests_passed == g_tests_run ? 0 : 1;
}
