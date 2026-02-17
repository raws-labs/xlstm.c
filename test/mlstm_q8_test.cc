/* mLSTM INT8 kernel unit tests
 *
 * Tests the quantized mLSTM cell against reference values from the
 * f32 kernel. Quantization introduces small errors — tests use relaxed
 * tolerance compared to f32 tests.
 *
 * Build:
 *   make test
 * =========================================================================*/

#include "mlstm_q8.h"
#include "xlstm_quant.h"
#include "test_util.h"

#include <cstring>

// ============================================================================
// Reference test data — same golden values as f32 tests
// ============================================================================

#include "reference_data.h"

// ============================================================================
// Helper: set up quantized parameters from float test data
// ============================================================================

struct MlstmS8Setup {
    int8_t W_q[(4 * 2 + 2) * 3]; /* max [(4*H+2), I] = [10, 3] */
    int32_t b_q[4 * 2 + 2];      /* max [4*H+2] = [10] */
    int8_t input_q[3 * 3];       /* max [T, I] = [3, 3] */
    MlstmS8Params params;
};

static void PrepareMlstmS8(const float* W, const float* b,
                            const float* input, int T,
                            int I, int H,
                            float y_scale, float C_scale, float n_scale,
                            MlstmS8Setup* s) {
    int total = 4 * H + 2;
    XlstmQuantParam w_qp, x_qp, b_qp;

    /* Quantize weights (symmetric) */
    xlstm_quant_symmetric(W, total * I, &w_qp);
    xlstm_quantize_f32_to_s8(W, s->W_q, total * I, &w_qp);

    /* Quantize input (asymmetric) */
    xlstm_quant_asymmetric(input, T * I, &x_qp);
    xlstm_quantize_f32_to_s8(input, s->input_q, T * I, &x_qp);

    /* Quantize bias */
    b_qp.scale = w_qp.scale * x_qp.scale;
    b_qp.zero_point = 0;
    xlstm_quantize_f32_to_s32(b, s->b_q, total, &b_qp);

    /* Set params */
    s->params.cell_clip = 0.0f;
    s->params.W_scale = w_qp.scale;
    s->params.x_quant = x_qp;
    s->params.y_quant.scale = y_scale;
    s->params.y_quant.zero_point = 0;
    s->params.C_quant.scale = C_scale;
    s->params.C_quant.zero_point = 0;
    s->params.n_quant.scale = n_scale;
    s->params.n_quant.zero_point = 0;
}

// ============================================================================
// Test cases
// ============================================================================

bool TestMlstmS8SingleTimestep() {
    const int B = 1, T = 1, I = 3, H = 2;

    MlstmS8Setup s;
    PrepareMlstmS8(kMTest1_W, kMTest1_b, kMTest1_input, T, I, H,
                   0.01f, 0.01f, 0.01f, &s);

    int8_t y[H] = {0};
    int16_t C[H * H] = {0};
    int16_t n_state[H] = {0};
    float m_state[1] = {0};
    int8_t output[T * H] = {0};
    int32_t scratch[4 * H + 2] = {0};

    mlstm_eval_s8(s.input_q, s.W_q, s.b_q,
                  y, C, n_state, m_state, output, scratch,
                  B, T, I, H, &s.params);

    /* Dequantize output for comparison */
    float y_f[H], output_f[T * H];
    xlstm_dequantize_s8_to_f32(y, y_f, H, &s.params.y_quant);
    xlstm_dequantize_s8_to_f32(output, output_f, T * H, &s.params.y_quant);

    /* mLSTM Test1 output values are very small (~0.001), so they'll quantize
     * to 0 with scale=0.01. Use absolute tolerance that accepts this. */
    bool ok = true;
    ok &= ExpectNear("y", kMTest1_expected_y, y_f, H, 0.05f);
    ok &= ExpectNear("output", kMTest1_expected_y, output_f, H, 0.05f);
    ok &= ExpectFinite("m", m_state, 1);
    return ok;
}

bool TestMlstmS8MultipleTimesteps() {
    const int B = 1, T = 3, I = 3, H = 2;

    MlstmS8Setup s;
    PrepareMlstmS8(kMTest1_W, kMTest1_b, kMTest2_input, T, I, H,
                   0.01f, 0.01f, 0.01f, &s);

    int8_t y[H] = {0};
    int16_t C[H * H] = {0};
    int16_t n_state[H] = {0};
    float m_state[1] = {0};
    int8_t output[T * H] = {0};
    int32_t scratch[4 * H + 2] = {0};

    mlstm_eval_s8(s.input_q, s.W_q, s.b_q,
                  y, C, n_state, m_state, output, scratch,
                  B, T, I, H, &s.params);

    /* Dequantize for comparison */
    float y_f[H], output_f[T * H];
    xlstm_dequantize_s8_to_f32(y, y_f, H, &s.params.y_quant);
    xlstm_dequantize_s8_to_f32(output, output_f, T * H, &s.params.y_quant);

    bool ok = true;
    ok &= ExpectNear("y_final", kMTest2_expected_y, y_f, H, 0.10f);
    ok &= ExpectNear("output_all", kMTest2_expected_output, output_f, T * H, 0.10f);
    ok &= ExpectFinite("m", m_state, 1);
    return ok;
}

bool TestMlstmS8OverflowPrevention() {
    const int B = 1, T = 1, I = 3, H = 2;

    /* Large values require wider quant scales */
    MlstmS8Setup s;
    PrepareMlstmS8(kMTest3_W, kMTest3_b, kMTest3_input, T, I, H,
                   0.15f, 0.01f, 0.01f, &s);

    int8_t y[H] = {0};
    int16_t C[H * H] = {0};
    int16_t n_state[H] = {0};
    float m_state[1] = {0};
    int8_t output[T * H] = {0};
    int32_t scratch[4 * H + 2] = {0};

    mlstm_eval_s8(s.input_q, s.W_q, s.b_q,
                  y, C, n_state, m_state, output, scratch,
                  B, T, I, H, &s.params);

    /* Dequantize */
    float y_f[H];
    xlstm_dequantize_s8_to_f32(y, y_f, H, &s.params.y_quant);

    bool ok = true;
    ok &= ExpectFinite("y", y_f, H);
    ok &= ExpectFinite("m", m_state, 1);
    ok &= ExpectNear("y", kMTest3_expected_y, y_f, H, 1.0f);
    ok &= ExpectNear("m", kMTest3_expected_m, m_state, 1, 1.0f);
    return ok;
}

bool TestMlstmS8QuantizationBound() {
    /* Run q8 kernel on test cases 1 and 2, verify max error is bounded. */
    const int I = 3, H = 2;
    float max_err = 0.0f;

    /* Test 1 */
    {
        const int T = 1;
        MlstmS8Setup s;
        PrepareMlstmS8(kMTest1_W, kMTest1_b, kMTest1_input, T, I, H,
                       0.01f, 0.01f, 0.01f, &s);

        int8_t y[H] = {0};
        int16_t C[H * H] = {0};
        int16_t n_state[H] = {0};
        float m_state[1] = {0};
        int8_t output[1 * H] = {0};
        int32_t scratch[4 * H + 2] = {0};

        mlstm_eval_s8(s.input_q, s.W_q, s.b_q,
                      y, C, n_state, m_state, output, scratch,
                      1, T, I, H, &s.params);

        float y_f[H];
        xlstm_dequantize_s8_to_f32(y, y_f, H, &s.params.y_quant);
        for (int i = 0; i < H; ++i) {
            float err = std::abs(y_f[i] - kMTest1_expected_y[i]);
            if (err > max_err) max_err = err;
        }
    }

    /* Test 2 */
    {
        const int T = 3;
        MlstmS8Setup s;
        PrepareMlstmS8(kMTest1_W, kMTest1_b, kMTest2_input, T, I, H,
                       0.01f, 0.01f, 0.01f, &s);

        int8_t y[H] = {0};
        int16_t C[H * H] = {0};
        int16_t n_state[H] = {0};
        float m_state[1] = {0};
        int8_t output[3 * H] = {0};
        int32_t scratch[4 * H + 2] = {0};

        mlstm_eval_s8(s.input_q, s.W_q, s.b_q,
                      y, C, n_state, m_state, output, scratch,
                      1, T, I, H, &s.params);

        float y_f[H];
        xlstm_dequantize_s8_to_f32(y, y_f, H, &s.params.y_quant);
        for (int i = 0; i < H; ++i) {
            float err = std::abs(y_f[i] - kMTest2_expected_y[i]);
            if (err > max_err) max_err = err;
        }
    }

    std::printf("  max absolute error vs f32: %.6f\n", max_err);

    if (max_err > 0.15f) {
        std::printf("  FAIL: max error %.6f exceeds bound 0.15\n", max_err);
        return false;
    }
    return true;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::printf("[==========] Running mLSTM INT8 kernel tests\n");

    RUN_TEST(TestMlstmS8SingleTimestep);
    RUN_TEST(TestMlstmS8MultipleTimesteps);
    RUN_TEST(TestMlstmS8OverflowPrevention);
    RUN_TEST(TestMlstmS8QuantizationBound);

    std::printf("[==========] %d/%d tests passed\n", g_tests_passed, g_tests_run);
    return g_tests_passed == g_tests_run ? 0 : 1;
}
