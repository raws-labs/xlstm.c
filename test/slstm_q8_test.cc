/* sLSTM INT8 kernel unit tests
 *
 * Tests the quantized sLSTM cell against reference values from the
 * f32 kernel. Quantization introduces small errors — tests use relaxed
 * tolerance compared to f32 tests.
 *
 * Build:
 *   make test
 * =========================================================================*/

#include "slstm_q8.h"
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

struct SlstmS8Setup {
    int8_t W_q[4 * 2 * 2];       /* max [4*H, I] = [8, 2] */
    int8_t R_q[4 * 2 * 2];       /* max [4*H, H] = [8, 2] */
    int32_t b_q[4 * 2];          /* max [4*H] = [8] */
    int8_t input_q[3 * 2];       /* max [T, I] = [3, 2] */
    SlstmS8Params params;
};

static void PrepareS8(const float* W, const float* R, const float* b,
                       const float* input, int T,
                       int I, int H,
                       float y_scale, float c_scale, float n_scale,
                       SlstmS8Setup* s) {
    XlstmQuantParam w_qp, r_qp, x_qp, b_qp;

    /* Quantize weights (symmetric) */
    xlstm_quant_symmetric(W, 4 * H * I, &w_qp);
    xlstm_quantize_f32_to_s8(W, s->W_q, 4 * H * I, &w_qp);

    xlstm_quant_symmetric(R, 4 * H * H, &r_qp);
    xlstm_quantize_f32_to_s8(R, s->R_q, 4 * H * H, &r_qp);

    /* Quantize input (asymmetric) */
    xlstm_quant_asymmetric(input, T * I, &x_qp);
    xlstm_quantize_f32_to_s8(input, s->input_q, T * I, &x_qp);

    /* Quantize bias (scale = W_scale * x_scale, zp = 0) */
    b_qp.scale = w_qp.scale * x_qp.scale;
    b_qp.zero_point = 0;
    xlstm_quantize_f32_to_s32(b, s->b_q, 4 * H, &b_qp);

    /* Set params */
    s->params.cell_clip = 0.0f;
    s->params.W_scale = w_qp.scale;
    s->params.R_scale = r_qp.scale;
    s->params.x_quant = x_qp;
    s->params.y_quant.scale = y_scale;
    s->params.y_quant.zero_point = 0;
    s->params.c_quant.scale = c_scale;
    s->params.c_quant.zero_point = 0;
    s->params.n_quant.scale = n_scale;
    s->params.n_quant.zero_point = 0;
}

// ============================================================================
// Test cases
// ============================================================================

bool TestS8SingleTimestep() {
    const int B = 1, T = 1, I = 2, H = 2;

    SlstmS8Setup s;
    PrepareS8(kTest1_W, kTest1_R, kTest1_b, kTest1_input, T, I, H,
              0.01f, 0.01f, 0.01f, &s);

    int8_t y[H] = {0};
    int16_t c[H] = {0};
    int16_t n_state[H] = {0};
    float m_state[H] = {0};
    int8_t output[T * H] = {0};
    int32_t scratch[4 * H] = {0};

    slstm_eval_s8(s.input_q, s.W_q, s.R_q, s.b_q,
                  y, c, n_state, m_state, output, scratch,
                  B, T, I, H, &s.params);

    /* Dequantize output for comparison */
    float y_f[H], output_f[T * H];
    xlstm_dequantize_s8_to_f32(y, y_f, H, &s.params.y_quant);
    xlstm_dequantize_s8_to_f32(output, output_f, T * H, &s.params.y_quant);

    bool ok = true;
    ok &= ExpectNear("y", kTest1_expected_y, y_f, H, 0.05f);
    ok &= ExpectNear("output", kTest1_expected_y, output_f, H, 0.05f);
    ok &= ExpectFinite("m", m_state, H);
    return ok;
}

bool TestS8MultipleTimesteps() {
    const int B = 1, T = 3, I = 2, H = 2;

    SlstmS8Setup s;
    PrepareS8(kTest1_W, kTest1_R, kTest1_b, kTest2_input, T, I, H,
              0.01f, 0.01f, 0.01f, &s);

    int8_t y[H] = {0};
    int16_t c[H] = {0};
    int16_t n_state[H] = {0};
    float m_state[H] = {0};
    int8_t output[T * H] = {0};
    int32_t scratch[4 * H] = {0};

    slstm_eval_s8(s.input_q, s.W_q, s.R_q, s.b_q,
                  y, c, n_state, m_state, output, scratch,
                  B, T, I, H, &s.params);

    /* Dequantize for comparison */
    float y_f[H], output_f[T * H];
    xlstm_dequantize_s8_to_f32(y, y_f, H, &s.params.y_quant);
    xlstm_dequantize_s8_to_f32(output, output_f, T * H, &s.params.y_quant);

    bool ok = true;
    ok &= ExpectNear("y_final", kTest2_expected_y, y_f, H, 0.10f);
    ok &= ExpectNear("output_all", kTest2_expected_output, output_f, T * H, 0.10f);
    ok &= ExpectFinite("m", m_state, H);
    return ok;
}

bool TestS8OverflowPrevention() {
    const int B = 1, T = 1, I = 2, H = 2;

    SlstmS8Setup s;
    PrepareS8(kTest3_W, kTest3_R, kTest3_b, kTest3_input, T, I, H,
              0.01f, 0.01f, 0.01f, &s);

    int8_t y[H] = {0};
    int16_t c[H] = {0};
    int16_t n_state[H] = {0};
    float m_state[H] = {0};
    int8_t output[T * H] = {0};
    int32_t scratch[4 * H] = {0};

    slstm_eval_s8(s.input_q, s.W_q, s.R_q, s.b_q,
                  y, c, n_state, m_state, output, scratch,
                  B, T, I, H, &s.params);

    /* Dequantize */
    float y_f[H];
    xlstm_dequantize_s8_to_f32(y, y_f, H, &s.params.y_quant);

    bool ok = true;
    ok &= ExpectFinite("y", y_f, H);
    ok &= ExpectFinite("m", m_state, H);
    ok &= ExpectNear("y", kTest3_expected_y, y_f, H, 0.10f);
    ok &= ExpectNear("m", kTest3_expected_m, m_state, H, 0.10f);
    return ok;
}

bool TestS8QuantizationBound() {
    /* Run q8 kernel on all 3 test cases and verify the max absolute error
     * vs f32 golden values stays within a reasonable bound. */
    const int I = 2, H = 2;
    float max_err = 0.0f;

    /* Test 1: single timestep */
    {
        const int T = 1;
        SlstmS8Setup s;
        PrepareS8(kTest1_W, kTest1_R, kTest1_b, kTest1_input, T, I, H,
                  0.01f, 0.01f, 0.01f, &s);

        int8_t y[H] = {0};
        int16_t c[H] = {0};
        int16_t n_state[H] = {0};
        float m_state[H] = {0};
        int8_t output[1 * H] = {0};
        int32_t scratch[4 * H] = {0};

        slstm_eval_s8(s.input_q, s.W_q, s.R_q, s.b_q,
                      y, c, n_state, m_state, output, scratch,
                      1, T, I, H, &s.params);

        float y_f[H];
        xlstm_dequantize_s8_to_f32(y, y_f, H, &s.params.y_quant);
        for (int i = 0; i < H; ++i) {
            float err = std::abs(y_f[i] - kTest1_expected_y[i]);
            if (err > max_err) max_err = err;
        }
    }

    /* Test 2: multi-step */
    {
        const int T = 3;
        SlstmS8Setup s;
        PrepareS8(kTest1_W, kTest1_R, kTest1_b, kTest2_input, T, I, H,
                  0.01f, 0.01f, 0.01f, &s);

        int8_t y[H] = {0};
        int16_t c[H] = {0};
        int16_t n_state[H] = {0};
        float m_state[H] = {0};
        int8_t output[3 * H] = {0};
        int32_t scratch[4 * H] = {0};

        slstm_eval_s8(s.input_q, s.W_q, s.R_q, s.b_q,
                      y, c, n_state, m_state, output, scratch,
                      1, T, I, H, &s.params);

        float y_f[H];
        xlstm_dequantize_s8_to_f32(y, y_f, H, &s.params.y_quant);
        for (int i = 0; i < H; ++i) {
            float err = std::abs(y_f[i] - kTest2_expected_y[i]);
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
    std::printf("[==========] Running sLSTM INT8 kernel tests\n");

    RUN_TEST(TestS8SingleTimestep);
    RUN_TEST(TestS8MultipleTimesteps);
    RUN_TEST(TestS8OverflowPrevention);
    RUN_TEST(TestS8QuantizationBound);

    std::printf("[==========] %d/%d tests passed\n", g_tests_passed, g_tests_run);
    return g_tests_passed == g_tests_run ? 0 : 1;
}
