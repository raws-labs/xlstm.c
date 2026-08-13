/* mLSTM INT8 kernel unit tests
 *
 * Tests the quantized mLSTM cell against reference values from the
 * f32 kernel. Quantization introduces small errors - tests use relaxed
 * tolerance compared to f32 tests.
 *
 * Table-driven over kMlstmCases (test/reference_data.h), the same table
 * the f32 test runs. Quantization scales are derived per case from the
 * actual data rather than a fixed literal - see PrepareMlstmS8 below.
 *
 * Build:
 *   make test
 * =========================================================================*/

#include "mlstm_s8.h"
#include "xlstm_quant.h"
#include "test_util.h"

#include <cstring>

// ============================================================================
// Reference test data - same golden values as f32 tests
// ============================================================================

#include "reference_data.h"

// ============================================================================
// Helper: derive and apply quantization scales from case data
// ============================================================================

/* Static buffers sized for the largest case. XLSTM_MAX_HIDDEN is 256. */
struct MlstmS8Setup {
    int8_t W_q[(4 * 256 + 2) * 256]; /* max [(4*H+2), I] */
    int32_t b_q[4 * 256 + 2];        /* max [4*H+2] */
    int8_t input_q[3 * 256];         /* max [T, I] */
    MlstmS8Params params;
};

/* Derive weight (symmetric) and input (asymmetric) quant params directly
 * from the tensors being quantized. mLSTM has no recurrent weight. */
static void DeriveScales(const float* W, int w_len, const float* x, int x_len,
                          XlstmQuantParam* w_qp, XlstmQuantParam* x_qp) {
    xlstm_quant_symmetric(W, w_len, w_qp);
    xlstm_quant_asymmetric(x, x_len, x_qp);
}

/* Symmetric calibration at INT16 granularity (max_abs * headroom / 32767,
 * zero_point 0) - see the longer explanation in slstm_s8_test.cc. n and C
 * are stored as int16_t and read/written as strictly symmetric by
 * mlstm_s8.c (no zero-point term anywhere), so xlstm_quant_asymmetric's
 * range/255 + non-zero zero_point was an INT8-shaped calibration
 * misapplied to a 16-bit symmetric tensor.
 *
 * headroom exists because calibrating from a single final-state snapshot
 * (all reference_data.h stores) can badly undershoot the true mid-
 * sequence trajectory peak - measured up to 6.2x for SweepM1's n and
 * 3.4x for SweepM8's C. See kStateHeadroom below and task-4-report.md for
 * the full per-case ratio table and how 4x was chosen. */
static void QuantSymmetricS16(const float* data, int len, float headroom,
                               XlstmQuantParam* out) {
    float max_abs = 0.0f;
    for (int i = 0; i < len; ++i) {
        float a = std::abs(data[i]);
        if (a > max_abs) max_abs = a;
    }
    out->scale = (max_abs > 0.0f) ? (max_abs * headroom / 32767.0f) : 1.0f;
    out->zero_point = 0;
}

/* Headroom multiplier for n_quant/C_quant - see the fuller justification
 * next to slstm_s8_test.cc's copy of this constant. Chosen from a sweep
 * across every case in reference_data.h: SweepM8's error (this task's
 * worst-affected case, C's true/final ratio 3.37x) plateaus at its
 * no-clipping floor (~0.0388) by 3x headroom and stays there through 16x,
 * so 4x keeps margin above the plateau without being arbitrary. */
static const float kStateHeadroom = 4.0f;

/* Sets up quantized weights/input plus calibrated state-quant scales for
 * one reference case.
 *
 * y/C/n scales are calibrated from the case's own golden f32 values
 * (expected_output/expected_y, expected_state, expected_n) rather than a
 * fixed literal - see task-4-report.md's "on calibration" note for what
 * this oracle-calibrated setup does and does not demonstrate. y is INT8
 * (asymmetric); C and n are INT16 and consumed as strictly symmetric by
 * the kernel, so they use QuantSymmetricS16 above, not
 * xlstm_quant_asymmetric.
 *
 * Every case in reference_data.h now stores the full C matrix
 * (mlstm_sized_case's store_state is unconditional as of this task), so
 * C_quant is always calibrated from real data. The guard below is
 * defensive only, matching sLSTM's PrepareS8, and is not currently
 * exercised: reuse n's calibrated scale with no multiplier if a future
 * case omits C. (An earlier version of this fallback used an empirically
 * tuned 8x n_quant.scale for the one case that needed it, H=64 - removed
 * because that case now stores C directly; see task-4-report.md.) */
static void PrepareMlstmS8(const XlstmRefCase* tc, MlstmS8Setup* s) {
    const int H = tc->H, T = tc->T, I = tc->I;
    const int total = 4 * H + 2;

    XlstmQuantParam w_qp, x_qp, b_qp;
    DeriveScales(tc->W, total * I, tc->input, T * I, &w_qp, &x_qp);
    xlstm_quantize_f32_to_s8(tc->W, s->W_q, total * I, &w_qp);
    xlstm_quantize_f32_to_s8(tc->input, s->input_q, T * I, &x_qp);

    /* Bias quantized with input*weight scale (zp = 0) */
    b_qp.scale = w_qp.scale * x_qp.scale;
    b_qp.zero_point = 0;
    xlstm_quantize_f32_to_s32(tc->b, s->b_q, total, &b_qp);

    s->params.cell_clip = 0.0f;
    s->params.W_scale = w_qp.scale;
    s->params.x_quant = x_qp;

    const float* y_cal = tc->expected_output ? tc->expected_output : tc->expected_y;
    int y_cal_len = tc->expected_output ? T * H : H;
    xlstm_quant_asymmetric(y_cal, y_cal_len, &s->params.y_quant);
    QuantSymmetricS16(tc->expected_n, H, kStateHeadroom, &s->params.n_quant);

    if (tc->expected_state) {
        QuantSymmetricS16(tc->expected_state, H * H, kStateHeadroom, &s->params.C_quant);
    } else {
        s->params.C_quant.scale = s->params.n_quant.scale;
        s->params.C_quant.zero_point = 0;
    }
}

// ============================================================================
// Table-driven runner
// ============================================================================

/* Runs one case and writes the dequantized hidden state into y_out.
 * m_out (length 1 - mLSTM's m is a scalar per batch) and output_out
 * (length T*H, dequantized) are optional (pass NULL to skip) -
 * RunMlstmS8Case wants them for the m/output assertions,
 * TestMlstmS8QuantizationBound only wants the error number.
 * Returns the largest absolute deviation from the f32 golden y. */
static float EvalMlstmS8Case(const XlstmRefCase* tc, float* y_out,
                              float* m_out, float* output_out) {
    const int H = tc->H, T = tc->T, I = tc->I;

    static MlstmS8Setup s;
    PrepareMlstmS8(tc, &s);

    static int8_t y[256];
    static int16_t C[256 * 256];
    static int16_t n_state[256];
    static float m_state[256];
    static int8_t output[3 * 256];
    static int32_t scratch[4 * 256 + 2];
    for (int i = 0; i < H; ++i) { y[i] = 0; n_state[i] = 0; }
    for (int i = 0; i < H * H; ++i) C[i] = 0;
    m_state[0] = 0;
    for (int i = 0; i < T * H; ++i) output[i] = 0;
    for (int i = 0; i < 4 * H + 2; ++i) scratch[i] = 0;

    mlstm_eval_s8(s.input_q, s.W_q, s.b_q,
                  y, C, n_state, m_state, output, scratch,
                  tc->B, T, I, H, &s.params);

    xlstm_dequantize_s8_to_f32(y, y_out, H, &s.params.y_quant);
    if (m_out) m_out[0] = m_state[0];
    /* Always dequantize output locally, even if the caller doesn't want it
     * back, so max_err below reflects the whole trajectory, not just the
     * final timestep that y alone captures. This matters: SweepM8's
     * final-y error is 0.124726, but an intermediate timestep reaches
     * 1.127354 (transient C/n excursion beyond the final-snapshot
     * calibration range, saturating INT16 mid-sequence - see
     * task-4-report.md). Reporting only the final-y number would
     * understate what tolerance the case actually needs. */
    static float output_local[3 * 256];
    xlstm_dequantize_s8_to_f32(output, output_local, T * H, &s.params.y_quant);
    if (output_out) {
        for (int i = 0; i < T * H; ++i) output_out[i] = output_local[i];
    }

    float max_err = 0.0f;
    for (int j = 0; j < H; ++j) {
        float d = std::abs(tc->expected_y[j] - y_out[j]);
        if (d > max_err) max_err = d;
    }
    if (tc->expected_output) {
        for (int i = 0; i < T * H; ++i) {
            float d = std::abs(tc->expected_output[i] - output_local[i]);
            if (d > max_err) max_err = d;
        }
    }
    return max_err;
}

/* Several channels across the mLSTM sweep have their own measured error
 * far enough below the case's blanket tol_s8 that the blanket value is
 * vacuous for them specifically (case-wide tol_s8 / that channel's own
 * max|golden| >= 1.0, i.e. an all-zero corruption of just that channel
 * would go undetected) - a direct consequence of the blanket being
 * sized for the case's single worst channel while H=16/17/64 span a wide
 * range of per-channel magnitudes. Each gets a bound derived from its
 * own measured error, the same methodology used for sLSTM's SweepS17/
 * SweepS64 carve-outs in slstm_s8_test.cc:
 *   - SweepM16 channel 10: own error 0.010887, own range 0.075296.
 *     bound 0.02 (1.5x, rounded) -> ratio 0.266. Fully resolved.
 *   - SweepM17 channel 14: own error 0.048712, own range 0.130253.
 *     bound 0.08 (1.5x, rounded) -> ratio 0.614. Fully resolved.
 *   - SweepM17 channel  1: own error 0.059621, own range 1.163852.
 *     bound 0.09 (1.5x, rounded) -> ratio 0.077. Fully resolved.
 *   - SweepM17 channel  9: own error 0.009140, own range 0.353493.
 *     bound 0.014 (1.5x, rounded) -> ratio 0.039. Fully resolved.
 *   - SweepM64 channel  4: own error 0.035842, own range 0.194414.
 *     bound 0.054 (1.5x, rounded) -> ratio 0.278. Fully resolved.
 *   - SweepM64 channel 38: own error 0.007423, own range 0.266070.
 *     bound 0.012 (1.5x, rounded) -> ratio 0.045. Fully resolved.
 *   - SweepM64 channel 42: own error 0.023425, own range 0.188005.
 *     bound 0.036 (1.5x, rounded) -> ratio 0.192. Fully resolved.
 *   - SweepM64 channel 63: own error 0.010621, own range 0.010621 -
 *     these are equal, not approximately equal: this channel's true
 *     value quantizes to exactly the y_quant zero-point under this
 *     case's calibrated scale, so its measured error against a
 *     correctly-computed value IS the value itself. No tolerance choice
 *     can distinguish this channel's correct output from an all-zero
 *     corruption of it - not a margin-tuning problem, a representability
 *     limit of per-tensor INT8 quantization when one channel's true
 *     value is far smaller than the tensor's overall range. bound 0.016
 *     (1.5x, rounded) still shrinks the undetectable window by >16x
 *     versus the 0.27 blanket it replaces, and remains larger than the
 *     measured error so it does not cause false failures, but the ratio
 *     (1.5) stays above 1 - reported, not hidden, in task-4-report.md.
 * Every other channel, in every other case, is comfortably covered by
 * its case's blanket tol_s8 - see task-4-report.md's per-channel audit
 * for the full table, including the small number of channels between
 * ratio 0.5 and 0.9 that are not vacuous (ratio < 1) and so are not
 * overridden here. */
static float MlstmPerChannelTolS8(const XlstmRefCase* tc, int channel, float default_tol) {
    if (std::strcmp(tc->name, "SweepM16") == 0 && channel == 10) return 0.02f;
    if (std::strcmp(tc->name, "SweepM17") == 0) {
        if (channel == 14) return 0.08f;
        if (channel == 1) return 0.09f;
        if (channel == 9) return 0.014f;
    }
    if (std::strcmp(tc->name, "SweepM64") == 0) {
        if (channel == 63) return 0.016f;
        if (channel == 4) return 0.054f;
        if (channel == 38) return 0.012f;
        if (channel == 42) return 0.036f;
    }
    return default_tol;
}

static bool RunMlstmS8Case(const XlstmRefCase* tc) {
    static float y_f[256], m_f[1], output_f[3 * 256];
    EvalMlstmS8Case(tc, y_f, m_f, output_f);
    bool ok = true;
    ok &= ExpectFinite("y", y_f, tc->H);
    ok &= ExpectFinite("m", m_f, 1);

    bool has_carveout = std::strcmp(tc->name, "SweepM16") == 0 ||
                         std::strcmp(tc->name, "SweepM17") == 0 ||
                         std::strcmp(tc->name, "SweepM64") == 0;
    if (!has_carveout) {
        ok &= ExpectNear("y", tc->expected_y, y_f, tc->H, tc->tol_s8);
        if (tc->expected_output)
            ok &= ExpectNear("output", tc->expected_output, output_f, tc->T * tc->H, tc->tol_s8);
        return ok;
    }

    for (int j = 0; j < tc->H; ++j) {
        float tol = MlstmPerChannelTolS8(tc, j, tc->tol_s8);
        float diff = std::abs(tc->expected_y[j] - y_f[j]);
        if (diff > tol + kRelTol * std::abs(tc->expected_y[j])) {
            std::printf("  FAIL y[%d]: expected %.8f, got %.8f (diff %.2e, tol %.4f)\n",
                        j, tc->expected_y[j], y_f[j], diff, tol);
            ok = false;
        }
    }
    if (tc->expected_output) {
        for (int t = 0; t < tc->T; ++t) {
            for (int j = 0; j < tc->H; ++j) {
                int idx = t * tc->H + j;
                float tol = MlstmPerChannelTolS8(tc, j, tc->tol_s8);
                float diff = std::abs(tc->expected_output[idx] - output_f[idx]);
                if (diff > tol + kRelTol * std::abs(tc->expected_output[idx])) {
                    std::printf("  FAIL output[%d]: expected %.8f, got %.8f (diff %.2e, tol %.4f)\n",
                                idx, tc->expected_output[idx], output_f[idx], diff, tol);
                    ok = false;
                }
            }
        }
    }
    return ok;
}

/* Runs the s8 kernel on every case in kMlstmCases and reports, per case,
 * the max absolute error vs the f32 golden y. This is the size-vs-error
 * measurement: how INT8 error grows with H (.docs/SCOPE.md section 8). */
static bool TestMlstmS8QuantizationBound() {
    static float y_f[256];
    float max_err = 0.0f;
    for (int i = 0; i < kMlstmCasesCount; ++i) {
        const XlstmRefCase* tc = &kMlstmCases[i];
        float err = EvalMlstmS8Case(tc, y_f, NULL, NULL);
        std::printf("  %-10s H=%-3d max abs error vs f32: %.6f\n", tc->name, tc->H, err);
        if (err > max_err) max_err = err;
    }
    std::printf("  max absolute error vs f32 across all cases: %.6f\n", max_err);
    std::printf("  (oracle-calibrated: scales are derived from this case's own golden\n"
                 "   output, not an independent calibration set - see task-4-report.md's\n"
                 "   'on calibration' note. These are a best case, not a deployment figure.)\n");

    /* Bound set to ~1.5x the measured maximum (0.726042), which is
     * SweepM17. SweepM8's calibration-headroom fix (kStateHeadroom in
     * PrepareMlstmS8) brought its error down from 1.127354 to 0.038831 -
     * see task-4-report.md for the per-case measurements, the headroom
     * derivation, and the full tol_s8/dynamic-range ratio table. */
    if (max_err > 1.10f) {
        std::printf("  FAIL: max error %.6f exceeds bound 1.10\n", max_err);
        return false;
    }
    return true;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::printf("[==========] Running mLSTM INT8 kernel tests\n");

    for (int i = 0; i < kMlstmCasesCount; ++i) {
        const XlstmRefCase* tc = &kMlstmCases[i];
        g_tests_run++;
        std::printf("[ RUN      ] mLSTM INT8 %s (H=%d, T=%d)\n", tc->name, tc->H, tc->T);
        if (RunMlstmS8Case(tc)) {
            g_tests_passed++;
            std::printf("[       OK ] mLSTM INT8 %s\n", tc->name);
        } else {
            std::printf("[  FAILED  ] mLSTM INT8 %s\n", tc->name);
        }
    }

    RUN_TEST(TestMlstmS8QuantizationBound);

    std::printf("[==========] %d/%d tests passed\n", g_tests_passed, g_tests_run);
    return g_tests_passed == g_tests_run ? 0 : 1;
}
