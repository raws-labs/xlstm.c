/* sLSTM INT8 kernel unit tests
 *
 * Tests the quantized sLSTM cell against reference values from the
 * f32 kernel. Quantization introduces small errors - tests use relaxed
 * tolerance compared to f32 tests.
 *
 * Table-driven over kSlstmCases (test/reference_data.h), the same table
 * the f32 test runs. Quantization scales are derived per case from the
 * actual data rather than a fixed literal - see PrepareS8 below.
 *
 * Build:
 *   make test
 * =========================================================================*/

#include "slstm_s8.h"
#include "xlstm_quant.h"
#include "test_util.h"

// ============================================================================
// Reference test data - same golden values as f32 tests
// ============================================================================

#include "reference_data.h"

// ============================================================================
// Helper: derive and apply quantization scales from case data
// ============================================================================

/* Static buffers sized for the largest case. XLSTM_MAX_HIDDEN is 256. */
struct SlstmS8Setup {
    int8_t W_q[4 * 256 * 256];   /* max [4*H, I] */
    int8_t R_q[4 * 256 * 256];   /* max [4*H, H] */
    int32_t b_q[4 * 256];        /* max [4*H] */
    int8_t input_q[3 * 256];     /* max [T, I] */
    SlstmS8Params params;
};

/* Derive weight (symmetric) and input (asymmetric) quant params directly
 * from the tensors being quantized. R is optional so the same helper can
 * be reused where there is no recurrent weight (mLSTM has its own copy,
 * since it has no R at all). */
static void DeriveScales(const float* W, int w_len,
                          const float* R, int r_len,
                          const float* x, int x_len,
                          XlstmQuantParam* w_qp,
                          XlstmQuantParam* r_qp,
                          XlstmQuantParam* x_qp) {
    xlstm_quant_symmetric(W, w_len, w_qp);
    if (R && r_len > 0) xlstm_quant_symmetric(R, r_len, r_qp);
    xlstm_quant_asymmetric(x, x_len, x_qp);
}

/* Symmetric calibration at INT16 granularity (max_abs * headroom / 32767,
 * zero_point 0). c/n/C are stored as int16_t and the kernels read/write
 * them as strictly symmetric (no zero-point term anywhere in
 * slstm_s8.c/mlstm_s8.c) - xlstm_quant_asymmetric's scale = range/255 and
 * non-zero zero_point are an INT8-shaped calibration silently misapplied
 * to a 16-bit, symmetric tensor. That throws away about 7 bits of the
 * available range and was inflating every INT8-vs-f32 error this task
 * measures. xlstm_quant.c's existing functions are untouched; this is a
 * local, test-only helper.
 *
 * headroom exists because calibrating from a single final-state snapshot
 * (all reference_data.h stores) can badly undershoot the true mid-
 * sequence trajectory peak: measured up to 6.2x for mLSTM SweepM1's n and
 * 3.4x for SweepM8's C (see kStateHeadroom below and task-4-report.md for
 * the full per-case ratio table). Pass 1.0f for no headroom (weights/
 * inputs use plain xlstm_quant_symmetric/asymmetric instead and never
 * call this with headroom != kStateHeadroom). */
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

/* Headroom multiplier for c_quant/n_quant (sLSTM) and n_quant/C_quant
 * (mLSTM). Chosen from measurement, not trial: swept 1x-16x across every
 * case in reference_data.h and picked the point where the worst affected
 * case (mLSTM SweepM8, true/final ratio 3.37x for C) plateaus at its
 * "no-clipping" floor (~0.0388, matching an ablation that replaces C/n
 * with their exact float values) - that happens by 3x and holds through
 * 16x, so 4x keeps a margin above the plateau point without being
 * arbitrarily large. sLSTM is completely insensitive to this value (its
 * dominant error source is INT8 W/x matmul noise, not state clipping),
 * so headroom only matters for mLSTM here, but is applied uniformly to
 * keep one number to justify rather than two. See task-4-report.md for
 * the sweep data. */
static const float kStateHeadroom = 4.0f;

/* Sets up quantized weights/input plus calibrated state-quant scales for
 * one reference case.
 *
 * y/c/n scales are calibrated from the case's own golden f32 values
 * (expected_output/expected_y, expected_state, expected_n) rather than a
 * fixed literal - see task-4-report.md's "on calibration" note for what
 * this oracle-calibrated setup does and does not demonstrate. A fixed
 * 0.01 scale independent of H saturates the INT8 y range once H grows
 * past a couple dozen. y is INT8 (asymmetric, 255 levels); c and n are
 * INT16 and consumed as strictly symmetric by the kernel, so they are
 * calibrated with QuantSymmetricS16 above, not xlstm_quant_asymmetric. */
static void PrepareS8(const XlstmRefCase* tc, SlstmS8Setup* s) {
    const int H = tc->H, T = tc->T, I = tc->I;

    XlstmQuantParam w_qp, r_qp, x_qp, b_qp;
    DeriveScales(tc->W, 4 * H * I, tc->R, 4 * H * H, tc->input, T * I,
                 &w_qp, &r_qp, &x_qp);
    xlstm_quantize_f32_to_s8(tc->W, s->W_q, 4 * H * I, &w_qp);
    xlstm_quantize_f32_to_s8(tc->R, s->R_q, 4 * H * H, &r_qp);
    xlstm_quantize_f32_to_s8(tc->input, s->input_q, T * I, &x_qp);

    /* Bias quantized with input*weight scale (zp = 0) */
    b_qp.scale = w_qp.scale * x_qp.scale;
    b_qp.zero_point = 0;
    xlstm_quantize_f32_to_s32(tc->b, s->b_q, 4 * H, &b_qp);

    s->params.cell_clip = 0.0f;
    s->params.W_scale = w_qp.scale;
    s->params.R_scale = r_qp.scale;
    s->params.x_quant = x_qp;

    /* Calibrate y over the whole output sequence when it was stored;
     * T=1 cases without a stored output fall back to expected_y, which
     * equals output at T=1 anyway (same convention the f32 runner uses). */
    const float* y_cal = tc->expected_output ? tc->expected_output : tc->expected_y;
    int y_cal_len = tc->expected_output ? T * H : H;
    xlstm_quant_asymmetric(y_cal, y_cal_len, &s->params.y_quant);

    QuantSymmetricS16(tc->expected_n, H, kStateHeadroom, &s->params.n_quant);

    /* expected_state (c) is normally present, but store_state is a
     * generic per-case emitter flag, not an sLSTM-specific guarantee -
     * guard it the same way mLSTM's C_quant already does, rather than
     * assume it is always non-NULL. Fallback: c and n share the same
     * per-step recurrence shape (f_gate*prev + i_gate*bounded_term, with
     * the bounded term in c's case tanh(z) in [-1,1] and in n's case
     * i_gate itself in [0,1]) so they are the same order of magnitude;
     * reuse n's already-calibrated scale with no extra multiplier. This
     * path is not currently exercised by any case in reference_data.h. */
    if (tc->expected_state) {
        QuantSymmetricS16(tc->expected_state, H, kStateHeadroom, &s->params.c_quant);
    } else {
        s->params.c_quant.scale = s->params.n_quant.scale;
        s->params.c_quant.zero_point = 0;
    }
}

// ============================================================================
// Table-driven runner
// ============================================================================

/* Runs one case and writes the dequantized hidden state into y_out.
 * m_out (length H) and output_out (length T*H, dequantized) are optional
 * (pass NULL to skip) - RunSlstmS8Case wants them for the m/output
 * assertions, TestS8QuantizationBound only wants the error number.
 * Returns the largest absolute deviation from the f32 golden y. */
static float EvalSlstmS8Case(const XlstmRefCase* tc, float* y_out,
                              float* m_out, float* output_out) {
    const int H = tc->H, T = tc->T, I = tc->I;

    static SlstmS8Setup s;
    PrepareS8(tc, &s);

    static int8_t y[256];
    static int16_t c[256], n_state[256];
    static float m_state[256];
    static int8_t output[3 * 256];
    static int32_t scratch[4 * 256];
    for (int i = 0; i < H; ++i) { y[i] = 0; c[i] = 0; n_state[i] = 0; m_state[i] = 0; }
    for (int i = 0; i < T * H; ++i) output[i] = 0;
    for (int i = 0; i < 4 * H; ++i) scratch[i] = 0;

    slstm_eval_s8(s.input_q, s.W_q, s.R_q, s.b_q,
                  y, c, n_state, m_state, output, scratch,
                  tc->B, T, I, H, &s.params);

    xlstm_dequantize_s8_to_f32(y, y_out, H, &s.params.y_quant);
    if (m_out) {
        for (int i = 0; i < H; ++i) m_out[i] = m_state[i];
    }
    /* Always dequantize output locally, even if the caller doesn't want it
     * back, so max_err below reflects the whole trajectory. A case can
     * have its worst error at an intermediate timestep rather than the
     * final one that y alone captures (measured for SweepM8: final-y-only
     * showed 0.124726, but an intermediate timestep reaches 1.127354 -
     * see task-4-report.md). Reporting only the final-y number would
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

/* Every INT8 assertion is checked one channel at a time, so the bound
 * that channel is held to has to come from tc->tol_s8_per_channel[j] -
 * computed at generation time in generate_reference.py
 * (compute_tol_s8_per_channel; see that function's module docstring for
 * the full derivation and task-4-report.md for the measured data) from
 * that channel's own error and its own dynamic range, not from
 * tc->tol_s8 (the case-wide max, kept only for coarse reporting) or from
 * any other channel's value. A case-wide denominator or a borrowed bound
 * both hide a small-valued channel behind the case's larger-valued ones.
 *
 * No per-case name-matching lives here on purpose: an earlier version of
 * this file hardcoded a short list of "cases with a carve-out" that had
 * to be kept in sync by hand with the tolerance-selection function, and
 * silently did nothing for a case someone forgot to add to the list.
 * Every case, including the ones with no unusually-small channels, goes
 * through the same per-channel loop below. */
static bool RunSlstmS8Case(const XlstmRefCase* tc) {
    static float y_f[256], m_f[256], output_f[3 * 256];
    EvalSlstmS8Case(tc, y_f, m_f, output_f);
    bool ok = true;
    ok &= ExpectFinite("y", y_f, tc->H);
    ok &= ExpectFinite("m", m_f, tc->H);

    for (int j = 0; j < tc->H; ++j) {
        float tol = tc->tol_s8_per_channel[j];
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
                float tol = tc->tol_s8_per_channel[j];
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

/* Runs the s8 kernel on every case in kSlstmCases and reports, per case,
 * the max absolute error vs the f32 golden y. This is the size-vs-error
 * measurement: how INT8 error grows with H (.docs/SCOPE.md section 8). */
static bool TestS8QuantizationBound() {
    static float y_f[256];
    float max_err = 0.0f;
    for (int i = 0; i < kSlstmCasesCount; ++i) {
        const XlstmRefCase* tc = &kSlstmCases[i];
        float err = EvalSlstmS8Case(tc, y_f, NULL, NULL);
        std::printf("  %-10s H=%-3d max abs error vs f32: %.6f\n", tc->name, tc->H, err);
        if (err > max_err) max_err = err;
    }
    std::printf("  max absolute error vs f32 across all cases: %.6f\n", max_err);
    std::printf("  (oracle-calibrated: scales are derived from this case's own golden\n"
                 "   output, not an independent calibration set - see task-4-report.md's\n"
                 "   'on calibration' note. These are a best case, not a deployment figure.)\n");

    /* Bound set to ~1.5x the measured maximum (0.231354), which is
     * SweepS17's channel 16 at an intermediate timestep - see
     * task-4-report.md for the per-case measurements and why H=17 is an
     * outlier rather than a trend. This aggregate bound covers the whole
     * table with one number, purely for a quick human-readable summary;
     * it is not what keeps this test suite sensitive to a real
     * regression on a specific channel - RunSlstmS8Case's per-channel
     * loop against tc->tol_s8_per_channel is. */
    if (max_err > 0.35f) {
        std::printf("  FAIL: max error %.6f exceeds bound 0.35\n", max_err);
        return false;
    }
    return true;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::printf("[==========] Running sLSTM INT8 kernel tests\n");

    for (int i = 0; i < kSlstmCasesCount; ++i) {
        const XlstmRefCase* tc = &kSlstmCases[i];
        g_tests_run++;
        std::printf("[ RUN      ] sLSTM INT8 %s (H=%d, T=%d)\n", tc->name, tc->H, tc->T);
        if (RunSlstmS8Case(tc)) {
            g_tests_passed++;
            std::printf("[       OK ] sLSTM INT8 %s\n", tc->name);
        } else {
            std::printf("[  FAILED  ] sLSTM INT8 %s\n", tc->name);
        }
    }

    RUN_TEST(TestS8QuantizationBound);

    std::printf("[==========] %d/%d tests passed\n", g_tests_passed, g_tests_run);
    return g_tests_passed == g_tests_run ? 0 : 1;
}
