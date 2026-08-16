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
#include "test_config.h"
#include "test_util.h"

// ============================================================================
// Reference test data - same golden values as f32 tests
// ============================================================================

#include "reference_data.h"

// ============================================================================
// Helper: derive and apply quantization scales from case data
// ============================================================================

/* Static buffers sized for the largest case. See test_config.h for what
 * XLSTM_TEST_MAX_H bounds and how it differs from XLSTM_MAX_HIDDEN. */
struct SlstmS8Setup {
    int8_t W_q[4 * XLSTM_TEST_MAX_H * XLSTM_TEST_MAX_H];   /* max [4*H, I] */
    int8_t R_q[4 * XLSTM_TEST_MAX_H * XLSTM_TEST_MAX_H];   /* max [4*H, H] */
    int32_t b_q[4 * XLSTM_TEST_MAX_H];        /* max [4*H] */
    int8_t input_q[3 * XLSTM_TEST_MAX_H];     /* max [T, I] */
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

/* c/n/C are INT16 and strictly symmetric, so they are calibrated with
 * xlstm_quant_symmetric_s16 (see its comment in include/xlstm_quant.h for
 * why xlstm_quant_asymmetric is the wrong tool for them). It used to be a
 * local copy here; it moved into the shared quant layer because the
 * framework adapter integration tests need the identical calibration and
 * a third hand-synchronized copy is one too many.
 *
 * headroom exists because calibrating from a single final-state snapshot
 * (all reference_data.h stores) can badly undershoot the true mid-
 * sequence trajectory peak: measured up to 6.2x for mLSTM SweepM1's n and
 * 3.4x for SweepM8's C (see kStateHeadroom below for the full per-case
 * ratio table).
 *
 * Headroom multiplier for c_quant/n_quant (sLSTM) and n_quant/C_quant
 * (mLSTM). Chosen from measurement, not trial: swept 1x-16x across every
 * case in reference_data.h and picked the point where the worst affected
 * case (mLSTM SweepM8, true/final ratio 3.37x for C) plateaus at its
 * "no-clipping" floor (~0.0388, matching an ablation that replaces C/n
 * with their exact float values) - that happens by 3x and holds through
 * 16x, so 4x keeps a margin above the plateau point without being
 * arbitrarily large. sLSTM is completely insensitive to this value (its
 * dominant error source is INT8 W/x matmul noise, not state clipping),
 * so headroom only matters for mLSTM here, but is applied uniformly to
 * keep one number to justify rather than two.
 * the sweep data. */
static constexpr float kStateHeadroom = 4.0f;

/* generate_reference.py's numpy replica calibrates c_quant/n_quant with
 * the same formula and headroom (GENERATOR_HEADROOM there,
 * XLSTM_GENERATOR_HEADROOM here) to derive every tol_s8_per_channel bound
 * in reference_data.h. The two are independent, hand-synchronized copies
 * of the same constant; this catches the case where one changes and the
 * other doesn't, rather than letting every bound in the file quietly
 * stop matching what it was calibrated against. */
static_assert(kStateHeadroom == XLSTM_GENERATOR_HEADROOM,
              "kStateHeadroom (slstm_s8_test.cc) must match "
              "XLSTM_GENERATOR_HEADROOM (generate_reference.py's "
              "GENERATOR_HEADROOM, emitted into reference_data.h) - "
              "every tol_s8_per_channel bound was derived assuming they agree");

/* Sets up quantized weights/input plus calibrated state-quant scales for
 * one reference case.
 *
 * y/c/n scales are calibrated from the case's own golden f32 values
 * (expected_output/expected_y, expected_state, expected_n) rather than a
 * fixed literal - see the calibration note below for what
 * this oracle-calibrated setup does and does not demonstrate. A fixed
 * 0.01 scale independent of H saturates the INT8 y range once H grows
 * past a couple dozen. y is INT8 (asymmetric, 255 levels); c and n are
 * INT16 and consumed as strictly symmetric by the kernel, so they are
 * calibrated with xlstm_quant_symmetric_s16, not xlstm_quant_asymmetric. */
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

    xlstm_quant_symmetric_s16(tc->expected_n, H, kStateHeadroom, &s->params.n_quant);

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
        xlstm_quant_symmetric_s16(tc->expected_state, H, kStateHeadroom, &s->params.c_quant);
    } else {
        s->params.c_quant.scale = s->params.n_quant.scale;
        s->params.c_quant.zero_point = 0;
    }
}

// ============================================================================
// Table-driven runner
// ============================================================================

/* Runs one case and writes the dequantized hidden state into y_out.
 * m_out (length H), c_out/n_out (length H, dequantized with the same
 * scales the kernel was handed) and output_out (length T*H, dequantized)
 * are optional (pass NULL to skip) - RunSlstmS8Case wants them for the
 * m/c/n/output assertions, TestS8QuantizationBound only wants the error
 * number. Returns the largest absolute deviation from the f32 golden y.
 *
 * c and n are read back on purpose. The kernel contract is caller-owned
 * state updated in place across calls (streaming inference), so a wrong
 * exit state silently corrupts every subsequent call; and the INT16
 * requantize-and-saturate path in slstm_s8.c is reached by nothing else
 * in this repo - the f32 suite does not execute it at all. */
static float EvalSlstmS8Case(const XlstmRefCase* tc, float* y_out,
                              float* m_out, float* c_out, float* n_out,
                              float* output_out) {
    const int H = tc->H, T = tc->T, I = tc->I;

    static SlstmS8Setup s;
    PrepareS8(tc, &s);

    static int8_t y[XLSTM_TEST_MAX_H];
    static int16_t c[XLSTM_TEST_MAX_H], n_state[XLSTM_TEST_MAX_H];
    static float m_state[XLSTM_TEST_MAX_H];
    static int8_t output[3 * XLSTM_TEST_MAX_H];
    static int32_t scratch[4 * XLSTM_TEST_MAX_H];
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
    /* c and n are INT16 and strictly symmetric (zero_point is never read
     * by slstm_s8.c), so dequantizing is a plain scale multiply with the
     * very scales PrepareS8 handed the kernel - no xlstm_dequantize helper
     * exists for int16_t. */
    if (c_out) {
        for (int i = 0; i < H; ++i) c_out[i] = (float)c[i] * s.params.c_quant.scale;
    }
    if (n_out) {
        for (int i = 0; i < H; ++i) n_out[i] = (float)n_state[i] * s.params.n_quant.scale;
    }
    /* Always dequantize output locally, even if the caller doesn't want it
     * back, so max_err below reflects the whole trajectory. A case can
     * have its worst error at an intermediate timestep rather than the
     * final one that y alone captures (measured for SweepM8: final-y-only
     * showed 0.124726, but an intermediate timestep reaches 1.127354 -
     *). Reporting only the final-y number would
     * understate what tolerance the case actually needs. */
    static float output_local[3 * XLSTM_TEST_MAX_H];
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
 * the full derivation) from
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
    /* Buffers below hold one batch (T*H for output, H for the states), and
     * the assertions read batch 0's slice only, so a B>1 case would both
     * overflow output_f at H=256 and check a third of what it ran. Every
     * case in reference_data.h is B=1; fail loudly rather than silently if
     * that ever changes. */
    if (tc->B != 1) {
        std::printf("  FAIL %s: B=%d, but this runner is written for B=1 only\n",
                    tc->name, tc->B);
        return false;
    }

    static float y_f[XLSTM_TEST_MAX_H], m_f[XLSTM_TEST_MAX_H], c_f[XLSTM_TEST_MAX_H], n_f[XLSTM_TEST_MAX_H], output_f[3 * XLSTM_TEST_MAX_H];
    /* The return value is the case-wide max error, which only
     * TestS8QuantizationBound's summary uses; the per-channel and
     * per-tensor assertions below are what decide this case. */
    (void)EvalSlstmS8Case(tc, y_f, m_f, c_f, n_f, output_f);
    bool ok = true;
    ok &= ExpectFinite("y", y_f, tc->H);
    ok &= ExpectFinite("m", m_f, tc->H);
    ok &= ExpectFinite("c", c_f, tc->H);
    ok &= ExpectFinite("n", n_f, tc->H);

    /* Value assertions on the exit state, one bound per element. Without
     * these the suite is finiteness-only on c/n/m: a kernel that zeroes all
     * three after the timestep loop passed 11/11 before they were added,
     * even though the same mutation fails the f32 suite on the first case.
     * m is the project's headline claim (it stays float32 in INT8 precisely
     * because it matters) and Test3 pins it at 100.0, which no INT8 y
     * assertion can see. Per element and not per tensor because a
     * tensor-wide bound left every below-average element individually
     * zeroable - see ExpectStatePerElem in test_util.h and
     * compute_state_tol_per_elem in generate_reference.py. */
    int unasserted = 0;
    ok &= ExpectStatePerElem("m", tc->expected_m, m_f,
                             tc->tol_s8_m_per_elem, tc->H, &unasserted);
    if (tc->expected_state)
        ok &= ExpectStatePerElem("c", tc->expected_state, c_f,
                                 tc->tol_s8_state_per_elem, tc->H, &unasserted);
    ok &= ExpectStatePerElem("n", tc->expected_n, n_f,
                             tc->tol_s8_n_per_elem, tc->H, &unasserted);
    if (unasserted > 0) {
        /* Reported, not hidden: these are elements whose golden is exactly
         * zero or whose honest measured error already spans their whole
         * dynamic range, so no bound can be both non-vacuous and free of
         * false failures. See compute_state_tol_per_elem. They are still
         * covered by the drift detector below, which needs no bound. */
        std::printf("  note: %d of %d exit-state elements have no usable bound "
                    "(unassertable, see compute_state_tol_per_elem; still "
                    "drift-checked)\n",
                    unasserted, 3 * tc->H);
    }

    /* Exit-state drift consistency - the twin of the output floor check
     * further down, for the half of the kernel's contract that check
     * cannot see. See ExpectStateFloorConsistent in test_util.h for the
     * measured factor and what it is guarding against. NULL-guarded the
     * same way as tol_s8_floor_per_channel. */
    ok &= ExpectStateFloorConsistent("m", tc->expected_m, m_f,
                                     tc->tol_s8_m_floor_per_elem, tc->H);
    if (tc->expected_state)
        ok &= ExpectStateFloorConsistent("c", tc->expected_state, c_f,
                                         tc->tol_s8_state_floor_per_elem, tc->H);
    ok &= ExpectStateFloorConsistent("n", tc->expected_n, n_f,
                                     tc->tol_s8_n_floor_per_elem, tc->H);

    /* tol_s8_per_channel is populated for every case in reference_data.h
     * today (build_cases()'s post-processing pass in
     * generate_reference.py), but _emit_table falls back to NULL for a
     * case that omits it from that pass, so this cannot assume
     * non-NULL - guarded the same way PrepareS8 already guards
     * expected_state, rather than segfault on a future case that adds
     * itself without going through that pass. Fallback: the case-wide
     * tol_s8 applied uniformly, matching this file's behavior before
     * per-channel bounds existed. */
    static float uniform_tol[XLSTM_TEST_MAX_H];
    const float* per_channel_tol = tc->tol_s8_per_channel;
    if (!per_channel_tol) {
        for (int j = 0; j < tc->H; ++j) uniform_tol[j] = tc->tol_s8;
        per_channel_tol = uniform_tol;
    }

    /* channel_ref[j] is the largest golden magnitude channel j takes over
     * y and every stored output timestep - the scale the floor check's
     * relative term is measured against below. */
    static float channel_err[XLSTM_TEST_MAX_H], channel_ref[XLSTM_TEST_MAX_H];
    for (int j = 0; j < tc->H; ++j) { channel_err[j] = 0.0f; channel_ref[j] = 0.0f; }

    for (int j = 0; j < tc->H; ++j) {
        float tol = per_channel_tol[j];
        float diff = std::abs(tc->expected_y[j] - y_f[j]);
        if (diff > channel_err[j]) channel_err[j] = diff;
        if (std::abs(tc->expected_y[j]) > channel_ref[j])
            channel_ref[j] = std::abs(tc->expected_y[j]);
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
                float tol = per_channel_tol[j];
                float diff = std::abs(tc->expected_output[idx] - output_f[idx]);
                if (diff > channel_err[j]) channel_err[j] = diff;
                if (std::abs(tc->expected_output[idx]) > channel_ref[j])
                    channel_ref[j] = std::abs(tc->expected_output[idx]);
                if (diff > tol + kRelTol * std::abs(tc->expected_output[idx])) {
                    std::printf("  FAIL output[%d]: expected %.8f, got %.8f (diff %.2e, tol %.4f)\n",
                                idx, tc->expected_output[idx], output_f[idx], diff, tol);
                    ok = false;
                }
            }
        }
    }

    /* Floor consistency: tol_s8_floor_per_channel is what
     * generate_reference.py's numpy replica measured for this exact
     * kernel+calibration. If a kernel change isn't mirrored in the
     * replica, every derived bound quietly stops matching reality
     * instead of failing loudly - this assertion is what makes that
     * failure loud. 1.5x margin: the worst observed replica-vs-real-
     * kernel discrepancy across every channel in this table is ~1.9%
     * (measured), so 1.5x never false-fires today but still
     * catches a real divergence. Guarded the same way as
     * tol_s8_per_channel above; skipped (not defaulted) when absent,
     * since there is nothing meaningful to check consistency against.
     *
     * Be clear about what this is: `1.5 x floor` is TIGHTER than
     * tol_s8_per_channel on most channels, so it - not the per-channel
     * loop above - is the binding bound for this suite. Measured with a
     * multiplicative perturbation of pre-requantization y: 1.001x passes
     * (it is baked into floor by construction), 1.002x already fails
     * SweepS16 and SweepS64, 1.005x fails 4 of 11 cases (SweepS16,
     * SweepS64, Head2, Head2b) and 1.05x fails 7 of 11 - in every one of
     * those runs the failing assertion is this one, not the loop above.
     * That sensitivity is the point, but it also means this check fires
     * for a genuine kernel regression at least as often as for generator
     * drift, hence the two-sided message below.
     *
     * kFloorEps + kRelTol: see kFloorEps in test_util.h. Without them a
     * channel whose floor is exactly 0.0 demands bit-exact float output. */
    if (tc->tol_s8_floor_per_channel) {
        for (int j = 0; j < tc->H; ++j) {
            float floor = tc->tol_s8_floor_per_channel[j];
            float bound = floor * 1.5f + kFloorEps + kRelTol * channel_ref[j];
            if (channel_err[j] > bound) {
                std::printf("  FAIL floor-consistency ch[%d]: measured error %.6f exceeds "
                            "bound %.6f (1.5x the numpy replica's predicted floor %.6f, "
                            "plus float slack) - either the C kernel regressed or "
                            "generate_reference.py's replica drifted away from it; check "
                            "the kernel first if you just changed one\n",
                            j, channel_err[j], bound, floor);
                ok = false;
            }
        }
    }
    return ok;
}

/* Runs the s8 kernel on every case in kSlstmCases and reports, per case,
 * the max absolute error vs the f32 golden y. This is the size-vs-error
 * measurement: how INT8 error grows with H. */
static bool TestS8QuantizationBound() {
    static float y_f[XLSTM_TEST_MAX_H];
    float max_err = 0.0f;
    for (int i = 0; i < kSlstmCasesCount; ++i) {
        const XlstmRefCase* tc = &kSlstmCases[i];
        /* Same B=1 hazard the per-case runner guards, and this loop runs
         * unconditionally after it: EvalSlstmS8Case's static output buffer
         * holds T*H and the kernel writes output[(batch*T+t)*H+i], so a
         * B=2,T=3,H=256 case would overrun it by 768 bytes through this
         * path even though RunSlstmS8Case refused to run it. */
        if (tc->B != 1) {
            std::printf("  FAIL %s: B=%d, but this summary is written for B=1 only\n",
                        tc->name, tc->B);
            return false;
        }
        float err = EvalSlstmS8Case(tc, y_f, NULL, NULL, NULL, NULL);
        std::printf("  %-10s H=%-3d max abs error vs f32: %.6f (worst-channel bound: %.4f)\n",
                     tc->name, tc->H, err, tc->tol_s8);
        if (err > max_err) max_err = err;
    }
    std::printf("  max absolute error vs f32 across all cases: %.6f\n", max_err);
    std::printf("  (oracle-calibrated: scales are derived from this case's own golden\n"
                 "   output, not an independent calibration set. These are a\n"
                 "   best case, not a deployment figure.)\n");

    /* Bound set to ~1.5x the measured maximum (0.231354), which is
     * SweepS17's channel 16 at an intermediate timestep - see
     * the per-case measurements show why H=17 is an
     * outlier rather than a trend. This aggregate bound covers the whole
     * table with one number, purely for a quick human-readable summary;
     * it is not what keeps this test suite sensitive to a real regression
     * on a specific channel. That sensitivity comes from RunSlstmS8Case,
     * where the effective per-channel bound is
     * min(tol_s8_per_channel, 1.5 x tol_s8_floor_per_channel + slack) -
     * the floor term is the tighter of the two on most channels and is
     * what actually binds. */
    if (max_err > 0.35f) {
        std::printf("  FAIL: max error %.6f exceeds bound 0.35\n", max_err);
        return false;
    }
    return true;
}

// ============================================================================
// Main
// ============================================================================

#ifndef XLSTM_TEST_MAIN
#define XLSTM_TEST_MAIN main
#endif

int XLSTM_TEST_MAIN(void) {
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
