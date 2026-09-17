/* Copyright 2026 RAWS Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =========================================================================
 * mLSTM INT8 quantized implementation - pure C99
 *
 * Compute flow:
 *   1. INT8xINT8 matmul -> INT32 accumulator (SIMD-ready)
 *   2. Dequantize pre-activations to float
 *   3. Key scaling, stabilized gating in float
 *   4. Dequantize INT16 states, update in float, requantize to INT16
 *   5. Compute output via q^T C / normalizer, requantize to INT8
 * ===========================================================================*/

#include "mlstm_s8.h"
#include "xlstm_simd.h"
#include "xlstm_util.h"

#include <math.h>

void mlstm_step_s8(
    const int8_t* x,
    const int8_t* W_q,
    const int32_t* b_q,
    int8_t* y,
    int16_t* C,
    int16_t* n,
    float* m,
    int32_t* scratch,
    int input_size,
    int qk_size,
    int v_size,
    const MlstmS8Params* params)
{
    int DQ = qk_size;
    int DV = v_size;
    int I = input_size;
    int total = 2 * DQ + 2 * DV + 2;
    int i, j;

    float wx_scale = params->W_scale * params->x_quant.scale;
    int32_t x_zp = params->x_quant.zero_point;

    /* 1+2. INT8xINT8 matmul -> float pre-activations.
     *       scratch layout:
     *         [q(DQ), k(DQ), v(DV), i_raw(1), f_raw(1), o_raw(DV)] */
    int32_t acc[4 * XLSTM_MAX_HIDDEN + 2];
    xlstm_matvec_s8(W_q, x, acc, total, I, x_zp);

    float* preact = (float*)scratch;
    for (i = 0; i < total; ++i) {
        preact[i] = (float)acc[i] * wx_scale + (float)b_q[i] * wx_scale;
    }

    /* Extract projections from pre-activations */
    float* q     = preact;                        /* [DQ] */
    float* k     = preact + DQ;                   /* [DQ] */
    float* v     = preact + 2 * DQ;               /* [DV] */
    float i_raw  = preact[2 * DQ + DV];           /* scalar */
    float f_raw  = preact[2 * DQ + DV + 1];       /* scalar */
    float* o_raw = preact + 2 * DQ + DV + 2;      /* [DV] */

    /* 3. Scale query: q /= sqrt(DQ) - the q/k contraction sets it, not DV.
     *    Applied to q at readout, not to k on the way in, so C and n store
     *    unscaled k and match the reference's state. See mlstm.c. */
    float q_scale = 1.0f / sqrtf((float)DQ);
    for (i = 0; i < DQ; ++i) {
        q[i] *= q_scale;
    }

    /* 3b. Optional soft cap on the two gate preactivations. */
    if (params->gate_soft_cap > 0.0f) {
        i_raw = xlstm_soft_cap(i_raw, params->gate_soft_cap);
        f_raw = xlstm_soft_cap(f_raw, params->gate_soft_cap);
    }

    /* 4. Stabilized gates (scalar m) */
    float m_prev = m[0];
    float log_f_plus_m = xlstm_gate_log_sigmoidf(f_raw) + m_prev;
    float m_new = xlstm_maxf(log_f_plus_m, i_raw);

    /* No zero-exponent shortcut here, though one of these two arguments is
     * always zero. slstm_s8.c takes it because its xlstm_minf(., 1.0f) makes
     * it an identity; this spelling has no clamp, so at log_f_plus_m ==
     * i_raw == +inf the shortcut would answer 1 where the exponential
     * answers NaN. Its worth is what settles it: these two are per TIMESTEP,
     * not per hidden unit, so the shortcut would remove one exponential from
     * a step whose cost is the O(DQ*DV) loop below. */
    float f_gate = xlstm_gate_expf(log_f_plus_m - m_new);
    float i_gate = xlstm_gate_expf(i_raw - m_new);

    /* 5. Update C: dequant -> float update -> requant.
     *
     * The loop moved behind the backend contract. It was the only O(DQ*DV)
     * work in either INT8 cell that no backend could reach, so selecting a
     * backend changed the f32 step and left this one instruction for
     * instruction identical. The arithmetic and the requantization rounding
     * are unchanged; xlstm_simd_scalar.h holds the body ref runs and the
     * definition of bit-exactness for the rest.
     *
     * THE DIVIDE INSIDE IT STAYS, deliberately. Hoisting 1/scale out and
     * multiplying would remove a vdiv per element - the most expensive
     * instruction in the loop on Cortex-M4 - but x * (1/s) rounds twice where
     * x / s rounds once, so the two are not the same function. Measured over
     * 4M random (x, s) pairs drawn from the scale range these kernels
     * calibrate to: 25.9% of quotients differ before rounding and 1 in ~7,400
     * still differs after the INT16 round, i.e. a one-LSB state error. The
     * gate happens not to contain such a pair (0 of its 15,154 state
     * requantizations move), which is exactly why it is not evidence of
     * equivalence. Measured and rejected - do not "optimize" this without
     * redoing that measurement. On x86-64 it is worth 1 to 5%: callgrind puts
     * it at 2.5% of the loop's instructions, and a reciprocal-hoist probe
     * measured 0.84s against 0.85s minimum user time over twelve runs. */
    xlstm_rank1_update_s16(C, f_gate, i_gate, k, v, params->C_quant.scale,
                           params->cell_clip, DQ, DV);

    /* 6. Update n: dequant -> float update -> requant */
    for (i = 0; i < DQ; ++i) {
        float n_prev = (float)n[i] * params->n_quant.scale;
        float n_new = f_gate * n_prev + i_gate * k[i];
        float n_q = n_new / params->n_quant.scale;
        n[i] = (int16_t)xlstm_round_clamp_i32(n_q, -32768.0f, 32767.0f);
    }

    /* 7. Update m */
    m[0] = m_new;

    /* 8. Compute output: y = sigmoid(o) * (q^T C) / max(|q^T n|, exp(-m)) + eps
     *    Read back quantized states for output computation. */
    float qn = 0.0f;
    for (i = 0; i < DQ; ++i) {
        float n_f = (float)n[i] * params->n_quant.scale;
        qn += q[i] * n_f;
    }
    float denom = xlstm_maxf(fabsf(qn), xlstm_gate_expf(-m_new)) + 1e-6f;

    /* q^T C through the backend, so the readout walks C by contiguous rows.
     * Spelled inline it indexed C[i * DV + j] with j fixed in the inner loop,
     * which is a stride of DV int16 per step. One running accumulator per j,
     * summed in ascending i, is the order it had. */
    float qC[XLSTM_MAX_HIDDEN];
    for (j = 0; j < DV; ++j) {
        qC[j] = 0.0f;
    }
    xlstm_vecmat_s16(q, C, qC, params->C_quant.scale, DQ, DV);

    for (j = 0; j < DV; ++j) {
        float y_new = params->skip_output_gate
                          ? (qC[j] / denom)
                          : xlstm_gate_sigmoidf(o_raw[j]) * (qC[j] / denom);

        /* Requantize output to INT8 */
        float y_q = y_new / params->y_quant.scale + (float)params->y_quant.zero_point;
        y[j] = (int8_t)xlstm_round_clamp_i32(y_q, -128.0f, 127.0f);
    }
}

void mlstm_eval_s8(
    const int8_t* input,
    const int8_t* W_q,
    const int32_t* b_q,
    int8_t* y,
    int16_t* C,
    int16_t* n,
    float* m,
    int8_t* output,
    int32_t* scratch,
    int batch_size,
    int time_steps,
    int input_size,
    int qk_size,
    int v_size,
    const MlstmS8Params* params)
{
    int B = batch_size;
    int T = time_steps;
    int I = input_size;
    int DQ = qk_size;
    int DV = v_size;
    int batch, t, i;

    for (batch = 0; batch < B; ++batch) {
        for (t = 0; t < T; ++t) {
            const int8_t* x_t = input + (batch * T + t) * I;

            mlstm_step_s8(
                x_t, W_q, b_q,
                y + batch * DV,
                C + batch * DQ * DV,
                n + batch * DQ,
                m + batch * 1,
                scratch,
                I, DQ, DV, params);

            /* Copy hidden state to output */
            for (i = 0; i < DV; ++i) {
                output[(batch * T + t) * DV + i] = y[batch * DV + i];
            }
        }
    }
}
