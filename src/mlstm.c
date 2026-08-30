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
 * mLSTM core implementation - pure C99, only depends on math.h
 * ===========================================================================*/

#include "mlstm.h"
#include "xlstm_simd.h"
#include "xlstm_util.h"

#include <math.h>
#include <string.h>

/* ========================================================================== */
/* Core mLSTM computation                                                     */
/* ========================================================================== */

void mlstm_step_f32(
    const float* x,
    const float* W,
    const float* b,
    float* y,
    float* C,
    float* n,
    float* m,
    float* scratch,
    int input_size,
    int hidden_size,
    const MlstmParams* params)
{
    int H = hidden_size;
    int I = input_size;
    int total = 4 * H + 2;
    int i, j;

    /* 1. Compute pre-activations: scratch = W*x + b
     *    scratch layout: [q(H), k(H), v(H), i_raw(1), f_raw(1), o_raw(H)] */
    for (i = 0; i < total; ++i)
        scratch[i] = b[i];
    xlstm_matvec_f32(W, x, scratch, total, I);

    /* 2. Extract projections from scratch */
    float* q     = scratch;              /* [H] */
    float* k     = scratch + H;          /* [H] */
    float* v     = scratch + 2 * H;      /* [H] */
    float i_raw  = scratch[3 * H];       /* scalar */
    float f_raw  = scratch[3 * H + 1];   /* scalar */
    float* o_raw = scratch + 3 * H + 2;  /* [H] */

    /* 3. Scale key: k /= sqrt(H) */
    float k_scale = 1.0f / sqrtf((float)H);
    for (i = 0; i < H; ++i) {
        k[i] *= k_scale;
    }

    /* 4. Stabilized gates (scalar m) */
    float m_prev = m[0];
    float log_f_plus_m = xlstm_gate_log_sigmoidf(f_raw) + m_prev;
    float m_new = fmaxf(log_f_plus_m, i_raw);

    float f_gate = xlstm_gate_expf(log_f_plus_m - m_new);
    float i_gate = xlstm_gate_expf(i_raw - m_new);

    /* 5. Update C: C[r][c] = f_gate * C[r][c] + i_gate * k[r] * v[c] */
    xlstm_rank1_update_f32(C, f_gate, i_gate, k, v, H);

    /* Optional cell clipping */
    if (params && params->cell_clip > 0.0f) {
        float clip = params->cell_clip;
        for (i = 0; i < H * H; ++i) {
            C[i] = fmaxf(-clip, fminf(clip, C[i]));
        }
    }

    /* 6. Update n: n = f_gate * n + i_gate * k */
    for (i = 0; i < H; ++i) {
        n[i] = f_gate * n[i] + i_gate * k[i];
    }

    /* 7. Update m */
    m[0] = m_new;

    /* 8. Compute output: y = sigmoid(o) * (q^T C) / max(|q^T n|, exp(-m)) + eps
     *
     *    q^T C gives a vector of size H: out[j] = sum_i q[i] * C[i*H + j]
     *    q^T n gives a scalar: qn = sum_i q[i] * n[i] */
    float qn = 0.0f;
    for (i = 0; i < H; ++i) {
        qn += q[i] * n[i];
    }
    float denom = fmaxf(fabsf(qn), xlstm_gate_expf(-m_new)) + 1e-6f;

    /* q^T * C via vecmat (scatter-accumulate with contiguous row access) */
    float qC[XLSTM_MAX_HIDDEN];
    memset(qC, 0, (size_t)H * sizeof(float));
    xlstm_vecmat_f32(q, C, qC, H, H);
    for (j = 0; j < H; ++j) {
        y[j] = xlstm_gate_sigmoidf(o_raw[j]) * (qC[j] / denom);
    }
}

void mlstm_eval_f32(
    const float* input,
    const float* W,
    const float* b,
    float* y,
    float* C,
    float* n,
    float* m,
    float* output,
    float* scratch,
    int batch_size,
    int time_steps,
    int input_size,
    int hidden_size,
    const MlstmParams* params)
{
    int B = batch_size;
    int T = time_steps;
    int I = input_size;
    int H = hidden_size;
    int batch, t, i;

    for (batch = 0; batch < B; ++batch) {
        for (t = 0; t < T; ++t) {
            const float* x_t = input + (batch * T + t) * I;

            mlstm_step_f32(
                x_t, W, b,
                y + batch * H,
                C + batch * H * H,
                n + batch * H,
                m + batch * 1,
                scratch,
                I, H, params);

            /* Copy hidden state to output */
            for (i = 0; i < H; ++i) {
                output[(batch * T + t) * H + i] = y[batch * H + i];
            }
        }
    }
}
