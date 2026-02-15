/* Copyright 2026 RAWS labs
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
 * mLSTM core implementation — pure C99, only depends on math.h
 * ===========================================================================*/

#include "mlstm.h"
#include "xlstm_util.h"

#include <math.h>

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
    int i, j, r, c;

    /* 1. Compute pre-activations: scratch = W*x + b
     *    scratch layout: [q(H), k(H), v(H), i_raw(1), f_raw(1), o_raw(H)] */
    for (i = 0; i < total; ++i) {
        scratch[i] = b[i];
        for (j = 0; j < I; ++j) {
            scratch[i] += W[i * I + j] * x[j];
        }
    }

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
    float log_f_plus_m = log_sigmoid_f32(f_raw) + m_prev;
    float m_new = fmaxf(log_f_plus_m, i_raw);

    float f_gate = expf(log_f_plus_m - m_new);
    float i_gate = expf(i_raw - m_new);

    /* 5. Update C: C[r][c] = f_gate * C[r][c] + i_gate * k[r] * v[c] */
    for (r = 0; r < H; ++r) {
        for (c = 0; c < H; ++c) {
            C[r * H + c] = f_gate * C[r * H + c] + i_gate * k[r] * v[c];
        }
    }

    /* Optional cell clipping */
    if (params && params->cell_clip > 0.0f) {
        float clip = params->cell_clip;
        for (r = 0; r < H * H; ++r) {
            C[r] = fmaxf(-clip, fminf(clip, C[r]));
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
    float denom = fmaxf(fabsf(qn), expf(-m_new)) + 1e-6f;

    for (j = 0; j < H; ++j) {
        float qC_j = 0.0f;
        for (i = 0; i < H; ++i) {
            qC_j += q[i] * C[i * H + j];
        }
        y[j] = sigmoid_f32(o_raw[j]) * (qC_j / denom);
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
