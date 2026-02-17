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
 * mLSTM INT8 quantized implementation — pure C99
 *
 * Compute flow:
 *   1. INT8×INT8 matmul → INT32 accumulator (SIMD-ready)
 *   2. Dequantize pre-activations to float
 *   3. Key scaling, stabilized gating in float
 *   4. Dequantize INT16 states, update in float, requantize to INT16
 *   5. Compute output via q^T C / normalizer, requantize to INT8
 * ===========================================================================*/

#include "mlstm_q8.h"
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
    int hidden_size,
    const MlstmS8Params* params)
{
    int H = hidden_size;
    int I = input_size;
    int total = 4 * H + 2;
    int i, j, r, c;

    float wx_scale = params->W_scale * params->x_quant.scale;
    int32_t x_zp = params->x_quant.zero_point;

    /* 1+2. INT8×INT8 matmul → float pre-activations.
     *       scratch layout: [q(H), k(H), v(H), i_raw(1), f_raw(1), o_raw(H)] */
    float* preact = (float*)scratch;
    for (i = 0; i < total; ++i) {
        int32_t acc = 0;
        for (j = 0; j < I; ++j) {
            acc += (int32_t)W_q[i * I + j] * ((int32_t)x[j] - x_zp);
        }
        preact[i] = (float)acc * wx_scale + (float)b_q[i] * wx_scale;
    }

    /* Extract projections from pre-activations */
    float* q     = preact;              /* [H] */
    float* k     = preact + H;          /* [H] */
    float* v     = preact + 2 * H;      /* [H] */
    float i_raw  = preact[3 * H];       /* scalar */
    float f_raw  = preact[3 * H + 1];   /* scalar */
    float* o_raw = preact + 3 * H + 2;  /* [H] */

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

    /* 5. Update C: dequant → float update → requant */
    for (r = 0; r < H; ++r) {
        for (c = 0; c < H; ++c) {
            float C_prev = (float)C[r * H + c] * params->C_quant.scale;
            float C_new = f_gate * C_prev + i_gate * k[r] * v[c];

            if (params->cell_clip > 0.0f) {
                C_new = fmaxf(-params->cell_clip, fminf(params->cell_clip, C_new));
            }

            float C_q = C_new / params->C_quant.scale;
            C[r * H + c] = (int16_t)fmaxf(-32768.0f, fminf(32767.0f, roundf(C_q)));
        }
    }

    /* 6. Update n: dequant → float update → requant */
    for (i = 0; i < H; ++i) {
        float n_prev = (float)n[i] * params->n_quant.scale;
        float n_new = f_gate * n_prev + i_gate * k[i];
        float n_q = n_new / params->n_quant.scale;
        n[i] = (int16_t)fmaxf(-32768.0f, fminf(32767.0f, roundf(n_q)));
    }

    /* 7. Update m */
    m[0] = m_new;

    /* 8. Compute output: y = sigmoid(o) * (q^T C) / max(|q^T n|, exp(-m)) + eps
     *    Read back quantized states for output computation. */
    float qn = 0.0f;
    for (i = 0; i < H; ++i) {
        float n_f = (float)n[i] * params->n_quant.scale;
        qn += q[i] * n_f;
    }
    float denom = fmaxf(fabsf(qn), expf(-m_new)) + 1e-6f;

    for (j = 0; j < H; ++j) {
        float qC_j = 0.0f;
        for (i = 0; i < H; ++i) {
            float C_f = (float)C[i * H + j] * params->C_quant.scale;
            qC_j += q[i] * C_f;
        }
        float y_new = sigmoid_f32(o_raw[j]) * (qC_j / denom);

        /* Requantize output to INT8 */
        float y_q = y_new / params->y_quant.scale + (float)params->y_quant.zero_point;
        y[j] = (int8_t)fmaxf(-128.0f, fminf(127.0f, roundf(y_q)));
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
    int hidden_size,
    const MlstmS8Params* params)
{
    int B = batch_size;
    int T = time_steps;
    int I = input_size;
    int H = hidden_size;
    int batch, t, i;

    for (batch = 0; batch < B; ++batch) {
        for (t = 0; t < T; ++t) {
            const int8_t* x_t = input + (batch * T + t) * I;

            mlstm_step_s8(
                x_t, W_q, b_q,
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
