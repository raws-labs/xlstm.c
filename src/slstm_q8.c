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
 * sLSTM INT8 quantized implementation — pure C99
 *
 * Compute flow:
 *   1. INT8×INT8 matmul → INT32 accumulator (SIMD-ready)
 *   2. Dequantize pre-activations to float
 *   3. Gating + m-stabilization in float
 *   4. Dequantize INT16 states, update in float, requantize to INT16
 *   5. Requantize hidden output to INT8
 * ===========================================================================*/

#include "slstm_q8.h"
#include "xlstm_util.h"

#include <math.h>

void slstm_step_s8(
    const int8_t* x,
    const int8_t* W_q,
    const int8_t* R_q,
    const int32_t* b_q,
    int8_t* y,
    int16_t* c,
    int16_t* n,
    float* m,
    int32_t* scratch,
    int input_size,
    int hidden_size,
    const SlstmS8Params* params)
{
    int H = hidden_size;
    int I = input_size;
    int i, j;

    float wx_scale = params->W_scale * params->x_quant.scale;
    float ry_scale = params->R_scale * params->y_quant.scale;
    float b_scale  = wx_scale; /* bias quantized with input*weight scale */

    int32_t x_zp = params->x_quant.zero_point;
    int32_t y_zp = params->y_quant.zero_point;

    /* 1+2. INT8×INT8 matmul → INT32, then dequantize to float pre-activations.
     *       Scratch is reused as float* (sizeof(int32_t) == sizeof(float)). */
    float* preact = (float*)scratch;
    for (i = 0; i < 4 * H; ++i) {
        int32_t acc_wx = 0;
        for (j = 0; j < I; ++j) {
            acc_wx += (int32_t)W_q[i * I + j] * ((int32_t)x[j] - x_zp);
        }

        int32_t acc_ry = 0;
        for (j = 0; j < H; ++j) {
            acc_ry += (int32_t)R_q[i * H + j] * ((int32_t)y[j] - y_zp);
        }

        preact[i] = (float)acc_wx * wx_scale
                   + (float)acc_ry * ry_scale
                   + (float)b_q[i] * b_scale;
    }

    /* 3-7. Gating + state updates (same math as f32 kernel) */
    for (i = 0; i < H; ++i) {
        float i_raw = preact[i];
        float f_raw = preact[H + i];
        float z_raw = preact[2 * H + i];
        float o_raw = preact[3 * H + i];

        /* 4. Dequantize INT16 states to float (symmetric: zp=0) */
        float c_prev = (float)c[i] * params->c_quant.scale;
        float n_prev = (float)n[i] * params->n_quant.scale;
        float m_prev = m[i];

        /* 3. Stabilized gating */
        float log_f_plus_m = m_prev + log_sigmoid_f32(f_raw);

        float m_new;
        if (n[i] == 0) {
            /* First timestep (n state uninitialized) */
            m_new = i_raw;
        } else {
            m_new = fmaxf(i_raw, log_f_plus_m);
        }

        float i_gate = fminf(expf(i_raw - m_new), 1.0f);
        float f_gate = fminf(expf(log_f_plus_m - m_new), 1.0f);
        float o_gate = sigmoid_f32(o_raw);
        float c_input = tanhf(z_raw);

        /* 5. State updates in float */
        float c_new = f_gate * c_prev + i_gate * c_input;
        float n_new = f_gate * n_prev + i_gate;

        if (params->cell_clip > 0.0f) {
            c_new = fmaxf(-params->cell_clip, fminf(params->cell_clip, c_new));
        }

        float y_new = o_gate * (c_new / fmaxf(n_new, 1e-6f));

        /* 6. Requantize states to INT16 (symmetric: zp=0) */
        float c_q = c_new / params->c_quant.scale;
        c[i] = (int16_t)fmaxf(-32768.0f, fminf(32767.0f, roundf(c_q)));

        float n_q = n_new / params->n_quant.scale;
        n[i] = (int16_t)fmaxf(-32768.0f, fminf(32767.0f, roundf(n_q)));

        /* m stays float */
        m[i] = m_new;

        /* 7. Requantize output to INT8 */
        float y_q = y_new / params->y_quant.scale + (float)params->y_quant.zero_point;
        y[i] = (int8_t)fmaxf(-128.0f, fminf(127.0f, roundf(y_q)));
    }
}

void slstm_eval_s8(
    const int8_t* input,
    const int8_t* W_q,
    const int8_t* R_q,
    const int32_t* b_q,
    int8_t* y,
    int16_t* c,
    int16_t* n,
    float* m,
    int8_t* output,
    int32_t* scratch,
    int batch_size,
    int time_steps,
    int input_size,
    int hidden_size,
    const SlstmS8Params* params)
{
    int B = batch_size;
    int T = time_steps;
    int I = input_size;
    int H = hidden_size;
    int batch, t, i;

    for (batch = 0; batch < B; ++batch) {
        for (t = 0; t < T; ++t) {
            const int8_t* x_t = input + (batch * T + t) * I;

            slstm_step_s8(
                x_t, W_q, R_q, b_q,
                y + batch * H,
                c + batch * H,
                n + batch * H,
                m + batch * H,
                scratch,
                I, H, params);

            /* Copy hidden state to output */
            for (i = 0; i < H; ++i) {
                output[(batch * T + t) * H + i] = y[batch * H + i];
            }
        }
    }
}
