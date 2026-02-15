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
 * sLSTM core implementation — pure C99, only depends on math.h
 * ===========================================================================*/

#include "slstm.h"
#include "xlstm_util.h"

#include <math.h>

/* ========================================================================== */
/* Core sLSTM computation                                                     */
/* ========================================================================== */

void slstm_step_f32(
    const float* x,
    const float* W,
    const float* R,
    const float* b,
    float* y,
    float* c,
    float* n,
    float* m,
    float* scratch,
    int input_size,
    int hidden_size,
    const SlstmParams* params)
{
    int H = hidden_size;
    int I = input_size;
    int i, j;

    /* Gate pre-activations: scratch = W*x + R*y + b
     * scratch layout: [i_raw, f_raw, z_raw, o_raw] each of size H */
    for (i = 0; i < 4 * H; ++i) {
        scratch[i] = b[i];

        /* W*x contribution */
        for (j = 0; j < I; ++j) {
            scratch[i] += W[i * I + j] * x[j];
        }

        /* R*y contribution */
        for (j = 0; j < H; ++j) {
            scratch[i] += R[i * H + j] * y[j];
        }
    }

    /* Apply sLSTM gating with log-space stabilization */
    for (i = 0; i < H; ++i) {
        float i_raw = scratch[i];
        float f_raw = scratch[H + i];
        float z_raw = scratch[2 * H + i];
        float o_raw = scratch[3 * H + i];

        float c_prev = c[i];
        float n_prev = n[i];
        float m_prev = m[i];

        float log_f_plus_m = m_prev + log_sigmoid_f32(f_raw);

        float m_new;
        if (n_prev == 0.0f) {
            /* First timestep */
            m_new = i_raw;
        } else {
            m_new = fmaxf(i_raw, log_f_plus_m);
        }

        /* Clamped exponential gates */
        float i_gate = fminf(expf(i_raw - m_new), 1.0f);
        float f_gate = fminf(expf(log_f_plus_m - m_new), 1.0f);

        /* Standard activations */
        float o_gate = sigmoid_f32(o_raw);
        float c_input = tanhf(z_raw);

        /* State updates */
        float c_new = f_gate * c_prev + i_gate * c_input;
        float n_new = f_gate * n_prev + i_gate;

        /* Optional cell clipping */
        if (params && params->cell_clip > 0.0f) {
            c_new = fmaxf(-params->cell_clip, fminf(params->cell_clip, c_new));
        }

        /* Normalized output (with epsilon for stability) */
        float y_new = o_gate * (c_new / fmaxf(n_new, 1e-6f));

        /* Store updated states */
        c[i] = c_new;
        n[i] = n_new;
        m[i] = m_new;
        y[i] = y_new;
    }
}

void slstm_eval_f32(
    const float* input,
    const float* W,
    const float* R,
    const float* b,
    float* y,
    float* c,
    float* n,
    float* m,
    float* output,
    float* scratch,
    int batch_size,
    int time_steps,
    int input_size,
    int hidden_size,
    const SlstmParams* params)
{
    int B = batch_size;
    int T = time_steps;
    int I = input_size;
    int H = hidden_size;
    int batch, t, i;

    for (batch = 0; batch < B; ++batch) {
        for (t = 0; t < T; ++t) {
            const float* x_t = input + (batch * T + t) * I;

            slstm_step_f32(
                x_t, W, R, b,
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
