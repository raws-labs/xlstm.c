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
 * Portable mLSTM core library — pure C99, no framework dependencies.
 *
 * mLSTM is a variant of LSTM from the xLSTM paper (Hochreiter et al., 2024)
 * with a matrix-valued cell state and covariance-based memory retrieval.
 *
 * Weight layout — single packed W matrix [(4*H+2) rows × I cols]:
 *   Rows 0..H-1:       W_q (query projection)
 *   Rows H..2H-1:      W_k (key projection)
 *   Rows 2H..3H-1:     W_v (value projection)
 *   Row  3H:            w_i (scalar input gate)
 *   Row  3H+1:          w_f (scalar forget gate)
 *   Rows 3H+2..4H+1:   W_o (output gate)
 *
 * Bias b[4*H+2] follows the same layout.
 *
 * Reference: https://arxiv.org/abs/2405.04517
 * ===========================================================================*/

#ifndef MLSTM_H_
#define MLSTM_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float cell_clip; /* 0 = no clipping */
} MlstmParams;

/* Single timestep of mLSTM.
 *
 * State pointers (y, C, n, m) are updated in-place.
 * C is a flattened H×H matrix (row-major).
 * m is a scalar (single float).
 * Caller must provide a scratch buffer of at least (4*H+2) floats. */
void mlstm_step_f32(
    const float* x,       /* [input_size] */
    const float* W,       /* [(4*hidden_size+2), input_size] */
    const float* b,       /* [4*hidden_size+2] */
    float* y,             /* [hidden_size] out */
    float* C,             /* [hidden_size * hidden_size] in/out */
    float* n,             /* [hidden_size] in/out */
    float* m,             /* [1] in/out */
    float* scratch,       /* [4*hidden_size+2] caller-provided */
    int input_size,
    int hidden_size,
    const MlstmParams* params);

/* Full sequence evaluation: batch + time loop.
 *
 * Processes input[B, T, I] and writes output[B, T, H].
 * State tensors: y[B,H], C[B,H*H], n[B,H], m[B,1].
 * Caller must provide a scratch buffer of at least (4*H+2) floats. */
void mlstm_eval_f32(
    const float* input,   /* [batch_size, time_steps, input_size] */
    const float* W,       /* [(4*hidden_size+2), input_size] */
    const float* b,       /* [4*hidden_size+2] */
    float* y,             /* [batch_size, hidden_size] in/out */
    float* C,             /* [batch_size, hidden_size * hidden_size] in/out */
    float* n,             /* [batch_size, hidden_size] in/out */
    float* m,             /* [batch_size, 1] in/out */
    float* output,        /* [batch_size, time_steps, hidden_size] */
    float* scratch,       /* [4*hidden_size+2] caller-provided */
    int batch_size,
    int time_steps,
    int input_size,
    int hidden_size,
    const MlstmParams* params);

#ifdef __cplusplus
}
#endif

#endif /* MLSTM_H_ */
