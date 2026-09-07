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
 * Portable mLSTM core library - pure C99, no framework dependencies.
 *
 * mLSTM is a variant of LSTM from the xLSTM paper (Beck et al., 2024)
 * with a matrix-valued cell state and covariance-based memory retrieval.
 *
 * The query/key width (DQ) and the value width (DV) are separate. They are
 * equal for a square cell, which is the common case, but the memory C is
 * [DQ x DV] in general and the two are not interchangeable.
 *
 * Weight layout - single packed W matrix [(2*DQ+2*DV+2) rows x I cols]:
 *   Rows 0..DQ-1:                    W_q (query projection)
 *   Rows DQ..2DQ-1:                  W_k (key projection)
 *   Rows 2DQ..2DQ+DV-1:              W_v (value projection)
 *   Row  2DQ+DV:                      w_i (scalar input gate)
 *   Row  2DQ+DV+1:                    w_f (scalar forget gate)
 *   Rows 2DQ+DV+2..2DQ+2DV+1:        W_o (output gate)
 *
 * Bias b[2*DQ+2*DV+2] follows the same layout. With DQ == DV == H this is
 * the [(4*H+2) x I] layout by another name.
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

    /* Cap applied to the input and forget preactivations before the
     * stabilizer: cap * tanh(preact / cap). 0 leaves them uncapped, which is
     * what this library did before the knob existed. */
    float gate_soft_cap;

    /* Non-zero writes the UNGATED readout to y - q^T C / denom, without the
     * sigmoid(o) factor - and leaves the output gate to the caller. Nothing is
     * lost by doing so: o's preactivation is already in the caller's scratch,
     * at offset 2*qk_size + v_size + 2. This exists because a model may need a
     * normalization between the recurrence and the gate, which a fused gate
     * gives it no seam to insert. 0 applies the gate as usual. */
    int skip_output_gate;
} MlstmParams;

/* NOTE ON WIDTHS AND HEADS
 *
 * qk_size and v_size are PER-HEAD widths (DHQK and DHV in the reference), not
 * model widths. This kernel implements one head.
 *
 * For a multi-head cell, slice the weights per head and call this function
 * once per head, then concatenate the outputs. State buffers (y, C, n, m)
 * are per head and must not be shared between heads.
 *
 * The cell matrix C is [qk_size x v_size] PER HEAD. Total state therefore
 * grows as num_heads * DQ * DV, not as (num_heads * DQ) * (num_heads * DV).
 * n is [qk_size] and y is [v_size]: the two widths reach different buffers,
 * so a caller that sizes both from one number is wrong whenever they differ.
 *
 * The reference carries mLSTM heads as a leading batched dimension
 * (B, NH, S, DH), with per-head state c(B,NH,DHQK,DHV), n(B,NH,DHQK,1),
 * m(B,NH,1,1), and never entangles them - so the per-head calls really are
 * independent. How the q/k/v/i/f/o rows above map onto a given exporter's
 * fused projection matrix is that exporter's convention and is not fixed by
 * this header: the gate-major head slicing rule in slstm.h is proven for
 * sLSTM only and does not automatically carry over.
 */

/* Single timestep of mLSTM.
 *
 * State pointers (y, C, n, m) are updated in-place.
 * C is a flattened HxH matrix (row-major).
 * m is a scalar (single float).
 * Caller must provide a scratch buffer of at least (4*H+2) floats. */
void mlstm_step_f32(
    const float* x,       /* [input_size] */
    const float* W,       /* [(2*qk_size+2*v_size+2), input_size] */
    const float* b,       /* [2*qk_size+2*v_size+2] */
    float* y,             /* [v_size] out */
    float* C,             /* [qk_size * v_size] in/out */
    float* n,             /* [qk_size] in/out */
    float* m,             /* [1] in/out */
    float* scratch,       /* [2*qk_size+2*v_size+2] caller-provided */
    int input_size,
    int qk_size,
    int v_size,
    const MlstmParams* params);

/* Full sequence evaluation: batch + time loop.
 *
 * Processes input[B, T, I] and writes output[B, T, H].
 * State tensors: y[B,H], C[B,H*H], n[B,H], m[B,1].
 * Caller must provide a scratch buffer of at least (4*H+2) floats. */
void mlstm_eval_f32(
    const float* input,   /* [batch_size, time_steps, input_size] */
    const float* W,       /* [(2*qk_size+2*v_size+2), input_size] */
    const float* b,       /* [2*qk_size+2*v_size+2] */
    float* y,             /* [batch_size, v_size] in/out */
    float* C,             /* [batch_size, qk_size * v_size] in/out */
    float* n,             /* [batch_size, qk_size] in/out */
    float* m,             /* [batch_size, 1] in/out */
    float* output,        /* [batch_size, time_steps, v_size] */
    float* scratch,       /* [2*qk_size+2*v_size+2] caller-provided */
    int batch_size,
    int time_steps,
    int input_size,
    int qk_size,
    int v_size,
    const MlstmParams* params);

#ifdef __cplusplus
}
#endif

#endif /* MLSTM_H_ */
