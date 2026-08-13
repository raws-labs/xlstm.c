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
 * Portable sLSTM core library - pure C99, no framework dependencies.
 *
 * sLSTM is a variant of LSTM from the xLSTM paper (Beck et al., 2024)
 * with exponential gating and normalizer state for improved gradient flow
 * and numerical stability.
 *
 * Reference: https://arxiv.org/abs/2405.04517
 * ===========================================================================*/

#ifndef SLSTM_H_
#define SLSTM_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float cell_clip; /* 0 = no clipping */
} SlstmParams;

/* NOTE ON HIDDEN SIZE AND HEADS
 *
 * hidden_size is the PER-HEAD width (DH in the NX-AI reference), not the
 * model width. This kernel implements one head.
 *
 * For a multi-head cell, slice the weights per head and call this function
 * once per head, then concatenate the outputs. State buffers (y, c, n, m)
 * are per head and must not be shared between heads.
 *
 * The slicing is not the obvious one. The reference's fused weight rows run
 * gate-major over the fused width Hf = num_heads * hidden_size, so a head's
 * four gate blocks are strided across the matrix rather than contiguous.
 * With g in 0..3 the gate index (i, f, z, o), h the head, j in
 * 0..hidden_size-1, and R_stack the reference's recurrent parameter
 * (sLSTMCell_vanilla's _recurrent_kernel_, shape
 * (num_heads, 4*hidden_size, hidden_size) - one [4*DH, DH] matrix per head,
 * already in this library's packing):
 *
 *     W_h[g*hidden_size + j][:] = W_fused[g*Hf + h*hidden_size + j][:]
 *     b_h[g*hidden_size + j]    = b_fused[g*Hf + h*hidden_size + j]
 *     R_h                       = R_stack[h]  (no cross-head recurrence, so
 *                                              each head's [4*DH, DH] block
 *                                              is already contiguous)
 *     y_h[j]                    = y_fused[h*hidden_size + j]  (c, n, m too)
 *
 * Slicing rows [h*4*hidden_size, (h+1)*4*hidden_size) instead - the obvious
 * guess - yields a silently different model. See test/slstm_test.cc
 * TestHeadComposition for a worked example, and
 * test/derive_multihead_layout.py for how the rule was established.
 */

/* Single timestep of sLSTM.
 *
 * All state pointers (y, c, n, m) are updated in-place.
 * Caller must provide a scratch buffer of at least 4*hidden_size floats. */
void slstm_step_f32(
    const float* x,       /* [input_size] */
    const float* W,       /* [4*hidden_size, input_size] */
    const float* R,       /* [4*hidden_size, hidden_size] */
    const float* b,       /* [4*hidden_size] */
    float* y,             /* [hidden_size] in/out */
    float* c,             /* [hidden_size] in/out */
    float* n,             /* [hidden_size] in/out */
    float* m,             /* [hidden_size] in/out */
    float* scratch,       /* [4*hidden_size] caller-provided */
    int input_size,
    int hidden_size,
    const SlstmParams* params);

/* Full sequence evaluation: batch + time loop.
 *
 * Processes input[B, T, I] and writes output[B, T, H].
 * State tensors (y, c, n, m) are [B, H] and updated in-place.
 * Caller must provide a scratch buffer of at least 4*hidden_size floats. */
void slstm_eval_f32(
    const float* input,   /* [batch_size, time_steps, input_size] */
    const float* W,       /* [4*hidden_size, input_size] */
    const float* R,       /* [4*hidden_size, hidden_size] */
    const float* b,       /* [4*hidden_size] */
    float* y,             /* [batch_size, hidden_size] in/out */
    float* c,             /* [batch_size, hidden_size] in/out */
    float* n,             /* [batch_size, hidden_size] in/out */
    float* m,             /* [batch_size, hidden_size] in/out */
    float* output,        /* [batch_size, time_steps, hidden_size] */
    float* scratch,       /* [4*hidden_size] caller-provided */
    int batch_size,
    int time_steps,
    int input_size,
    int hidden_size,
    const SlstmParams* params);

#ifdef __cplusplus
}
#endif

#endif /* SLSTM_H_ */
