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
 * mLSTM INT8 quantized kernel - pure C99.
 *
 * Storage: INT8 weights/activations, INT16 cell matrix + normalizer,
 *          float m-stabilizer (scalar per batch element).
 * Compute: INT8xINT8 -> INT32 matmul, dequantize to float for gating,
 *          requantize states/output back to integer.
 *
 * Reference: https://arxiv.org/abs/2405.04517
 * ===========================================================================*/

#ifndef MLSTM_S8_H_
#define MLSTM_S8_H_

#include "xlstm_quant.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float cell_clip;
    float W_scale;             /* no R for mLSTM */
    XlstmQuantParam x_quant;
    XlstmQuantParam y_quant;
    XlstmQuantParam C_quant;   /* cell matrix (INT16) - [qk_size x v_size] */
    XlstmQuantParam n_quant;   /* normalizer (INT16) - [qk_size] */

    /* As documented on MlstmParams in mlstm.h. */
    float gate_soft_cap;
} MlstmS8Params;

/* NOTE ON HIDDEN SIZE AND HEADS
 *
 * qk_size and v_size are PER-HEAD widths (DHQK and DHV in the reference), not
 * model widths. This kernel implements one head.
 *
 * For a multi-head cell, slice the weights per head and call this function
 * once per head, then concatenate the outputs. State buffers (y, C, n, m)
 * are per head and must not be shared between heads.
 *
 * The cell matrix C is [qk_size x v_size] PER HEAD. Total state
 * therefore grows as num_heads * DH * DH, not as (num_heads * DH)^2 - which
 * is the whole memory argument for running xLSTM on an MCU.
 *
 * See the note above mlstm_step_f32 in mlstm.h on why the sLSTM gate-major
 * slicing rule does not automatically carry over to mLSTM. Quantization is
 * per call: MlstmS8Params carries one set of scales, so each head may be
 * calibrated separately.
 */

/* Single timestep of mLSTM (INT8 quantized).
 *
 * State pointers (y, C, n, m) are updated in-place.
 * C is a flattened HxH matrix (row-major, INT16).
 * m is a scalar (single float).
 * Caller must provide a scratch buffer of at least
 * (2*qk_size+2*v_size+2) int32_t. */
void mlstm_step_s8(
    const int8_t* x,          /* [I] */
    const int8_t* W_q,        /* [(2*qk_size+2*v_size+2), I] */
    const int32_t* b_q,       /* [2*qk_size+2*v_size+2] */
    int8_t* y,                /* [v_size] out */
    int16_t* C,               /* [qk_size*v_size] in/out */
    int16_t* n,               /* [qk_size] in/out */
    float* m,                 /* [1] in/out */
    int32_t* scratch,         /* [2*qk_size+2*v_size+2] */
    int input_size,
    int qk_size,
    int v_size,
    const MlstmS8Params* params);

/* Full sequence evaluation (INT8 quantized): batch + time loop.
 *
 * Processes input[B, T, I] and writes output[B, T, H] (all INT8).
 * State tensors: y[B,H] INT8, C[B,H*H] INT16, n[B,H] INT16, m[B,1] float.
 * Caller must provide a scratch buffer of at least
 * (2*qk_size+2*v_size+2) int32_t. */
void mlstm_eval_s8(
    const int8_t* input,      /* [B, T, I] */
    const int8_t* W_q,        /* [(2*qk_size+2*v_size+2), I] */
    const int32_t* b_q,       /* [2*qk_size+2*v_size+2] */
    int8_t* y,                /* [B, v_size] in/out */
    int16_t* C,               /* [B, qk_size*v_size] in/out */
    int16_t* n,               /* [B, qk_size] in/out */
    float* m,                 /* [B, 1] in/out */
    int8_t* output,           /* [B, T, v_size] */
    int32_t* scratch,         /* [2*qk_size+2*v_size+2] */
    int batch_size,
    int time_steps,
    int input_size,
    int qk_size,
    int v_size,
    const MlstmS8Params* params);

#ifdef __cplusplus
}
#endif

#endif /* MLSTM_S8_H_ */
