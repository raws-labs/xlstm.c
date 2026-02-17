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
 * mLSTM INT8 quantized kernel — pure C99.
 *
 * Storage: INT8 weights/activations, INT16 cell matrix + normalizer,
 *          float m-stabilizer (scalar per batch element).
 * Compute: INT8×INT8 → INT32 matmul, dequantize to float for gating,
 *          requantize states/output back to integer.
 *
 * Reference: https://arxiv.org/abs/2405.04517
 * ===========================================================================*/

#ifndef MLSTM_Q8_H_
#define MLSTM_Q8_H_

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
    XlstmQuantParam C_quant;   /* cell matrix (INT16) — H×H */
    XlstmQuantParam n_quant;   /* normalizer (INT16) */
} MlstmS8Params;

/* Single timestep of mLSTM (INT8 quantized).
 *
 * State pointers (y, C, n, m) are updated in-place.
 * C is a flattened H×H matrix (row-major, INT16).
 * m is a scalar (single float).
 * Caller must provide a scratch buffer of at least (4*H+2) int32_t. */
void mlstm_step_s8(
    const int8_t* x,          /* [I] */
    const int8_t* W_q,        /* [(4*H+2), I] */
    const int32_t* b_q,       /* [4*H+2] */
    int8_t* y,                /* [H] out */
    int16_t* C,               /* [H*H] in/out */
    int16_t* n,               /* [H] in/out */
    float* m,                 /* [1] in/out */
    int32_t* scratch,         /* [4*H+2] */
    int input_size,
    int hidden_size,
    const MlstmS8Params* params);

/* Full sequence evaluation (INT8 quantized): batch + time loop.
 *
 * Processes input[B, T, I] and writes output[B, T, H] (all INT8).
 * State tensors: y[B,H] INT8, C[B,H*H] INT16, n[B,H] INT16, m[B,1] float.
 * Caller must provide a scratch buffer of at least (4*H+2) int32_t. */
void mlstm_eval_s8(
    const int8_t* input,      /* [B, T, I] */
    const int8_t* W_q,        /* [(4*H+2), I] */
    const int32_t* b_q,       /* [4*H+2] */
    int8_t* y,                /* [B, H] in/out */
    int16_t* C,               /* [B, H*H] in/out */
    int16_t* n,               /* [B, H] in/out */
    float* m,                 /* [B, 1] in/out */
    int8_t* output,           /* [B, T, H] */
    int32_t* scratch,         /* [4*H+2] */
    int batch_size,
    int time_steps,
    int input_size,
    int hidden_size,
    const MlstmS8Params* params);

#ifdef __cplusplus
}
#endif

#endif /* MLSTM_Q8_H_ */
