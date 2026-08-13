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
 * sLSTM INT8 quantized kernel - pure C99.
 *
 * Storage: INT8 weights/activations, INT16 states, float m-stabilizer.
 * Compute: INT8xINT8 -> INT32 matmul, dequantize to float for gating,
 *          requantize states/output back to integer.
 *
 * Reference: https://arxiv.org/abs/2405.04517
 * ===========================================================================*/

#ifndef SLSTM_S8_H_
#define SLSTM_S8_H_

#include "xlstm_quant.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float cell_clip;
    /* Weight scales (symmetric, zp=0) */
    float W_scale;
    float R_scale;
    /* Tensor quantization params */
    XlstmQuantParam x_quant;     /* input */
    XlstmQuantParam y_quant;     /* hidden state / output */
    XlstmQuantParam c_quant;     /* cell state (INT16) */
    XlstmQuantParam n_quant;     /* normalizer (INT16) */
    /* m stays float - no param needed */
} SlstmS8Params;

/* NOTE ON HIDDEN SIZE AND HEADS
 *
 * hidden_size is the PER-HEAD width (DH in the NX-AI reference), not the
 * model width. This kernel implements one head.
 *
 * For a multi-head cell, slice the weights per head and call this function
 * once per head, then concatenate the outputs. State buffers (y, c, n, m)
 * are per head and must not be shared between heads. The slicing is
 * gate-major over the fused width and is NOT contiguous per head - see the
 * note above slstm_step_f32 in slstm.h for the exact expressions, and
 * test/slstm_test.cc TestHeadComposition for a worked example.
 *
 * Quantization is per call, not per model: SlstmS8Params carries one set of
 * scales, so each head may be calibrated separately. Quantizing the fused
 * matrix once and then slicing it is also valid but makes every head share a
 * single W_scale/R_scale.
 */

/* Single timestep of sLSTM (INT8 quantized).
 *
 * All state pointers (y, c, n, m) are updated in-place.
 * Caller must provide a scratch buffer of at least 4*hidden_size int32_t. */
void slstm_step_s8(
    const int8_t* x,          /* [input_size] */
    const int8_t* W_q,        /* [4*H, I] */
    const int8_t* R_q,        /* [4*H, H] */
    const int32_t* b_q,       /* [4*H] */
    int8_t* y,                /* [H] in/out */
    int16_t* c,               /* [H] in/out */
    int16_t* n,               /* [H] in/out */
    float* m,                 /* [H] in/out */
    int32_t* scratch,         /* [4*H] for accumulators */
    int input_size,
    int hidden_size,
    const SlstmS8Params* params);

/* Full sequence evaluation (INT8 quantized): batch + time loop.
 *
 * Processes input[B, T, I] and writes output[B, T, H] (all INT8).
 * State tensors: y[B,H] INT8, c[B,H] INT16, n[B,H] INT16, m[B,H] float.
 * Caller must provide a scratch buffer of at least 4*hidden_size int32_t. */
void slstm_eval_s8(
    const int8_t* input,      /* [B, T, I] */
    const int8_t* W_q,        /* [4*H, I] */
    const int8_t* R_q,        /* [4*H, H] */
    const int32_t* b_q,       /* [4*H] */
    int8_t* y,                /* [B, H] in/out */
    int16_t* c,               /* [B, H] in/out */
    int16_t* n,               /* [B, H] in/out */
    float* m,                 /* [B, H] in/out */
    int8_t* output,           /* [B, T, H] */
    int32_t* scratch,         /* [4*H] */
    int batch_size,
    int time_steps,
    int input_size,
    int hidden_size,
    const SlstmS8Params* params);

#ifdef __cplusplus
}
#endif

#endif /* SLSTM_S8_H_ */
