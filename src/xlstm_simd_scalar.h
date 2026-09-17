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
 * The scalar C99 kernels, as a single text.
 *
 * Not a translation unit: included by xlstm_simd_ref.c, which is nothing
 * but these six plus a name, and by any backend that accelerates some of
 * the contract and defers the rest. A partial backend that copied these
 * bodies instead would be a second reference free to drift from the first;
 * sharing the text means a fix lands in every backend that defers to it.
 *
 * Every backend is defined against ref, so these bodies are also the
 * definition of bit-exactness - in particular the f32 accumulation order
 * (one running accumulator seeded from out[i], summed in ascending j).
 * ===========================================================================*/

#ifndef XLSTM_SIMD_SCALAR_INC_
#define XLSTM_SIMD_SCALAR_INC_

#include "xlstm_util.h"

#include <stdint.h>

static inline void xlstm_scalar_matvec_f32(const float* M, const float* v,
                                           float* out, int rows, int cols)
{
    int i, j;
    for (i = 0; i < rows; ++i) {
        float acc = out[i];
        for (j = 0; j < cols; ++j) {
            acc += M[i * cols + j] * v[j];
        }
        out[i] = acc;
    }
}

static inline void xlstm_scalar_matvec_s8(const int8_t* M, const int8_t* v,
                                          int32_t* out, int rows, int cols,
                                          int32_t v_zp)
{
    int i, j;
    for (i = 0; i < rows; ++i) {
        int32_t acc = 0;
        for (j = 0; j < cols; ++j) {
            acc += (int32_t)M[i * cols + j] * ((int32_t)v[j] - v_zp);
        }
        out[i] = acc;
    }
}

static inline void xlstm_scalar_rank1_update_f32(float* C, float f_gate,
                                                 float i_gate, const float* k,
                                                 const float* v, int rows,
                                                 int cols)
{
    int r, c;
    for (r = 0; r < rows; ++r) {
        float ik_r = i_gate * k[r];
        for (c = 0; c < cols; ++c) {
            C[r * cols + c] = f_gate * C[r * cols + c] + ik_r * v[c];
        }
    }
}

static inline void xlstm_scalar_vecmat_f32(const float* q, const float* M,
                                           float* out, int rows, int cols)
{
    int i, j;
    for (i = 0; i < rows; ++i) {
        float qi = q[i];
        for (j = 0; j < cols; ++j) {
            out[j] += qi * M[i * cols + j];
        }
    }
}

/* The INT16 state pair. Both were loop bodies inside mlstm_step_s8 before the
 * contract grew to reach them, and both are written to be the same arithmetic
 * in the same order, because test/mlstm_s8_test.cc compares the exit state and
 * the output as integer codes, not as dequantized floats. */

static inline void xlstm_scalar_rank1_update_s16(int16_t* C, float f_gate,
                                                 float i_gate, const float* k,
                                                 const float* v, float scale,
                                                 float cell_clip, int rows,
                                                 int cols)
{
    int r, c;
    if (cell_clip > 0.0f) {
        for (r = 0; r < rows; ++r) {
            float ik_r = i_gate * k[r];
            int16_t* Crow = C + r * cols;
            for (c = 0; c < cols; ++c) {
                float C_new = f_gate * ((float)Crow[c] * scale) + ik_r * v[c];
                C_new = xlstm_maxf(-cell_clip, xlstm_minf(cell_clip, C_new));
                Crow[c] = (int16_t)xlstm_round_clamp_i32(C_new / scale,
                                                        -32768.0f, 32767.0f);
            }
        }
    } else {
        for (r = 0; r < rows; ++r) {
            float ik_r = i_gate * k[r];
            int16_t* Crow = C + r * cols;
            for (c = 0; c < cols; ++c) {
                float C_new = f_gate * ((float)Crow[c] * scale) + ik_r * v[c];
                Crow[c] = (int16_t)xlstm_round_clamp_i32(C_new / scale,
                                                        -32768.0f, 32767.0f);
            }
        }
    }
}

static inline void xlstm_scalar_vecmat_s16(const float* q, const int16_t* M,
                                           float* out, float scale, int rows,
                                           int cols)
{
    int i, j;
    for (i = 0; i < rows; ++i) {
        float qi = q[i];
        const int16_t* Mrow = M + i * cols;
        for (j = 0; j < cols; ++j) {
            out[j] += qi * ((float)Mrow[j] * scale);
        }
    }
}

#endif /* XLSTM_SIMD_SCALAR_INC_ */
