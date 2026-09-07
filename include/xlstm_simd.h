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
 * SIMD dispatch layer for xLSTM compute primitives.
 *
 * Backend selected at compile time via XLSTM_SIMD={auto|ref|sse2|neon|esp}.
 * Each backend implements these functions in its own .c file.
 * ===========================================================================*/

#ifndef XLSTM_SIMD_H_
#define XLSTM_SIMD_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Max hidden size for stack-allocated temporaries (avoids VLA on MCU). */
#ifndef XLSTM_MAX_HIDDEN
#define XLSTM_MAX_HIDDEN 256
#endif

/* f32 matrix-vector multiply-accumulate.
 * out[i] += sum_j M[i*cols+j] * v[j]   for i in [0, rows)
 * Caller must pre-fill out (e.g. out[i] = bias[i]). */
void xlstm_matvec_f32(const float* M, const float* v,
                      float* out, int rows, int cols);

/* INT8 matrix-vector multiply with zero-point subtraction.
 * out[i] = sum_j M[i*cols+j] * (v[j] - v_zp)   for i in [0, rows)
 * Overwrites out (caller manages separate accumulators). */
void xlstm_matvec_s8(const int8_t* M, const int8_t* v,
                     int32_t* out, int rows, int cols, int32_t v_zp);

/* Rank-1 update for mLSTM cell matrix (row-major rows x cols).
 * C[r*cols+c] = f_gate * C[r*cols+c] + i_gate * k[r] * v[c]
 *
 * rows is the q/k width and cols the v width; k is [rows], v is [cols]. They
 * are equal for a square cell, which is every case this library shipped
 * before the two were split, and the square path is bit-identical to it. */
void xlstm_rank1_update_f32(float* C, float f_gate, float i_gate,
                            const float* k, const float* v, int rows, int cols);

/* Left-multiply (vec * mat) for mLSTM output: out[j] = sum_i q[i] * M[i*cols+j]
 * M is row-major [rows x cols]. out must be zeroed by caller. */
void xlstm_vecmat_f32(const float* q, const float* M,
                      float* out, int rows, int cols);

/* Returns the name of the active SIMD backend. */
const char* xlstm_simd_backend(void);

#ifdef __cplusplus
}
#endif

#endif /* XLSTM_SIMD_H_ */
