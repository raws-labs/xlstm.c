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
 * Backend selected at compile time via
 * XLSTM_SIMD={auto|ref|sse2|neon|esp|cortexm|helium}. Each backend implements
 * these functions in its own .c file.
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

/* Left-multiply (vec * mat) for mLSTM output: out[j] += sum_i q[i]*M[i*cols+j]
 * M is row-major [rows x cols]. Caller must pre-fill out. */
void xlstm_vecmat_f32(const float* q, const float* M,
                      float* out, int rows, int cols);

/* The two above, for an INT16 cell matrix held on a symmetric scale.
 *
 * They exist because the INT8 mLSTM's O(rows*cols) state work was the only
 * part of either cell that no backend could reach: mlstm_step_s8 spelled both
 * loops inline, so selecting a backend changed the f32 step and left the INT8
 * step instruction for instruction identical. Measured at rows = cols = 64,
 * callgrind Ir, 200 steps, ref against sse2: the f32 step went 35.88M to
 * 11.84M while the INT8 step body stayed at 33,153,800 on both.
 *
 * Each dequantizes with `(float)C[i] * scale`, works in float, and the update
 * requantizes with `x / scale` and round-half-away-from-zero. The divide is
 * not a reciprocal multiply: see the note in src/mlstm_s8.c.
 *
 * cell_clip <= 0 disables clipping. The two spellings are separate loops
 * rather than a test per element, so the unclipped path - the one every
 * existing caller takes - is exactly the arithmetic it was. */
void xlstm_rank1_update_s16(int16_t* C, float f_gate, float i_gate,
                            const float* k, const float* v, float scale,
                            float cell_clip, int rows, int cols);

/* out[j] += sum_i q[i] * ((float)M[i*cols+j] * scale). Caller pre-fills out.
 * The sum over i runs in ascending order, one running accumulator per j, which
 * is the accumulation order mlstm_step_s8 had when it spelled this inline. */
void xlstm_vecmat_s16(const float* q, const int16_t* M, float* out,
                      float scale, int rows, int cols);

/* Returns the name of the active SIMD backend. */
const char* xlstm_simd_backend(void);

/* Returns "exact" or "approx": which of the two transcendental
 * implementations this build compiled, the XLSTM_GATES setting that produced
 * it. A binary could already name its SIMD backend and not its numerics,
 * although the two builds differ on every pair in test/perf_baseline.txt. */
const char* xlstm_gate_build(void);

#ifdef __cplusplus
}
#endif

#endif /* XLSTM_SIMD_H_ */
