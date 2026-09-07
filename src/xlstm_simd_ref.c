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
 * Scalar C99 reference backend - portable fallback for all platforms.
 *
 * The kernels themselves live in xlstm_simd_scalar.h so that a partial
 * backend can defer to this exact text rather than copy it.
 * ===========================================================================*/

#include "xlstm_simd.h"

#include "xlstm_simd_scalar.h"

void xlstm_matvec_f32(const float* M, const float* v,
                      float* out, int rows, int cols)
{
    xlstm_scalar_matvec_f32(M, v, out, rows, cols);
}

void xlstm_matvec_s8(const int8_t* M, const int8_t* v,
                     int32_t* out, int rows, int cols, int32_t v_zp)
{
    xlstm_scalar_matvec_s8(M, v, out, rows, cols, v_zp);
}

void xlstm_rank1_update_f32(float* C, float f_gate, float i_gate,
                            const float* k, const float* v, int rows, int cols)
{
    xlstm_scalar_rank1_update_f32(C, f_gate, i_gate, k, v, rows, cols);
}

void xlstm_vecmat_f32(const float* q, const float* M,
                      float* out, int rows, int cols)
{
    xlstm_scalar_vecmat_f32(q, M, out, rows, cols);
}

const char* xlstm_simd_backend(void)
{
    return "ref";
}
