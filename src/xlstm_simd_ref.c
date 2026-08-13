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
 * ===========================================================================*/

#include "xlstm_simd.h"

void xlstm_matvec_f32(const float* M, const float* v,
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

void xlstm_matvec_s8(const int8_t* M, const int8_t* v,
                     int32_t* out, int rows, int cols, int32_t v_zp)
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

void xlstm_rank1_update_f32(float* C, float f_gate, float i_gate,
                            const float* k, const float* v, int H)
{
    int r, c;
    for (r = 0; r < H; ++r) {
        float ik_r = i_gate * k[r];
        for (c = 0; c < H; ++c) {
            C[r * H + c] = f_gate * C[r * H + c] + ik_r * v[c];
        }
    }
}

void xlstm_vecmat_f32(const float* q, const float* M,
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

const char* xlstm_simd_backend(void)
{
    return "ref";
}
