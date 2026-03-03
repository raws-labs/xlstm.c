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
 * ESP32-S3 backend — uses ESP-DSP dot product and PIE int8 MAC.
 * Only compiles under ESP-IDF toolchain.
 * ===========================================================================*/

#include "xlstm_simd.h"
#include "dsps_dotprod.h"

void xlstm_matvec_f32(const float* M, const float* v,
                      float* out, int rows, int cols)
{
    int i;
    for (i = 0; i < rows; ++i) {
        float dot;
        dsps_dotprod_f32_ae32(M + i * cols, v, &dot, cols);
        out[i] += dot;
    }
}

void xlstm_matvec_s8(const int8_t* M, const int8_t* v,
                     int32_t* out, int rows, int cols, int32_t v_zp)
{
    /* TODO: PIE EE.VMULAS.S8.ACCX for 16-way int8 MAC.
     * Scalar fallback until PIE intrinsics are available. */
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
        /* Row-wise: use dsps_dotprod for read, but update is in-place.
         * Element-wise for now. */
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
    return "esp";
}
