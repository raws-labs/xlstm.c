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
 * ESP32-S3 backend - uses ESP-DSP dot product and PIE int8 MAC.
 * Only compiles under ESP-IDF toolchain.
 * ===========================================================================*/

#include "xlstm_simd.h"
#include "dsps_dotprod.h"

#include <stdint.h>

/* ESP-DSP's optimized dot products load 128 bits at a time
 * (EE.LDF.128.IP src, 16) and consume four floats per iteration, so both
 * operands must be 16-byte aligned and the length must be a multiple of 4.
 * Nothing in the ESP-DSP headers states this - it is visible only in the
 * .S sources - and violating it does not fault, it silently returns wrong
 * numbers.
 *
 * xlstm_matvec_f32 walks rows as M + i*cols, so row alignment depends on
 * both the base pointer and cols. Rather than assume, check, and fall back
 * to ESP-DSP's own ANSI implementation when the preconditions do not hold.
 * dsps_dotprod_f32_ansi is plain C with no alignment or length constraint.
 *
 * Note also that despite the header's doc comment claiming
 * "*dest += src1[i]*src2[i]", every implementation ASSIGNS (*dest = acc;
 * `ssi f0, a4, 0` in the assembly). The uninitialized `dot` below is
 * therefore correct - do not "fix" it by zeroing, and do not assume the
 * accumulate semantics the comment advertises.
 */
#define XLSTM_ESP_DSP_ALIGN 16

static inline int xlstm_esp_aligned(const void* p)
{
    return (((uintptr_t)p) & (XLSTM_ESP_DSP_ALIGN - 1)) == 0;
}

void xlstm_matvec_f32(const float* M, const float* v,
                      float* out, int rows, int cols)
{
    /* Hoisted: if cols is a multiple of 4 then every row shares the base
     * pointer's alignment, so this is decided once rather than per row. */
    const int fast = (cols % 4) == 0 && xlstm_esp_aligned(M) && xlstm_esp_aligned(v);
    int i;

    for (i = 0; i < rows; ++i) {
        if (fast) {
            /* Chip-generic macro: dispatches to _aes3 on ESP32-S3, _ae32 on
             * ESP32, _ansi elsewhere. The previous code hardcoded _ae32, the
             * ESP32 variant, which compiles on S3 (its guard is generic Xtensa
             * capability flags, not a chip check) but leaves the S3-optimized
             * path unused. */
            float dot;
            dsps_dotprod_f32(M + i * cols, v, &dot, cols);
            out[i] += dot;
        } else {
            /* Deliberately NOT dsps_dotprod_f32_ansi. That computes the dot
             * into a fresh accumulator and leaves the caller to do
             * out[i] += dot, whereas xlstm_simd_ref.c seeds one running
             * accumulator from out[i]. The mathematics is identical but the
             * float grouping is not, and on a cancellation-sensitive case it
             * moves the result: mLSTM SweepM17 y[0] came out 4.20e-05 from the
             * golden value against a ~3.98e-05 bound, failing the gate on the
             * esp backend's first end-to-end run.
             *
             * The fallback is plain C either way, so there is nothing to gain
             * from ESP-DSP here and a real divergence to lose. Match ref
             * exactly - it is the baseline every other backend is defined
             * against. */
            int j;
            float acc = out[i];
            for (j = 0; j < cols; ++j) {
                acc += M[i * cols + j] * v[j];
            }
            out[i] = acc;
        }
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
