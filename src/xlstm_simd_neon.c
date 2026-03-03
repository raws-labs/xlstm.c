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
 * ARM NEON backend — testable via qemu-aarch64.
 * ===========================================================================*/

#include "xlstm_simd.h"
#include <arm_neon.h>

void xlstm_matvec_f32(const float* M, const float* v,
                      float* out, int rows, int cols)
{
    int i, j;
    int cols4 = cols & ~3;

    for (i = 0; i < rows; ++i) {
        float32x4_t acc = vdupq_n_f32(0.0f);
        const float* row = M + i * cols;

        for (j = 0; j < cols4; j += 4) {
            float32x4_t m = vld1q_f32(row + j);
            float32x4_t x = vld1q_f32(v + j);
            acc = vmlaq_f32(acc, m, x);
        }

        float s = vaddvq_f32(acc);
        for (; j < cols; ++j) {
            s += row[j] * v[j];
        }
        out[i] += s;
    }
}

void xlstm_matvec_s8(const int8_t* M, const int8_t* v,
                     int32_t* out, int rows, int cols, int32_t v_zp)
{
    int i, j;
    int cols8 = cols & ~7;
    int16x8_t vzp = vdupq_n_s16((int16_t)v_zp);

    for (i = 0; i < rows; ++i) {
        int32x4_t acc = vdupq_n_s32(0);
        const int8_t* row = M + i * cols;

        for (j = 0; j < cols8; j += 8) {
            /* Load 8 int8 values, widen to int16 */
            int8x8_t mr = vld1_s8(row + j);
            int8x8_t vr = vld1_s8(v + j);

            int16x8_t m16 = vmovl_s8(mr);
            int16x8_t v16 = vsubq_s16(vmovl_s8(vr), vzp);

            /* Multiply and accumulate: int16*int16 -> int32 */
            acc = vmlal_s16(acc, vget_low_s16(m16), vget_low_s16(v16));
            acc = vmlal_s16(acc, vget_high_s16(m16), vget_high_s16(v16));
        }

        int32_t s = vaddvq_s32(acc);
        for (; j < cols; ++j) {
            s += (int32_t)row[j] * ((int32_t)v[j] - v_zp);
        }
        out[i] = s;
    }
}

void xlstm_rank1_update_f32(float* C, float f_gate, float i_gate,
                            const float* k, const float* v, int H)
{
    int r, c;
    int H4 = H & ~3;
    float32x4_t vf = vdupq_n_f32(f_gate);

    for (r = 0; r < H; ++r) {
        float32x4_t vik = vdupq_n_f32(i_gate * k[r]);
        float* Crow = C + r * H;

        for (c = 0; c < H4; c += 4) {
            float32x4_t cv = vld1q_f32(Crow + c);
            float32x4_t vv = vld1q_f32(v + c);
            cv = vmlaq_f32(vmulq_f32(vf, cv), vik, vv);
            vst1q_f32(Crow + c, cv);
        }
        for (; c < H; ++c) {
            Crow[c] = f_gate * Crow[c] + i_gate * k[r] * v[c];
        }
    }
}

void xlstm_vecmat_f32(const float* q, const float* M,
                      float* out, int rows, int cols)
{
    int i, j;
    int cols4 = cols & ~3;

    for (i = 0; i < rows; ++i) {
        float32x4_t vq = vdupq_n_f32(q[i]);
        const float* Mrow = M + i * cols;

        for (j = 0; j < cols4; j += 4) {
            float32x4_t mv = vld1q_f32(Mrow + j);
            float32x4_t ov = vld1q_f32(out + j);
            ov = vmlaq_f32(ov, vq, mv);
            vst1q_f32(out + j, ov);
        }
        for (; j < cols; ++j) {
            out[j] += q[i] * Mrow[j];
        }
    }
}

const char* xlstm_simd_backend(void)
{
    return "neon";
}
