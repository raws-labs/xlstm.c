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
 * ARM NEON backend - testable via qemu-aarch64.
 * ===========================================================================*/

#include "xlstm_simd.h"
#include <arm_neon.h>

/* Which body a call actually ran, for test/simd_gate.cc.
 *
 * Nothing here dispatches - the vector loop's own bound decides - so a build
 * can link this backend, reproduce every golden vector, and execute no vector
 * instruction at all with nothing able to see it. That makes it exactly the
 * kind of silent loss of coverage a gate has to be able to fail on. The flags
 * below are therefore set FROM the loop's own cursor AFTER the loop, never
 * re-derived from cols before it: a guard added around a vector body would
 * have to move the count with it.
 *
 * Three outcomes per kernel, and the third is the opposite of the helium
 * backend's. `tail` counts calls that left work OUTSIDE the vector body for a
 * scalar remainder, because Advanced SIMD has no predication and a width that
 * is not a multiple of the vector's cannot stay in the vector loop; helium
 * counts the narrowed pass that keeps it there.
 *
 * Off unless XLSTM_NEON_FASTPATH_COUNTERS is defined - the Makefile sets it
 * for the object the tests link and for nothing else - so a shipping kernel
 * carries neither the counters nor the flags. */
#ifdef XLSTM_NEON_FASTPATH_COUNTERS
unsigned long xlstm_neon_matvec_f32_vector = 0;
unsigned long xlstm_neon_matvec_f32_scalar = 0;
unsigned long xlstm_neon_matvec_f32_tail = 0;
unsigned long xlstm_neon_matvec_s8_vector = 0;
unsigned long xlstm_neon_matvec_s8_scalar = 0;
unsigned long xlstm_neon_matvec_s8_tail = 0;
unsigned long xlstm_neon_rank1_f32_vector = 0;
unsigned long xlstm_neon_rank1_f32_scalar = 0;
unsigned long xlstm_neon_rank1_f32_tail = 0;
unsigned long xlstm_neon_vecmat_f32_vector = 0;
unsigned long xlstm_neon_vecmat_f32_scalar = 0;
unsigned long xlstm_neon_vecmat_f32_tail = 0;
#define XLSTM_NEON_FLAGS int seen_vec_ = 0, seen_tail_ = 0
#define XLSTM_NEON_SEEN(vec, tail)                                    \
    ((void)(seen_vec_ |= (vec) != 0, seen_tail_ |= (tail) != 0))
#define XLSTM_NEON_COUNT(name)                                        \
    ((void)(seen_vec_ ? ++xlstm_neon_##name##_vector                  \
                      : ++xlstm_neon_##name##_scalar),                \
     (void)(seen_tail_ ? ++xlstm_neon_##name##_tail : 0ul))
#else
#define XLSTM_NEON_FLAGS ((void)0)
#define XLSTM_NEON_SEEN(vec, tail) ((void)0)
#define XLSTM_NEON_COUNT(name) ((void)0)
#endif

void xlstm_matvec_f32(const float* M, const float* v,
                      float* out, int rows, int cols)
{
    int i, j;
    int cols4 = cols & ~3;
    XLSTM_NEON_FLAGS;

    for (i = 0; i < rows; ++i) {
        float32x4_t acc = vdupq_n_f32(0.0f);
        const float* row = M + i * cols;

        for (j = 0; j < cols4; j += 4) {
            float32x4_t m = vld1q_f32(row + j);
            float32x4_t x = vld1q_f32(v + j);
            acc = vmlaq_f32(acc, m, x);
        }
        /* j has advanced iff a vector pass ran; short of cols is remainder. */
        XLSTM_NEON_SEEN(j > 0, j < cols);

        float s = vaddvq_f32(acc);
        for (; j < cols; ++j) {
            s += row[j] * v[j];
        }
        out[i] += s;
    }
    XLSTM_NEON_COUNT(matvec_f32);
}

void xlstm_matvec_s8(const int8_t* M, const int8_t* v,
                     int32_t* out, int rows, int cols, int32_t v_zp)
{
    int i, j;
    int cols8 = cols & ~7;
    /* The zero point is subtracted in int16, so this is bit-exact against the
     * scalar body for every |v_zp| <= 32640 and no further: at that point
     * v[j] - v_zp stops fitting an int16 lane for some int8 v[j] and the lane
     * wraps where the scalar body's int32 does not. Every caller passes an
     * int8 zero point, three orders inside that. */
    int16x8_t vzp = vdupq_n_s16((int16_t)v_zp);
    XLSTM_NEON_FLAGS;

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
        /* j has advanced iff a vector pass ran; short of cols is remainder. */
        XLSTM_NEON_SEEN(j > 0, j < cols);

        int32_t s = vaddvq_s32(acc);
        for (; j < cols; ++j) {
            s += (int32_t)row[j] * ((int32_t)v[j] - v_zp);
        }
        out[i] = s;
    }
    XLSTM_NEON_COUNT(matvec_s8);
}

void xlstm_rank1_update_f32(float* C, float f_gate, float i_gate,
                            const float* k, const float* v, int H)
{
    int r, c;
    int H4 = H & ~3;
    float32x4_t vf = vdupq_n_f32(f_gate);
    XLSTM_NEON_FLAGS;

    for (r = 0; r < H; ++r) {
        float32x4_t vik = vdupq_n_f32(i_gate * k[r]);
        float* Crow = C + r * H;

        for (c = 0; c < H4; c += 4) {
            float32x4_t cv = vld1q_f32(Crow + c);
            float32x4_t vv = vld1q_f32(v + c);
            cv = vmlaq_f32(vmulq_f32(vf, cv), vik, vv);
            vst1q_f32(Crow + c, cv);
        }
        /* c has advanced iff a vector pass ran; short of H is remainder. */
        XLSTM_NEON_SEEN(c > 0, c < H);

        for (; c < H; ++c) {
            Crow[c] = f_gate * Crow[c] + i_gate * k[r] * v[c];
        }
    }
    XLSTM_NEON_COUNT(rank1_f32);
}

void xlstm_vecmat_f32(const float* q, const float* M,
                      float* out, int rows, int cols)
{
    int i, j;
    int cols4 = cols & ~3;
    XLSTM_NEON_FLAGS;

    for (i = 0; i < rows; ++i) {
        float32x4_t vq = vdupq_n_f32(q[i]);
        const float* Mrow = M + i * cols;

        for (j = 0; j < cols4; j += 4) {
            float32x4_t mv = vld1q_f32(Mrow + j);
            float32x4_t ov = vld1q_f32(out + j);
            ov = vmlaq_f32(ov, vq, mv);
            vst1q_f32(out + j, ov);
        }
        /* j has advanced iff a vector pass ran; short of cols is remainder. */
        XLSTM_NEON_SEEN(j > 0, j < cols);

        for (; j < cols; ++j) {
            out[j] += q[i] * Mrow[j];
        }
    }
    XLSTM_NEON_COUNT(vecmat_f32);
}

const char* xlstm_simd_backend(void)
{
    return "neon";
}
