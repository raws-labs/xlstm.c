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
 * SSE2 backend for x86/x86-64.
 * ===========================================================================*/

#include "xlstm_simd.h"
#include <emmintrin.h> /* SSE2 */

/* Horizontal sum of 4 floats in an __m128. */
static inline float hsum_ps(__m128 v)
{
    __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

void xlstm_matvec_f32(const float* M, const float* v,
                      float* out, int rows, int cols)
{
    int i, j;
    int cols4 = cols & ~3;

    for (i = 0; i < rows; ++i) {
        __m128 acc = _mm_setzero_ps();
        const float* row = M + i * cols;

        for (j = 0; j < cols4; j += 4) {
            __m128 m = _mm_loadu_ps(row + j);
            __m128 x = _mm_loadu_ps(v + j);
            acc = _mm_add_ps(acc, _mm_mul_ps(m, x));
        }

        float s = hsum_ps(acc);
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

    /* SSE2 s8 dot: widen int8 to int16, use _mm_madd_epi16 for int16*int16->int32. */
    __m128i vzp = _mm_set1_epi16((int16_t)v_zp);

    for (i = 0; i < rows; ++i) {
        __m128i acc = _mm_setzero_si128();
        const int8_t* row = M + i * cols;

        for (j = 0; j < cols8; j += 8) {
            /* Load 8 bytes, sign-extend to 16-bit */
            __m128i mr = _mm_loadl_epi64((const __m128i*)(row + j));
            __m128i vr = _mm_loadl_epi64((const __m128i*)(v + j));

            /* Unpack int8 -> int16 (sign-extend) */
            __m128i m16 = _mm_srai_epi16(_mm_unpacklo_epi8(mr, mr), 8);
            __m128i v16 = _mm_srai_epi16(_mm_unpacklo_epi8(vr, vr), 8);

            /* Subtract zero point */
            v16 = _mm_sub_epi16(v16, vzp);

            /* int16*int16 -> int32 (adjacent pairs summed) */
            acc = _mm_add_epi32(acc, _mm_madd_epi16(m16, v16));
        }

        /* Horizontal sum of 4 int32 lanes */
        __m128i hi = _mm_shuffle_epi32(acc, _MM_SHUFFLE(1, 0, 3, 2));
        acc = _mm_add_epi32(acc, hi);
        hi = _mm_shuffle_epi32(acc, _MM_SHUFFLE(0, 1, 0, 1));
        acc = _mm_add_epi32(acc, hi);
        int32_t s = _mm_cvtsi128_si32(acc);

        /* Scalar tail */
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
    __m128 vf = _mm_set1_ps(f_gate);

    for (r = 0; r < H; ++r) {
        __m128 vik = _mm_set1_ps(i_gate * k[r]);
        float* Crow = C + r * H;

        for (c = 0; c < H4; c += 4) {
            __m128 cv = _mm_loadu_ps(Crow + c);
            __m128 vv = _mm_loadu_ps(v + c);
            cv = _mm_add_ps(_mm_mul_ps(vf, cv), _mm_mul_ps(vik, vv));
            _mm_storeu_ps(Crow + c, cv);
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
        __m128 vq = _mm_set1_ps(q[i]);
        const float* Mrow = M + i * cols;

        for (j = 0; j < cols4; j += 4) {
            __m128 mv = _mm_loadu_ps(Mrow + j);
            __m128 ov = _mm_loadu_ps(out + j);
            ov = _mm_add_ps(ov, _mm_mul_ps(vq, mv));
            _mm_storeu_ps(out + j, ov);
        }
        for (; j < cols; ++j) {
            out[j] += q[i] * Mrow[j];
        }
    }
}

const char* xlstm_simd_backend(void)
{
    return "sse2";
}
