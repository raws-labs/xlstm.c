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
 * Cortex-M4/M7/M33 backend - ARMv7E-M / ARMv8-M Mainline DSP extension.
 *
 * Two of the four kernels are accelerated:
 *
 *   xlstm_matvec_s8   SXTAB16 + SMLAD, two rows at a time. 16 instructions
 *                     per 8 MACs against the scalar body's 6 per MAC, at any
 *                     buffer alignment and any column count.
 *   xlstm_matvec_f32  two rows at a time with fmaf, which these FPUs issue
 *                     as a single VFMA.F32.
 *
 * xlstm_vecmat_f32 and xlstm_rank1_update_f32 defer to the scalar bodies in
 * xlstm_simd_scalar.inc - the same text xlstm_simd_ref.c compiles, not a
 * copy of it. There is no f32 SIMD on these cores, and rank-1 update is
 * bandwidth-bound rather than issue-bound, so an honest measurement is all
 * either would produce.
 *
 * Dependencies: ACLE (arm_acle.h ships with the compiler) plus three lines
 * of inline asm. Deliberately not CMSIS-NN: it would accelerate exactly one
 * of these kernels at the cost of the library's dependency-free property.
 * ===========================================================================*/

#include "xlstm_simd.h"

#include "xlstm_simd_scalar.inc"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifndef __ARM_FEATURE_SIMD32
#error "XLSTM_SIMD=cortexm needs the DSP extension: build with -mcpu=cortex-m4," \
       " cortex-m7 or cortex-m33 (plus -mthumb). Cortex-M0/M0+/M23 have no DSP" \
       " extension at all - use XLSTM_SIMD=ref there."
#endif

#include <arm_acle.h>

/* GCC will not fuse the rotate into SXTB16/SXTAB16 when the extension comes
 * from an ACLE intrinsic, and does not declare ACLE's __ror at all, so the
 * odd-byte lanes would each cost a separate ROR - about a sixth of the
 * inner loop below. Inline asm is not a dependency; these are pure register
 * operations, so the compiler may still schedule and CSE them freely. */
static inline int16x2_t xlstm_cm_sxtb16_ror8(int8x4_t x)
{
    int16x2_t r;
    __asm__("sxtb16 %0, %1, ror #8" : "=r"(r) : "r"(x));
    return r;
}

static inline int16x2_t xlstm_cm_sxtab16_ror8(int16x2_t a, int8x4_t x)
{
    int16x2_t r;
    __asm__("sxtab16 %0, %1, %2, ror #8" : "=r"(r) : "r"(a), "r"(x));
    return r;
}

/* One 4-byte group of a caller buffer, at whatever alignment it has. The #if
 * is decided at compile time, so the loop below carries no branch per load.
 *
 * Where unaligned access is permitted this memcpy is one LDR - the same
 * instruction, in the same addressing modes, that a word-typed load of a
 * known-aligned pointer produces. So there is nothing an alignment-specialised
 * second instance of the loop could win, and on a part whose instruction cache
 * is around 1 KB two instances measurably lose: at H=64 the aligned path ran
 * 10% slower on a Cortex-M4 with the same instruction count, purely from
 * cache pressure, while a Cortex-M7 with 16 KB gained. One body is both the
 * smaller and the faster answer, and this is where it is decided.
 *
 * Under -mno-unaligned-access the memcpy becomes a call to memcpy instead, so
 * that configuration gets an explicit byte assembly - which GCC folds straight
 * back into an unaligned LDR unless that same flag forbids it, hence the #if
 * rather than always spelling it this way. It costs about 4.5 instructions per
 * MAC against 2, on aligned data as well as unaligned, and is still under the
 * scalar body's 6. Firmware that sets CCR.UNALIGN_TRP (the recommended debug
 * setting, and a UsageFault on any unaligned LDR) builds with the flag - as it
 * must for any C in the image - and so issues no unaligned access at all.
 * Picking the load form at run time would recover the aligned case there, but
 * only by testing on every group load of both cases, in the configuration that
 * is already the slow one; the branchless byte assembly is the better trade.
 *
 * Byte order is not chosen here: all three groups of a pass are read the same
 * way, so any permutation cancels between the lanes SMLAD pairs up. */
static inline int8x4_t xlstm_cm_ld4(const int8_t* p)
{
    uint32_t w;
#ifdef __ARM_FEATURE_UNALIGNED
    memcpy(&w, p, sizeof w);
#else
    w = (uint32_t)(uint8_t)p[0] | ((uint32_t)(uint8_t)p[1] << 8) |
        ((uint32_t)(uint8_t)p[2] << 16) | ((uint32_t)(uint8_t)p[3] << 24);
#endif
    return (int8x4_t)w;
}

/* SXTAB16 folds -v_zp into the widening at no cost, which is why this
 * backend needs no row sums: it widens byte lanes to halfwords AND adds a
 * packed offset in the same instruction. Its halfword adds wrap rather than
 * saturate, so the fold is exact only while v[j] - v_zp fits int16, i.e.
 * |v_zp| <= 32640 for v[j] in [-128, 127]. Every caller passes an int8 zero
 * point; the bound is checked so that the kernel is bit-exact against ref
 * for any int32 the signature permits, not just the ones in practice.
 *
 * Accumulator headroom: in that int8 domain |M[j] * (v[j] - v_zp)| is at
 * most 127 * 255 = 32385, so an int32 accumulator overflows only past
 * ~66000 columns - unreachable at XLSTM_MAX_HIDDEN = 256, and equally an
 * overflow of ref's own int32 accumulator if it ever were reached. */
#define XLSTM_CM_ZP_MAX 32640

void xlstm_matvec_s8(const int8_t* M, const int8_t* v,
                     int32_t* out, int rows, int cols, int32_t v_zp)
{
    /* Two things still leave the DSP body: a zero point SXTAB16 cannot fold
     * exactly, and a cols of 0 or less, for which a row is not a range of
     * addresses at all and the loops below would step past out entirely.
     * Neither is alignment, and neither is cols % 4. Co-alignment of a row
     * against v is not obtainable when cols % 4 != 0 - each successive row
     * start shifts by another byte - so a path that insisted on aligned words
     * would abandon the DSP body on most rows of every odd hidden size, H = 17
     * among them, which is where the whole of the work is. xlstm_cm_ld4 serves
     * every group of every row instead, at whatever alignment it lands on. */
    if (cols <= 0 || v_zp > XLSTM_CM_ZP_MAX || v_zp < -XLSTM_CM_ZP_MAX) {
        xlstm_scalar_matvec_s8(M, v, out, rows, cols, v_zp);
        return;
    }

    {
        /* Past the last whole group; the 0 to 3 columns after it go scalar.
         * Walking pointers to it rather than indexing off a group counter is
         * what keeps the loop test folded into the address arithmetic. tail
         * comes from cols, so a multiple of 4 skips the remainder loops at
         * run time rather than by having been compiled without them. */
        const int8_t* const vend = v + ((size_t)cols & ~(size_t)3);
        const int tail = cols & 3;
        /* -v_zp in both halfword lanes, the addend SXTAB16 applies. */
        const int16x2_t nzp = (int16x2_t)(((uint32_t)(-v_zp) & 0xFFFFu) |
                                          ((uint32_t)(-v_zp) << 16));
        /* Both row loops run to a pointer rather than counting i: that leaves
         * one fewer value live across the inner loop, which is what keeps the
         * row epilogue off the stack. nrows folds a non-positive rows into an
         * empty range instead of a negative pointer offset. */
        const size_t stride = (size_t)cols;
        const size_t nrows = rows > 0 ? (size_t)rows : 0u;
        const int8_t* const rowend2 = M + (nrows & ~(size_t)1) * stride;
        const int8_t* const rowend = M + nrows * stride;
        const int8_t* row = M;
        int32_t* o = out;
        int j;

        /* Two rows per pass: the widened v lanes are computed once and
         * consumed twice, and the two accumulator chains are independent,
         * so neither stalls on the other's SMLAD result. */
        for (; row != rowend2; row += 2 * stride, o += 2) {
            const int8_t* p0 = row;
            const int8_t* p1 = row + stride;
            const int8_t* vp = v;
            int32_t a0 = 0;
            int32_t a1 = 0;

            for (; vp != vend; vp += 4, p0 += 4, p1 += 4) {
                int8x4_t w = xlstm_cm_ld4(vp);
                int16x2_t ve = __sxtab16(nzp, w);       /* v[4g+0], v[4g+2] */
                int16x2_t vo = xlstm_cm_sxtab16_ror8(nzp, w); /* +1, +3 */
                int8x4_t m0 = xlstm_cm_ld4(p0);
                int8x4_t m1 = xlstm_cm_ld4(p1);

                a0 = __smlad(__sxtb16(m0), ve, a0);
                a0 = __smlad(xlstm_cm_sxtb16_ror8(m0), vo, a0);
                a1 = __smlad(__sxtb16(m1), ve, a1);
                a1 = __smlad(xlstm_cm_sxtb16_ror8(m1), vo, a1);
            }
            /* All three pointers now sit on the first leftover column. */
            for (j = 0; j < tail; ++j) {
                a0 += (int32_t)p0[j] * ((int32_t)vp[j] - v_zp);
                a1 += (int32_t)p1[j] * ((int32_t)vp[j] - v_zp);
            }
            o[0] = a0;
            o[1] = a1;
        }

        for (; row != rowend; row += stride, ++o) {
            const int8_t* p0 = row;
            const int8_t* vp = v;
            int32_t a0 = 0;

            for (; vp != vend; vp += 4, p0 += 4) {
                int8x4_t w = xlstm_cm_ld4(vp);
                int8x4_t m0 = xlstm_cm_ld4(p0);

                a0 = __smlad(__sxtb16(m0), __sxtab16(nzp, w), a0);
                a0 = __smlad(xlstm_cm_sxtb16_ror8(m0),
                             xlstm_cm_sxtab16_ror8(nzp, w), a0);
            }
            for (j = 0; j < tail; ++j) {
                a0 += (int32_t)p0[j] * ((int32_t)vp[j] - v_zp);
            }
            *o = a0;
        }
    }
}

void xlstm_matvec_f32(const float* M, const float* v,
                      float* out, int rows, int cols)
{
    /* Same two-row shape, for the same reason: v[j] is loaded once for two
     * independent VFMA chains, which is what hides the FPU's multi-cycle
     * accumulate latency. fmaf is one VFMA.F32 on FPv4-SP and FPv5, so this
     * rounds once where the scalar body rounds twice. That is a deliberate
     * numeric difference from ref, and it is the source of the win; the
     * per-row summation order is otherwise identical. */
    int i = 0;
    int j;

    for (; i + 1 < rows; i += 2) {
        const float* r0 = M + (size_t)i * (size_t)cols;
        const float* r1 = r0 + cols;
        float a0 = out[i];
        float a1 = out[i + 1];

        for (j = 0; j < cols; ++j) {
            float x = v[j];
            a0 = fmaf(r0[j], x, a0);
            a1 = fmaf(r1[j], x, a1);
        }
        out[i] = a0;
        out[i + 1] = a1;
    }

    for (; i < rows; ++i) {
        const float* r0 = M + (size_t)i * (size_t)cols;
        float a0 = out[i];

        for (j = 0; j < cols; ++j) {
            a0 = fmaf(r0[j], v[j], a0);
        }
        out[i] = a0;
    }
}

void xlstm_rank1_update_f32(float* C, float f_gate, float i_gate,
                            const float* k, const float* v, int H)
{
    xlstm_scalar_rank1_update_f32(C, f_gate, i_gate, k, v, H);
}

void xlstm_vecmat_f32(const float* q, const float* M,
                      float* out, int rows, int cols)
{
    xlstm_scalar_vecmat_f32(q, M, out, rows, cols);
}

const char* xlstm_simd_backend(void)
{
    return "cortexm";
}
