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
 *                     per 8 MACs against the scalar body's 6 per MAC.
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

/* A 4-byte group read as one word. may_alias because the caller's buffers
 * are int8_t; the alignment of the type is what makes this a plain LDR
 * rather than a byte assembly, so it may only be used once the guard in
 * xlstm_matvec_s8 has established that the pointer really is word-aligned. */
typedef uint32_t xlstm_cm_word __attribute__((__may_alias__));

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
    /* The DSP path never issues an unaligned access rather than relying on
     * the core to forgive one: these cores do support unaligned LDR, but
     * only while CCR.UNALIGN_TRP is clear, and firmware that sets it (the
     * recommended debug setting) turns a group load into a UsageFault. The
     * usual mitigation, assembling the word from bytes, holds only if the
     * whole application is also built -mno-unaligned-access - otherwise
     * GCC's load merging folds that assembly straight back into one
     * unaligned LDR. A library cannot assert a flag on its caller's command
     * line, so this checks the pointers instead.
     *
     * v word-aligned aligns every group of v; M word-aligned with cols a
     * multiple of 4 makes every row start M + i*cols inherit that alignment
     * and leaves no ragged tail group. Anything else - a misaligned buffer,
     * or a cols that puts successive rows on different alignments - takes
     * the scalar body, which is exact by construction. */
    const size_t groups = (size_t)cols / 4u;
    const int aligned = ((((uintptr_t)M | (uintptr_t)v) & 3u) == 0u) &&
                        (cols % 4) == 0 && cols >= 0;

    if (!aligned || v_zp > XLSTM_CM_ZP_MAX || v_zp < -XLSTM_CM_ZP_MAX) {
        xlstm_scalar_matvec_s8(M, v, out, rows, cols, v_zp);
        return;
    }

    {
        const xlstm_cm_word* vw = (const xlstm_cm_word*)(const void*)v;
        const xlstm_cm_word* mw = (const xlstm_cm_word*)(const void*)M;
        /* -v_zp in both halfword lanes, the addend SXTAB16 applies. */
        const int16x2_t nzp = (int16x2_t)(((uint32_t)(-v_zp) & 0xFFFFu) |
                                          ((uint32_t)(-v_zp) << 16));
        int i = 0;
        size_t g;

        /* Two rows per pass: the widened v lanes are computed once and
         * consumed twice, and the two accumulator chains are independent,
         * so neither stalls on the other's SMLAD result. */
        for (; i + 1 < rows; i += 2) {
            const xlstm_cm_word* r0 = mw + (size_t)i * groups;
            const xlstm_cm_word* r1 = r0 + groups;
            int32_t a0 = 0;
            int32_t a1 = 0;

            for (g = 0; g < groups; ++g) {
                int8x4_t w = (int8x4_t)vw[g];
                int16x2_t ve = __sxtab16(nzp, w);          /* v[4g+0], v[4g+2] */
                int16x2_t vo = xlstm_cm_sxtab16_ror8(nzp, w); /* v[4g+1], v[4g+3] */
                int8x4_t m0 = (int8x4_t)r0[g];
                int8x4_t m1 = (int8x4_t)r1[g];

                a0 = __smlad(__sxtb16(m0), ve, a0);
                a0 = __smlad(xlstm_cm_sxtb16_ror8(m0), vo, a0);
                a1 = __smlad(__sxtb16(m1), ve, a1);
                a1 = __smlad(xlstm_cm_sxtb16_ror8(m1), vo, a1);
            }
            out[i] = a0;
            out[i + 1] = a1;
        }

        for (; i < rows; ++i) {
            const xlstm_cm_word* r0 = mw + (size_t)i * groups;
            int32_t a0 = 0;

            for (g = 0; g < groups; ++g) {
                int8x4_t w = (int8x4_t)vw[g];
                int8x4_t m0 = (int8x4_t)r0[g];

                a0 = __smlad(__sxtb16(m0), __sxtab16(nzp, w), a0);
                a0 = __smlad(xlstm_cm_sxtb16_ror8(m0),
                             xlstm_cm_sxtab16_ror8(nzp, w), a0);
            }
            out[i] = a0;
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
