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
 *   xlstm_matvec_f32  eight rows at a time with fmaf, which these FPUs issue
 *                     as a single VFMA.F32. 9 loads per 8 MACs against the
 *                     scalar body's 2, because v[j] is read once per block.
 *
 * xlstm_vecmat_f32 and xlstm_rank1_update_f32 defer to the scalar bodies in
 * xlstm_simd_scalar.h - the same text xlstm_simd_ref.c compiles, not a
 * copy of it. There is no f32 SIMD on these cores, and rank-1 update is
 * bandwidth-bound rather than issue-bound, so an honest measurement is all
 * either would produce.
 *
 * Dependencies: ACLE (arm_acle.h ships with the compiler) plus three lines
 * of inline asm. Deliberately not CMSIS-NN: it would accelerate exactly one
 * of these kernels at the cost of the library's dependency-free property.
 * ===========================================================================*/

#include "xlstm_simd.h"

#include "xlstm_simd_scalar.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* The requirement is the DSP extension, which is what __ARM_FEATURE_SIMD32
 * declares - not a core. Cortex-M4/M7/M33 are what this backend is written
 * for, but SXTAB16 and SMLAD are ARMv6 DSP instructions that A-profile also
 * has, so -march=armv7-a satisfies the guard as well and runs the arithmetic
 * under an emulator (make test-cortexm). That build gates the arithmetic
 * only: it does not reproduce M-profile alignment behaviour, the
 * -mno-unaligned-access load path, or anything about cycles. */
#ifndef __ARM_FEATURE_SIMD32
#error "XLSTM_SIMD=cortexm needs the DSP extension: build with -mcpu=cortex-m4," \
       " cortex-m7 or cortex-m33 (plus -mthumb), or -march=armv7-a+fp to run it" \
       " under emulation. Cortex-M0/M0+/M23 have no DSP extension at all - use" \
       " XLSTM_SIMD=ref there."
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
 * rather than a byte assembly, so it may only be used once the caller has
 * established that the pointer really is word-aligned. */
typedef uint32_t xlstm_cm_word __attribute__((__may_alias__));

/* One 4-byte group of a caller buffer. ALIGNED and the #if are both decided
 * at compile time, so neither instance of the loop below carries a branch.
 *
 * Aligned is a plain LDR. Unaligned is spelled two ways for one reason: where
 * unaligned access is permitted memcpy is a single LDR, but under
 * -mno-unaligned-access it becomes a call to memcpy, and the byte assembly
 * that avoids the call is folded straight back into an unaligned LDR unless
 * that same flag forbids it. Each spelling is used exactly where it is the
 * good one. Firmware that sets CCR.UNALIGN_TRP (the recommended debug
 * setting, and a UsageFault on any unaligned LDR) builds with the flag - as
 * it must for any C in the image - and gets a kernel that issues no unaligned
 * access at all, at about 4.5 instructions per MAC against 2, still under the
 * scalar body's 6.
 *
 * Byte order is not chosen here: all three groups of a pass are read the same
 * way, so any permutation cancels between the lanes SMLAD pairs up. */
static inline int8x4_t xlstm_cm_ld4(const int8_t* p, int aligned)
{
    uint32_t w;
    if (aligned) {
        w = *(const xlstm_cm_word*)(const void*)p;
    } else {
#ifdef __ARM_FEATURE_UNALIGNED
        memcpy(&w, p, sizeof w);
#else
        w = (uint32_t)(uint8_t)p[0] | ((uint32_t)(uint8_t)p[1] << 8) |
            ((uint32_t)(uint8_t)p[2] << 16) | ((uint32_t)(uint8_t)p[3] << 24);
#endif
    }
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

/* The DSP body. ALIGNED means both what its name says and that cols is a
 * multiple of 4 - one flag because the dispatch below tests them together and
 * a word-typed load needs both. It reaches xlstm_cm_ld4 as a constant from
 * both call sites; always_inline so that stays true however the inliner's
 * size heuristic feels about two loops, since a surviving runtime `aligned`
 * would put a branch on every group load of the common case. */
static inline __attribute__((always_inline)) void
xlstm_cm_matvec_s8(const int8_t* M, const int8_t* v, int32_t* out,
                   int rows, int cols, int32_t v_zp, int aligned)
{
    /* Whole groups fill [0, nb); the 0 to 3 columns above them go scalar.
     * Deriving tail from ALIGNED rather than from cols is what deletes the
     * scalar remainder outright from the aligned instance. */
    const size_t nb = (size_t)cols & ~(size_t)3;
    const int tail = aligned ? 0 : (cols & 3);
    /* -v_zp in both halfword lanes, the addend SXTAB16 applies. */
    const int16x2_t nzp = (int16x2_t)(((uint32_t)(-v_zp) & 0xFFFFu) |
                                      ((uint32_t)(-v_zp) << 16));
    /* Both row loops run to a pointer rather than counting i: that leaves one
     * fewer value live across the inner loop, which is what keeps the row
     * epilogue off the stack. On a part whose stack is uncached that is worth
     * far more than the instructions it saves - about 10 cycles per avoided
     * spill on a Cortex-M7 reading its stack over the AXI bus with the data
     * cache off, against about 1 on a core whose SRAM is single-cycle. nrows
     * folds a non-positive rows into an empty range instead of a negative
     * pointer offset. */
    const size_t stride = (size_t)cols;
    const size_t nrows = rows > 0 ? (size_t)rows : 0u;
    const int8_t* const rowend2 = M + (nrows & ~(size_t)1) * stride;
    const int8_t* const rowend = M + nrows * stride;
    const int8_t* const vtop = v + nb;
    const int8_t* row = M;
    int32_t* o = out;
    int j;

    /* Two rows per pass: the widened v lanes are computed once and
     * consumed twice, and the two accumulator chains are independent,
     * so neither stalls on the other's SMLAD result. */
    for (; row != rowend2; row += 2 * stride, o += 2) {
        const int8_t* p0 = row + nb;
        const int8_t* p1 = p0 + stride;
        const int8_t* vp = vtop;
        int32_t a0 = 0;
        int32_t a1 = 0;

        /* Groups are consumed from the top of the row downwards, and that
         * choice is load-bearing rather than cosmetic: descending, every
         * cursor advances before it is read, which is a pre-indexed
         * LDR Rd,[Rn,#-4]!. Ascending, the same three cursors advance after
         * the read and two of them come out as post-indexed LDR Rd,[Rn],#4.
         * Measured, a Cortex-M4 runs the post-indexed spelling about 2 cycles
         * per iteration slower than the otherwise byte-identical loop, and
         * the penalty does not scale with how many of the three are
         * post-indexed. M7 and M33 are near-indifferent to it.
         *
         * Order costs nothing to give up: every term is an exact integer
         * product and the accumulator cannot overflow (see below), so a
         * descending sum is bit-identical to an ascending one. */
        while (p0 != row) {
            int8x4_t w, m0, m1;
            int16x2_t ve, vo;

            vp -= 4;
            p0 -= 4;
            p1 -= 4;
            w = xlstm_cm_ld4(vp, aligned);
            ve = __sxtab16(nzp, w);              /* v[4g+0], v[4g+2] */
            vo = xlstm_cm_sxtab16_ror8(nzp, w);  /* v[4g+1], v[4g+3] */
            m0 = xlstm_cm_ld4(p0, aligned);
            m1 = xlstm_cm_ld4(p1, aligned);

            a0 = __smlad(__sxtb16(m0), ve, a0);
            a0 = __smlad(xlstm_cm_sxtb16_ror8(m0), vo, a0);
            a1 = __smlad(__sxtb16(m1), ve, a1);
            a1 = __smlad(xlstm_cm_sxtb16_ror8(m1), vo, a1);
        }
        /* All three cursors are back at column 0 of their own row, and the
         * leftover columns are the ones above the last whole group. */
        for (j = 0; j < tail; ++j) {
            a0 += (int32_t)p0[nb + j] * ((int32_t)vp[nb + j] - v_zp);
            a1 += (int32_t)p1[nb + j] * ((int32_t)vp[nb + j] - v_zp);
        }
        o[0] = a0;
        o[1] = a1;
    }

    for (; row != rowend; row += stride, ++o) {
        const int8_t* p0 = row + nb;
        const int8_t* vp = vtop;
        int32_t a0 = 0;

        while (p0 != row) {
            int8x4_t w, m0;

            vp -= 4;
            p0 -= 4;
            w = xlstm_cm_ld4(vp, aligned);
            m0 = xlstm_cm_ld4(p0, aligned);

            a0 = __smlad(__sxtb16(m0), __sxtab16(nzp, w), a0);
            a0 = __smlad(xlstm_cm_sxtb16_ror8(m0),
                         xlstm_cm_sxtab16_ror8(nzp, w), a0);
        }
        for (j = 0; j < tail; ++j) {
            a0 += (int32_t)p0[nb + j] * ((int32_t)vp[nb + j] - v_zp);
        }
        *o = a0;
    }
}

void xlstm_matvec_s8(const int8_t* M, const int8_t* v,
                     int32_t* out, int rows, int cols, int32_t v_zp)
{
    /* Two things still leave the DSP body: a zero point SXTAB16 cannot fold
     * exactly, and a cols of 0 or less, for which a row is not a range of
     * addresses at all and the loops below would step past out entirely.
     * Neither is alignment. */
    if (cols <= 0 || v_zp > XLSTM_CM_ZP_MAX || v_zp < -XLSTM_CM_ZP_MAX) {
        xlstm_scalar_matvec_s8(M, v, out, rows, cols, v_zp);
        return;
    }

    /* v word-aligned aligns every group of v; M word-aligned with cols a
     * multiple of 4 makes every row start M + i*cols inherit that alignment
     * and leaves no ragged tail group. That case gets word-typed loads.
     *
     * Anything else runs the same body over unaligned group loads, which is a
     * load form rather than a slower kernel. Co-alignment is not being worked
     * around here, it is unobtainable: SMLAD needs 4 bytes of a row and 4 of v
     * at one column index, cols % 4 != 0 shifts each row start against v by a
     * further byte, and no per-row prefix can recover that for more than one
     * row in four. A path that insisted on aligned words would therefore
     * abandon the DSP body on most rows of every odd hidden size - H = 17
     * among them - which is where the whole of the work is. */
    if ((((uintptr_t)M | (uintptr_t)v) & 3u) == 0u && (cols & 3) == 0) {
        xlstm_cm_matvec_s8(M, v, out, rows, cols, v_zp, 1);
    } else {
        xlstm_cm_matvec_s8(M, v, out, rows, cols, v_zp, 0);
    }
}

void xlstm_matvec_f32(const float* M, const float* v,
                      float* out, int rows, int cols)
{
    /* Blocked by rows, because this kernel is short of loads rather than of
     * multiplies: v[j] is read once and consumed by every row of the block, so
     * B rows cost B + 1 loads per B MACs - 1.5 loads/MAC at B = 2, 1.125 at
     * B = 8. The B accumulator chains are independent, which also hides the
     * FPU's multi-cycle accumulate latency, and fmaf is one VFMA.F32 on FPv4-SP
     * and FPv5, so this rounds once where the scalar body rounds twice. That is
     * a deliberate numeric difference from ref and the source of the win.
     *
     * B = 8 because both register files run out there at once:
     *
     *   Singles: a block holds 2B + 1 live - B accumulators, B weights in
     *   flight, v[j]. AAPCS leaves 16 scratch (s0-s15), so B = 8 is the first
     *   width that touches a callee-saved one, at one VPUSH/VPOP of d8 per
     *   call rather than per iteration.
     *
     *   Cores: the inner loop needs B + 2 of the 14 usable (B row cursors, the
     *   v cursor, its limit), leaving 4 for an outer loop that wants 5 - so one
     *   block-invariant is reloaded per block, about 4 loads per 8 rows. B = 12
     *   would want all 14 and start shuttling registers through the stack per
     *   block, which on a part whose stack is uncached costs more than the
     *   0.04 loads/MAC the deeper block saves.
     *
     * All three cores compile the body below to 9 loads and 19 instructions per
     * 8 MACs with no stack reference in it at all.
     *
     * How rows interleave changes; summation within a row does not. Each output
     * still starts at out[i] and accumulates ascending j through the same fmaf,
     * and rows are independent, so this is bit-identical to the two-row shape
     * it replaces and the golden data does not move. */
    const size_t stride = (size_t)cols;
    int i = 0;
    int j;

    /* A cols of 0 or less is not a range of addresses, so it never reaches
     * the cursor loop below; the counted loops that follow already do nothing
     * for it, exactly as the scalar body does. */
    if (cols > 0) {
        for (; i + 7 < rows; i += 8) {
            const float* p0 = M + (size_t)i * stride;
            const float* p1 = p0 + stride;
            const float* p2 = p1 + stride;
            const float* p3 = p2 + stride;
            const float* p4 = p3 + stride;
            const float* p5 = p4 + stride;
            const float* p6 = p5 + stride;
            const float* p7 = p6 + stride;
            const float* vp = v;
            const float* const vend = v + stride;
            float a0 = out[i], a1 = out[i + 1];
            float a2 = out[i + 2], a3 = out[i + 3];
            float a4 = out[i + 4], a5 = out[i + 5];
            float a6 = out[i + 6], a7 = out[i + 7];

            /* Every cursor is walked rather than indexed: eight base pointers
             * plus an index would not fit the core registers, and the row
             * pointers are derived here rather than in the outer loop so that
             * only one of them has to stay live across it. */
            while (vp != vend) {
                float x = *vp++;
                a0 = fmaf(*p0++, x, a0);
                a1 = fmaf(*p1++, x, a1);
                a2 = fmaf(*p2++, x, a2);
                a3 = fmaf(*p3++, x, a3);
                a4 = fmaf(*p4++, x, a4);
                a5 = fmaf(*p5++, x, a5);
                a6 = fmaf(*p6++, x, a6);
                a7 = fmaf(*p7++, x, a7);
            }
            out[i] = a0; out[i + 1] = a1;
            out[i + 2] = a2; out[i + 3] = a3;
            out[i + 4] = a4; out[i + 5] = a5;
            out[i + 6] = a6; out[i + 7] = a7;
        }
    }

    /* Both callers make rows a multiple of two - 4H for sLSTM, 4H + 2 for
     * mLSTM - so what is left here is 0, 2, 4 or 6 rows and the two-row pass
     * covers all of it. A four-row tier would only ever serve the 4 or 6, at
     * H = 1 and H = 17, where it is a fraction of a percent of the work. */
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
