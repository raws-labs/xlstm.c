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
 * Armv8.1-M MVE backend - Helium. Cortex-M55, Cortex-M85.
 *
 * All four contract functions are accelerated, and every one of them is
 * BIT-IDENTICAL to the scalar body in xlstm_simd_scalar.h at every shape,
 * every alignment and every zero point. That is the design constraint the
 * rest of this file is arranged around, not a happy result:
 *
 *   xlstm_matvec_f32  four ROWS per vector, one column at a time. Lane r
 *                     holds row r's accumulator, so each output still sums
 *                     ascending j into its own running total exactly as the
 *                     scalar body does. Costs a gather load; buys exactness.
 *   xlstm_matvec_s8   16 columns per VMLADAVA.S8, with VADDVA.S8 running the
 *                     row sum alongside so the zero point folds in at the
 *                     end. Integers, so order is free.
 *   xlstm_rank1_update_f32
 *                     four elements per pass, VMUL + VMUL + VADD. Nothing
 *                     sums across elements here, so the only thing to
 *                     preserve is that neither multiply gets contracted into
 *                     the add.
 *   xlstm_vecmat_f32  four columns of out[] held in a vector across the row
 *                     loop. This is the one kernel whose natural vector
 *                     dimension is already the scalar body's, so it is exact
 *                     and contiguous at once.
 *
 * WHAT PREDICATION BUYS, WHICH IS THE REASON THIS BACKEND EXISTS
 *
 * MVE loads and stores are predicated per lane, and an inactive lane makes
 * no memory access at all. Two whole classes of work that the other vector
 * backends here spend most of their length on therefore do not arise:
 *
 *   No alignment path. MVE contiguous accesses require alignment to the
 *   ELEMENT, not to the vector: VLDRB.8 takes any address, VLDRW.32 takes
 *   any 4-byte one, which every float already is. So no call in this file
 *   dispatches on where a buffer landed. Compare xlstm_simd_esp.c, which
 *   assembles each unaligned 16-column window from the two aligned blocks
 *   containing it, and xlstm_simd_cortexm.c, which carries two spellings of
 *   its group load.
 *
 *   No scalar remainder. A partial vector is the same instruction under a
 *   narrower predicate, so an odd hidden size runs the vector body over its
 *   last few lanes rather than falling out of it. H = 17 is not a special
 *   case anywhere below: matvec_f32 has no column tail at all (its vector
 *   dimension is rows, and its partial row block is the same loop with
 *   clamped offsets), and the other three end in one predicated pass.
 *
 * What is left of dispatch is a shape with no work in it - no rows, no
 * columns, an H of zero. Those defer to xlstm_simd_scalar.h, the same text
 * xlstm_simd_ref.c compiles, rather than to a copy of it.
 *
 * `make test-helium` cross-compiles the four golden-vector suites plus
 * test/helium_gate.cc for Cortex-M55 and runs them on an emulated MPS3
 * AN547; read that target's comment before quoting a green run.
 * ===========================================================================*/

#include "xlstm_simd.h"

#include "xlstm_simd_scalar.h"

#include <stddef.h>
#include <stdint.h>

/* The requirement is MVE, which is Armv8.1-M Mainline plus the vector
 * extension: Cortex-M55 and Cortex-M85. __ARM_FEATURE_MVE is ACLE's name for
 * it, bit 0 for the integer half and bit 1 for the floating-point half.
 *
 * Worth stating because it costs an afternoon otherwise: -mfloat-abi=soft
 * turns MVE OFF even on -mcpu=cortex-m55, so a build that looks like it asked
 * for this backend lands here with no feature macro at all. */
#ifndef __ARM_FEATURE_MVE
#error "XLSTM_SIMD=helium is the Armv8.1-M MVE backend: build it with" \
       " -mcpu=cortex-m55 or -mcpu=cortex-m85, -mthumb and -mfloat-abi=hard" \
       " (soft-float disables MVE), or use XLSTM_SIMD=cortexm on an Armv7E-M" \
       " part and XLSTM_SIMD=ref anywhere else."
#endif

#include <arm_mve.h>

/* MVE without its floating-point half is a real configuration - both cores
 * can be built integer-only - and it still runs the INT8 kernel, which is
 * what an MCU deployment is usually there for. The three f32 kernels defer to
 * the scalar bodies in that build rather than failing to compile. */
#define XLSTM_HL_HAVE_FP ((__ARM_FEATURE_MVE & 2) != 0)

/* Which body a call took, for test/helium_gate.cc. Three outcomes per
 * kernel rather than two, because the second thing this backend claims is
 * that an odd size stays in the vector body: `predicated` counts the calls
 * that ended in a narrowed pass, and the gate asserts it ticks on exactly
 * the sizes whose shape says it must. Without it a kernel that quietly grew
 * a scalar remainder would still look accelerated.
 *
 * Off unless XLSTM_HELIUM_FASTPATH_COUNTERS is defined (make test-helium
 * sets it, the ordinary build does not), so a shipping kernel carries
 * neither the counters nor the increment. */
#ifdef XLSTM_HELIUM_FASTPATH_COUNTERS
unsigned long xlstm_helium_matvec_f32_vector = 0;
unsigned long xlstm_helium_matvec_f32_scalar = 0;
unsigned long xlstm_helium_matvec_f32_predicated = 0;
unsigned long xlstm_helium_matvec_s8_vector = 0;
unsigned long xlstm_helium_matvec_s8_scalar = 0;
unsigned long xlstm_helium_matvec_s8_predicated = 0;
unsigned long xlstm_helium_rank1_f32_vector = 0;
unsigned long xlstm_helium_rank1_f32_scalar = 0;
unsigned long xlstm_helium_rank1_f32_predicated = 0;
unsigned long xlstm_helium_vecmat_f32_vector = 0;
unsigned long xlstm_helium_vecmat_f32_scalar = 0;
unsigned long xlstm_helium_vecmat_f32_predicated = 0;
#define XLSTM_HL_COUNT(name, vec, pred)                              \
    ((void)((vec) ? ++xlstm_helium_##name##_vector                   \
                  : ++xlstm_helium_##name##_scalar),                 \
     (void)((pred) ? ++xlstm_helium_##name##_predicated : 0ul))
#else
#define XLSTM_HL_COUNT(name, vec, pred) ((void)0)
#endif

/* ---------------------------------------------------------------------------
 * INT8 matrix-vector: out[i] = sum_j M[i][j] * (v[j] - v_zp)
 *
 * VMLADAVA.S8 multiplies 16 int8 pairs and adds all 16 products into a
 * 32-bit general-purpose accumulator in one instruction. It has no zero-point
 * operand, so the subtraction is folded afterwards using the identity
 *
 *     sum_j M[i][j] * (v[j] - z)  ==  sum_j M[i][j] * v[j]  -  z * sum_j M[i][j]
 *
 * and the row sum it needs is one more instruction per group, VADDVA.S8.
 * Measured on the emitted code: eight instructions per pass of the two-row
 * loop, which is 32 MACs, against the scalar body's six per MAC.
 *
 * The identity holds in int32 for EVERY int32 zero point, not just the int8
 * ones a quantizer produces: both sides are evaluated modulo 2^32, and two
 * expressions equal over the integers are equal modulo 2^32 whatever their
 * intermediates did. So this kernel has no zero-point bound to check and no
 * fallback to fall into, which is the one place it differs in kind from its
 * siblings - xlstm_simd_cortexm.c leaves the DSP body above |z| = 32640
 * because SXTAB16's halfword lanes wrap, xlstm_simd_esp.c above |z| = 254
 * because it packs the zero point into int8 lanes.
 *
 * Accumulator headroom, for the arithmetic rather than the identity: a term
 * |M[j] * (v[j] - z)| is at most 127 * 255 = 32385 in the int8 domain, so
 * int32 saturates only past ~66000 columns. VMLADAVA.S8 is the non-saturating
 * form, and the 16 products it sums internally reach 16 * 16256 at worst.
 * ------------------------------------------------------------------------ */

/* dot - z * rowsum, in unsigned arithmetic so that a zero point large enough
 * to overflow the product wraps rather than being undefined. The scalar body
 * this must match wraps too; unsigned is how that is spelled without relying
 * on it. */
static inline int32_t xlstm_hl_fold_zp(int32_t dot, int32_t rowsum,
                                       int32_t v_zp)
{
    return (int32_t)((uint32_t)dot - (uint32_t)v_zp * (uint32_t)rowsum);
}

void xlstm_matvec_s8(const int8_t* M, const int8_t* v,
                     int32_t* out, int rows, int cols, int32_t v_zp)
{
    /* Whole 16-column groups, then at most one narrowed pass over what is
     * left. `tail` is a count of columns and never a reason to leave the
     * vector body. */
    const size_t stride = (size_t)cols;
    const int nb = cols & ~15;
    const int tail = cols & 15;
    int i = 0;

    /* A cols or rows of 0 or less is not a range of addresses; the shared
     * scalar body already does nothing for it, correctly. */
    if (cols <= 0 || rows <= 0) {
        XLSTM_HL_COUNT(matvec_s8, 0, 0);
        xlstm_scalar_matvec_s8(M, v, out, rows, cols, v_zp);
        return;
    }
    XLSTM_HL_COUNT(matvec_s8, 1, tail != 0);

    /* Two rows per pass. The v group is loaded once and consumed twice, and
     * more importantly the two VMLADAVA chains are independent: consecutive
     * accumulations into one scalar register serialise on that register,
     * which is the across-vector reduction's cost rather than its throughput.
     * 8 instructions per 32 MACs here against the single-row loop's 10. */
    for (; i + 1 < rows; i += 2) {
        const int8_t* p0 = M + (size_t)i * stride;
        const int8_t* p1 = p0 + stride;
        /* Walked rather than indexed, and this is not cosmetic: with a shared
         * index gcc keeps three base registers plus j live and rebuilds each
         * address in the loop, 12 instructions per pass. Walking the three
         * cursors turns every load into a post-indexed VLDRB.8 and the pass
         * into 8. */
        const int8_t* q0 = p0;
        const int8_t* q1 = p1;
        const int8_t* qv = v;
        int32_t d0 = 0, d1 = 0, s0 = 0, s1 = 0;
        int j;

        for (j = 0; j < nb; j += 16) {
            const int8x16_t vv = vldrbq_s8(qv);
            const int8x16_t a0 = vldrbq_s8(q0);
            const int8x16_t a1 = vldrbq_s8(q1);

            qv += 16;
            q0 += 16;
            q1 += 16;
            d0 = vmladavaq_s8(d0, a0, vv);
            d1 = vmladavaq_s8(d1, a1, vv);
            s0 = vaddvaq_s8(s0, a0);
            s1 = vaddvaq_s8(s1, a1);
        }
        if (tail != 0) {
            /* Only the three LOADS are predicated. An inactive lane makes no
             * access, so nothing is read past the row, and the zeroing form
             * leaves those lanes at 0 - which contributes 0 to both the
             * product sum and the row sum, so the four arithmetic
             * instructions need no predicate of their own. */
            const mve_pred16_t p = vctp8q((uint32_t)tail);
            const int8x16_t vv = vldrbq_z_s8(v + j, p);
            const int8x16_t a0 = vldrbq_z_s8(p0 + j, p);
            const int8x16_t a1 = vldrbq_z_s8(p1 + j, p);

            d0 = vmladavaq_s8(d0, a0, vv);
            d1 = vmladavaq_s8(d1, a1, vv);
            s0 = vaddvaq_s8(s0, a0);
            s1 = vaddvaq_s8(s1, a1);
        }
        out[i] = xlstm_hl_fold_zp(d0, s0, v_zp);
        out[i + 1] = xlstm_hl_fold_zp(d1, s1, v_zp);
    }

    for (; i < rows; ++i) {
        const int8_t* p0 = M + (size_t)i * stride;
        int32_t d0 = 0, s0 = 0;
        int j;

        for (j = 0; j < nb; j += 16) {
            const int8x16_t vv = vldrbq_s8(v + j);
            const int8x16_t a0 = vldrbq_s8(p0 + j);

            d0 = vmladavaq_s8(d0, a0, vv);
            s0 = vaddvaq_s8(s0, a0);
        }
        if (tail != 0) {
            const mve_pred16_t p = vctp8q((uint32_t)tail);
            const int8x16_t vv = vldrbq_z_s8(v + j, p);
            const int8x16_t a0 = vldrbq_z_s8(p0 + j, p);

            d0 = vmladavaq_s8(d0, a0, vv);
            s0 = vaddvaq_s8(s0, a0);
        }
        out[i] = xlstm_hl_fold_zp(d0, s0, v_zp);
    }
}

#if XLSTM_HL_HAVE_FP

/* ---------------------------------------------------------------------------
 * f32 matrix-vector: out[i] += sum_j M[i][j] * v[j]
 *
 * WHY THE VECTOR DIMENSION IS ROWS AND NOT COLUMNS
 *
 * Four columns per vector is the obvious shape. It ends in a horizontal
 * reduction, which reassociates the sum - and since MVE has no across-vector
 * float add, that reduction is lane extracts on top of the loop.
 *
 * Stated honestly, because it was measured rather than assumed: the obvious
 * shape PASSES the golden vectors on this core. Reassociation was not ruled
 * out by the goldens here, it was ruled out on two other grounds.
 *
 * The first is that the margin is thin and case-dependent, not absent. On the
 * esp backend a far smaller change - moving out[i] from the seed of the
 * accumulator to a final add - put mLSTM SweepM17 y[0] 4.20e-05 from its
 * golden value against a bound of about 3.98e-05. A formulation that clears
 * today's bounds at today's widths on today's weights is not the same thing
 * as one that cannot move a golden.
 *
 * The second is what exactness does for the gate, and it is the stronger
 * reason. Because every f32 kernel here is bit-identical to the scalar body,
 * test/helium_gate.cc compares by EQUALITY. That is what lets it catch a
 * single fused multiply-add creeping into a loop: injected, it shows up as a
 * 3e-08 difference, and no tolerance wide enough to permit reassociation
 * would ever fail on it. Choosing exactness buys a gate that fails on
 * last-bit defects; choosing speed here would have bought a gate that cannot.
 *
 * With lane r holding row r's accumulator, every output starts at out[i + r]
 * and takes its terms in ascending j through one multiply and one add, which
 * is xlstm_scalar_matvec_f32 element for element. The price is that M[i][j]
 * for four consecutive i is a stride-cols read, so the load is a gather:
 * VLDRW.U32 Qd, [Rn, Qm, UXTW #2], one base plus a vector of element offsets.
 *
 * Four working instructions per four MACs - LDR of v[j], the gather, VMUL by
 * a scalar from a general-purpose register, VADD - which gcc emits as a
 * seven-instruction DLS/LE loop once the address bookkeeping and the
 * zero-cost branch are counted, against the scalar body's six per MAC. Be
 * careful reading that as a speedup, and this is the one place in
 * this backend where the accelerated form may genuinely lose: a word gather
 * is several beats on a Cortex-M55 where a contiguous 128-bit load is two,
 * and nothing available here measures cycles. If a measurement on silicon
 * ever shows the gather losing to a blocked scalar loop, the answer is to
 * take the scalar body from xlstm_simd_scalar.h, not to reassociate.
 *
 * There is no tail. The row count is the vector dimension, so a partial block
 * is the same loop under a narrower predicate; the column count never meets a
 * vector boundary at all. Both are why H = 17 costs this kernel nothing.
 * ------------------------------------------------------------------------ */

void xlstm_matvec_f32(const float* M, const float* v,
                      float* out, int rows, int cols)
{
    /* {0, 1, 2, 3}, the row index within a block. */
    const uint32x4_t lane = vidupq_n_u32(0u, 1);
    int i;

    if (cols <= 0 || rows <= 0) {
        XLSTM_HL_COUNT(matvec_f32, 0, 0);
        xlstm_scalar_matvec_f32(M, v, out, rows, cols);
        return;
    }
    XLSTM_HL_COUNT(matvec_f32, 1, (rows & 3) != 0);

    for (i = 0; i < rows; i += 4) {
        /* Lanes past the last row are clamped onto it rather than predicated
         * off. A gather with an in-bounds offset in every lane keeps ONE
         * inner loop for full and partial blocks alike - no second body, no
         * VPST in the hot loop - and the duplicate lanes are discarded by the
         * predicated store below. For a full block the clamp is the identity,
         * since rows - i - 1 is then at least 3. */
        const uint32x4_t off =
            vmulq_n_u32(vminq_u32(lane, vdupq_n_u32((uint32_t)(rows - i - 1))),
                        (uint32_t)cols);
        const mve_pred16_t p = vctp32q((uint32_t)(rows - i));
        const float* base = M + (size_t)i * (size_t)cols;
        /* Seeded from out[], not from zero: the contract accumulates, and
         * where the seed enters the sum is exactly what the esp backend
         * measured a golden's worth of difference on. */
        float32x4_t acc = vldrwq_z_f32(out + i, p);
        int j;

        /* Counted, not walked to a pointer limit. The two spellings are the
         * same loop and gcc compiles them differently: counted, it emits a
         * DLS/LE hardware loop, seven instructions with a zero-cost branch;
         * walked, it emits CMP and BNE and one more instruction besides.
         * The opposite of what xlstm_simd_cortexm.c wants, and worth the note
         * because the two files sit next to each other. */
        for (j = 0; j < cols; ++j) {
            const float32x4_t m =
                vldrwq_gather_shifted_offset_f32(base + j, off);
            /* Separate VMUL and VADD, never VFMA: the scalar body rounds the
             * product and then the sum, and this has to round in the same two
             * places. gcc does not contract across intrinsics, so this is
             * stable rather than a flag's doing - but the gate compares
             * against the scalar body bit for bit, so a build that started
             * contracting would be caught rather than trusted. */
            acc = vaddq_f32(acc, vmulq_n_f32(m, v[j]));
        }
        vstrwq_p_f32(out + i, acc, p);
    }
}

/* ---------------------------------------------------------------------------
 * Rank-1 update: C[r][c] = f * C[r][c] + (i * k[r]) * v[c]
 *
 * Four elements per pass and nothing sums across them, so this is the one
 * kernel where lane order and accumulation order are both beside the point.
 * What has to match is CONTRACTION: two multiplies feed one add, and fusing
 * either of them changes the last bit. VMUL, VMUL, VADD keeps all three
 * roundings the scalar body has.
 *
 * Six working instructions per four elements - two loads, two multiplies, an
 * add and a store - against about five per element scalar. The whole H x H
 * state is read and written every timestep, so this kernel is bandwidth-bound
 * and the store width is most of what it gains.
 *
 * The row loop walks C in memory order rather than blocking columns to hoist
 * the v load out of it. Hoisting would save one load per four elements and
 * cost the sequential walk of an H x H state that does not fit a Cortex-M
 * cache line budget at the widths this library targets.
 * ------------------------------------------------------------------------ */

void xlstm_rank1_update_f32(float* C, float f_gate, float i_gate,
                            const float* k, const float* v, int rows, int cols)
{
    const int nb = cols & ~3;
    const int tail = cols & 3;
    int r;

    if (rows <= 0 || cols <= 0) {
        XLSTM_HL_COUNT(rank1_f32, 0, 0);
        xlstm_scalar_rank1_update_f32(C, f_gate, i_gate, k, v, rows, cols);
        return;
    }
    XLSTM_HL_COUNT(rank1_f32, 1, tail != 0);

    for (r = 0; r < rows; ++r) {
        const float ik_r = i_gate * k[r];
        float* Cr = C + (size_t)r * (size_t)cols;
        int c;

        for (c = 0; c < nb; c += 4) {
            const float32x4_t cv = vldrwq_f32(Cr + c);
            const float32x4_t vv = vldrwq_f32(v + c);

            vstrwq_f32(Cr + c, vaddq_f32(vmulq_n_f32(cv, f_gate),
                                         vmulq_n_f32(vv, ik_r)));
        }
        if (tail != 0) {
            /* The same three instructions over fewer lanes. An inactive lane
             * is neither read nor written, so the columns past cols keep
             * whatever the caller had there - which for a row-major
             * rows x cols state is the next row, and must not be touched. */
            const mve_pred16_t p = vctp32q((uint32_t)tail);
            const float32x4_t cv = vldrwq_z_f32(Cr + c, p);
            const float32x4_t vv = vldrwq_z_f32(v + c, p);

            vstrwq_p_f32(Cr + c, vaddq_f32(vmulq_n_f32(cv, f_gate),
                                           vmulq_n_f32(vv, ik_r)), p);
        }
    }
}

/* ---------------------------------------------------------------------------
 * Left multiply: out[j] += sum_i q[i] * M[i][j]
 *
 * The one kernel whose natural vector dimension is already the scalar body's.
 * out[j] is the accumulator and j is the contiguous direction, so four
 * columns of out[] live in one vector across the whole row loop, each lane
 * taking its terms in ascending i through one multiply and one add. Exact and
 * contiguous at the same time, with no gather and no reduction.
 *
 * Four working instructions per four MACs: the LDR of q[i], a contiguous
 * VLDRW of the row, VMUL by scalar, VADD - six emitted, with the row-stride
 * add that a strided walk needs and the zero-cost branch. Nothing loads or
 * stores out[] inside the loop. The narrowed pass is the same six plus one
 * VPST, which is what a scalar remainder would have cost per COLUMN.
 * ------------------------------------------------------------------------ */

/* One block of up to four columns. `predicated` is a compile-time constant at
 * both call sites - always_inline so it stays one, since a surviving runtime
 * test would put a VPST on every load of the common case rather than on the
 * one block that needs it. */
static inline __attribute__((always_inline)) void
xlstm_hl_vecmat_block(const float* q, const float* M, float* out,
                      int rows, int cols, mve_pred16_t p, int predicated)
{
    float32x4_t acc = predicated ? vldrwq_z_f32(out, p) : vldrwq_f32(out);
    const float* mp = M;
    int i;

    for (i = 0; i < rows; ++i) {
        const float32x4_t mv =
            predicated ? vldrwq_z_f32(mp, p) : vldrwq_f32(mp);

        acc = vaddq_f32(acc, vmulq_n_f32(mv, q[i]));
        mp += cols;
    }
    if (predicated) {
        vstrwq_p_f32(out, acc, p);
    } else {
        vstrwq_f32(out, acc);
    }
}

void xlstm_vecmat_f32(const float* q, const float* M,
                      float* out, int rows, int cols)
{
    const int nb = cols & ~3;
    const int tail = cols & 3;
    int j;

    if (rows <= 0 || cols <= 0) {
        XLSTM_HL_COUNT(vecmat_f32, 0, 0);
        xlstm_scalar_vecmat_f32(q, M, out, rows, cols);
        return;
    }
    XLSTM_HL_COUNT(vecmat_f32, 1, tail != 0);

    for (j = 0; j < nb; j += 4) {
        xlstm_hl_vecmat_block(q, M + j, out + j, rows, cols, 0, 0);
    }
    if (tail != 0) {
        /* The last columns of every row, under a predicate. The load would
         * otherwise reach into the row below - and past the end of M on the
         * final row - which is what makes this predicated rather than merely
         * narrow. */
        xlstm_hl_vecmat_block(q, M + j, out + j, rows, cols,
                              vctp32q((uint32_t)tail), 1);
    }
}

#else /* MVE without its floating-point half */

void xlstm_matvec_f32(const float* M, const float* v,
                      float* out, int rows, int cols)
{
    XLSTM_HL_COUNT(matvec_f32, 0, 0);
    xlstm_scalar_matvec_f32(M, v, out, rows, cols);
}

void xlstm_rank1_update_f32(float* C, float f_gate, float i_gate,
                            const float* k, const float* v, int rows, int cols)
{
    XLSTM_HL_COUNT(rank1_f32, 0, 0);
    xlstm_scalar_rank1_update_f32(C, f_gate, i_gate, k, v, rows, cols);
}

void xlstm_vecmat_f32(const float* q, const float* M,
                      float* out, int rows, int cols)
{
    XLSTM_HL_COUNT(vecmat_f32, 0, 0);
    xlstm_scalar_vecmat_f32(q, M, out, rows, cols);
}

#endif /* XLSTM_HL_HAVE_FP */

const char* xlstm_simd_backend(void)
{
    return "helium";
}
