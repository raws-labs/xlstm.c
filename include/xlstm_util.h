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
 * Shared utilities for xLSTM kernels (sLSTM, mLSTM) - pure inline C99.
 * ===========================================================================*/

#ifndef XLSTM_UTIL_H_
#define XLSTM_UTIL_H_

#include <math.h>
#include <stdint.h>

static inline float sigmoid_f32(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float log_sigmoid_f32(float x) {
    /* log(sigmoid(x)) = -softplus(-x)
     * Split for numerical stability. */
    if (x >= 0.0f) {
        return -logf(1.0f + expf(-x));
    } else {
        return x - logf(1.0f + expf(x));
    }
}

/* ---------------------------------------------------------------------------
 * Call-free min/max/round for the quantized kernels' hot loops.
 *
 * WHY THESE EXIST. fminf/fmaxf/roundf are only single instructions on an FPU
 * that has vminnm/vmaxnm/vrinta. Armv7E-M's FPv4-SP (Cortex-M4) has none of
 * the three, so on that core each one is a real `bl` into a non-leaf newlib
 * routine - newlib's fminf and fmaxf are 18 instructions that call
 * __fpclassifyf TWICE on the common path. Measured with gcc 13.2.1 -O2, the
 * O(H^2) C-update loop in mlstm_s8.c was 28 instructions containing five such
 * calls on Cortex-M4, and 22 instructions containing zero on Cortex-M7/M33
 * (FPv5), from identical source. That gap is why an M4 latency figure for the
 * INT8 mLSTM cell measured newlib more than it measured xLSTM.
 *
 * The explicit form below is the portable default and costs FPv5 something,
 * because gcc does NOT fold `a < b ? a : b` back into vminnm - the ternary
 * returns b when b is NaN, vminnm returns the non-NaN operand, so they are
 * not the same function and gcc is right to refuse. It emits vcmp + vmrs +
 * vsel, three instructions instead of one. Nor is there a portable spelling
 * of roundf's half-away-from-zero rule that gcc folds into vrinta. Measured
 * (dynamic instructions per mlstm_step_s8 call at H=8, under qemu-system-arm)
 * the explicit form alone cost M7 5417 -> 6381 and M33 5778 -> 6830 while
 * winning M4 12486 -> 6904. XLSTM_FPU_HAS_MINMAX_ROUND below buys the FPv5
 * cores that back without giving up the portable default.
 *
 * These are DELIBERATELY not drop-in fminf/fmaxf/roundf replacements - see the
 * per-function notes on where they differ. They are named for what the
 * kernels need, not for the libm functions they displace.
 * ------------------------------------------------------------------------- */

/* XLSTM_FPU_HAS_MINMAX_ROUND - a declared FPU capability, not a chip list.
 *
 * Set it to 1 for a target whose FPU implements IEEE-754 minNum/maxNum and
 * round-half-away-from-zero as single instructions, so that the compiler
 * lowers fminf/fmaxf/roundf to them instead of calling libm. On Arm that is
 * FPv5 and Armv8 (vminnm/vmaxnm/vrinta, FMINNM/FMAXNM/FRINTA); FPv4-SP has
 * none of the three. Default 0: a target nobody has characterized gets the
 * portable explicit form and can never silently acquire a libm call inside an
 * O(H^2) loop.
 *
 * WHY THIS IS BUILD-DECLARED AND NOT SNIFFED. There is no predefined macro
 * that expresses it, and every available proxy is measurably wrong on one of
 * this project's own three targets (gcc 13.2.1, arm-none-eabi):
 *
 *     core  arch macro              __ARM_FP  has vminnm/vrinta
 *     M4    __ARM_ARCH_7EM__        4         no
 *     M7    __ARM_ARCH_7EM__        14        YES
 *     M33   __ARM_ARCH_8M_MAIN__    4         YES
 *
 * M4 and M7 share an arch macro and differ in capability, so an arch test is
 * wrong for M7 - the target this matters most for. M4 and M33 share __ARM_FP
 * (it encodes which precisions exist, not which ARMv8 instructions do), so a
 * precision test is wrong for M33. ACLE's __ARM_FEATURE_NUMERIC_MAXMIN and
 * __ARM_FEATURE_DIRECTED_ROUNDING would say exactly the right thing, but gcc
 * does not define either for M-profile at all. So the build declares it,
 * alongside the -mfpu that decides it (set by whatever Cortex-M build links
 * this header - out of this repository's scope; see the xlstm.c-hil harness).
 *
 * The auto-detect below is a genuine capability test where one exists:
 * __ARM_FEATURE_NUMERIC_MAXMIN is ACLE's macro for exactly these numeric
 * min/max instructions, and Arm introduced numeric min/max and directed
 * rounding together as the one IEEE-754-2008 group (ARMv8-A, and FPv5 for
 * M-profile), so it is sound to read it as covering both. AArch64 defines it,
 * which is how the `neon` build gets the fast path.
 *
 * SETTING IT TO 1 WHERE IT IS NOT TRUE IS A PERFORMANCE BUG, NOT A
 * CORRECTNESS BUG: the two paths are bit-identical for every float input
 * (see xlstm_round_clamp_i32), so a mis-declared target computes exactly the
 * same numbers, just via libm calls. */
#ifndef XLSTM_FPU_HAS_MINMAX_ROUND
#  if defined(__ARM_FEATURE_NUMERIC_MAXMIN)
#    define XLSTM_FPU_HAS_MINMAX_ROUND 1
#  else
#    define XLSTM_FPU_HAS_MINMAX_ROUND 0
#  endif
#endif

/* min/max by plain comparison. DELIBERATELY NOT capability-gated - read on.
 *
 * Equivalent to fminf/fmaxf for every non-NaN input, with two differences that
 * only appear on inputs the kernels should never produce:
 *
 *   - NaN in the SECOND operand. `a < b ? a : b` returns b (the NaN); fminf
 *     returns a. When the FIRST operand is NaN the two agree (both return b),
 *     and when both are NaN both return NaN.
 *   - A tie between +0.0 and -0.0, where C leaves fminf/fmaxf free to return
 *     either operand (glibc, and Arm's vminnm/vmaxnm, both return the
 *     positively-signed one; this returns b).
 *
 * These are exactly the call sites where a capability-gated libm spelling
 * could NOT be proven bit-identical to the explicit one, so they do not get
 * one. That is not a missed optimization: all of them are O(1) or O(H) per
 * timestep, and the two cell_clip clamps sit inside a branch that is not even
 * taken unless the caller sets cell_clip. The whole O(H^2) cost this file
 * exists to fix lives in xlstm_round_clamp_i32 below, which IS gated, and
 * which IS provably identical on both paths for every float input.
 *
 * Keeping these explicit on every target is what makes the two paths agree
 * bit for bit, so a Cortex-M4 and a Cortex-M7 running the same weights cannot
 * disagree numerically. A cross-target divergence would be far worse than the
 * instructions it would save. */
static inline float xlstm_minf(float a, float b) {
    return a < b ? a : b;
}

static inline float xlstm_maxf(float a, float b) {
    return a > b ? a : b;
}

/* WHY THERE IS NO GATED ONE-SIDED CLAMP HERE, though it looks free.
 *
 * The min/max sites whose second operand is a known-good bound (1.0f, 1e-6f,
 * or params->cell_clip on the branch where it has already been tested > 0)
 * appear to admit a gated fminf/fmaxf spelling: for a QUIET NaN, both
 * `v < bound ? v : bound` and `fminf(v, bound)` yield bound, and no
 * signed-zero tie is possible against a non-zero bound. That was implemented,
 * and then the exhaustive check over all 2^32 float patterns rejected it:
 *
 *   quiet NaN   8388608 encodings, all agree
 *   SIGNALLING  8388606 encodings, ALL DIFFER - fminf(sNaN, b) returns the
 *   NaN         quieted sNaN, the ternary returns b
 *
 * xlstm_round_clamp_i32 is immune because roundf runs first and quiets its
 * input, which is why it survives the same test with zero mismatches. These
 * sites have no such step. Making them agree would rest on an argument that
 * no sNaN can reach them - true today, since IEEE-754 arithmetic never
 * RETURNS a signalling NaN and every operand here is an arithmetic result,
 * but it is an argument about the whole call graph rather than a property of
 * the function, and one a later edit could silently invalidate.
 *
 * It was measured before being given up: gating these would have been worth
 * about 1.2% of slstm_step_s8 on M7/M33 and nothing at all on mLSTM, whose
 * O(H^2) loop this file exists to fix and which is already at parity. That is
 * not worth trading an unconditional proof for a conditional one. */

/* Saturating round-to-int, replacing the requantization idiom
 *
 *     (intN_t)fmaxf(lo, fminf(hi, roundf(v)))
 *
 * that appears at every INT8/INT16 requantization point in slstm_s8.c and
 * mlstm_s8.c. Preconditions: lo and hi are integral, lo <= hi, and both are
 * exactly representable as float and as int32_t (all call sites pass INT8 or
 * INT16 limits). Returns the rounded, clamped value as int32_t; the caller
 * casts to its own width, exactly as before.
 *
 * WHY IT IS EXACTLY EQUIVALENT, in the two ways it rearranges the original:
 *
 * 1. Clamp before round instead of after. round() is non-decreasing and is
 *    the identity on integers, and lo/hi are integers, so
 *    round(clamp(v)) == clamp(round(v)) for every v: inside [lo,hi] both are
 *    round(v); above hi, round(v) >= round(hi) == hi so both are hi; below lo
 *    symmetrically. The clamp is written in fminf/fmaxf's original operand
 *    order so the non-finite cases land identically too - `(v <= hi) ? v : hi`
 *    maps NaN to hi, which is what fminf(hi, NaN) returns, and +/-Inf saturate
 *    to hi/lo as before. Doing the clamp first also removes the only path on
 *    which the float-to-int conversion could see an out-of-range value, which
 *    is undefined behaviour in C (the original was already safe for the same
 *    reason - it clamped before its cast too).
 *
 * 2. Truncate-then-fix-up instead of roundf, branchlessly. `(int32_t)v`
 *    truncates toward zero (one instruction: vcvt.s32.f32 / cvttss2si /
 *    fcvtzs). Every step of the fix-up is exact, so the result is roundf's
 *    half-AWAY-from-zero rule bit for bit, on both signs and on exact .5:
 *      - `d = v - (float)t` is exact. A binary float's fractional part is
 *        always representable, and |t| <= 32768 after the clamp so `(float)t`
 *        is exact too. d carries v's sign and |d| < 1.
 *      - `d + d` is exact (doubling only decrements the exponent) and |d+d|
 *        < 2, so the cast is in range.
 *      - `(int32_t)(d + d)` truncates toward zero, giving 0 when |d| < 0.5
 *        and +/-1 when |d| >= 0.5 - which IS round-half-away-from-zero, with
 *        the tie landing on the away side because 2 * 0.5 is exactly 1.0.
 *    This is why the familiar `(int)(v + 0.5f)` is NOT used: besides being
 *    wrong for negative v, the addition itself rounds - for v = 0x1.fffffep-2
 *    (the largest float below 0.5) v + 0.5f ties to 1.0f and it answers 1
 *    where roundf answers 0.
 *    The fix-up cannot overshoot the clamp: if |d| >= 0.5 then the stepped
 *    value is at most v + 0.5 <= hi + 0.5, and it is an integer, so it is
 *    <= hi. Symmetrically for lo.
 *
 * The branchless form is not a style choice: the `if (d >= 0.5f) t += 1;
 * else if (d <= -0.5f) t -= 1;` spelling was measured against it on all three
 * cores and is slower on every one (M4 6988 vs 6904, M7 6444 vs 6381, M33
 * 6977 vs 6830 instructions per mlstm_step_s8 call), before counting the
 * branch penalty on the cores with no branch predictor.
 *
 * THE TWO PATHS ARE THE SAME FUNCTION. This is the one helper that is
 * capability-gated, and the gate is only sound because the libm spelling and
 * the explicit spelling agree on EVERY float input, not merely on the ones
 * these kernels are expected to produce. Case by case, for integral lo/hi:
 *
 *   finite, in range   both give round(v)                (proved in 1 and 2)
 *   finite, above hi   round(v) >= hi so both give hi; explicit clamps first
 *   finite, below lo   symmetric
 *   +Inf               roundf(+Inf)=+Inf, fminf(hi,+Inf)=hi -> hi; explicit:
 *                      (+Inf <= hi) is false -> hi
 *   -Inf               fminf(hi,-Inf)=-Inf, fmaxf(lo,-Inf)=lo -> lo;
 *                      explicit: (-Inf >= lo) is false -> lo
 *   NaN                roundf(NaN)=NaN, fminf(hi,NaN)=hi -> hi; explicit:
 *                      (NaN <= hi) is false -> hi
 *   -0.0               both reach the cast with -0.0 (lo/hi are never 0, so
 *                      there is no signed-zero tie) and (int32_t)(-0.0) == 0
 *
 * Verified exhaustively rather than only argued: all 4,294,967,296 float bit
 * patterns compared between the two implementations, for both (lo,hi) pairs
 * the kernels use, zero mismatches. */
#if XLSTM_FPU_HAS_MINMAX_ROUND
static inline int32_t xlstm_round_clamp_i32(float v, float lo, float hi) {
    /* Lowers to vrinta + vminnm + vmaxnm + vcvt (Arm FPv5/Armv8): four
     * instructions, no calls. Identical results to the branch below. */
    return (int32_t)fmaxf(lo, fminf(hi, roundf(v)));
}
#else
static inline int32_t xlstm_round_clamp_i32(float v, float lo, float hi) {
    int32_t t;
    float d;

    v = (v <= hi) ? v : hi;   /* NaN -> hi, matching fminf(hi, NaN) */
    v = (v >= lo) ? v : lo;

    t = (int32_t)v;
    d = v - (float)t;
    return t + (int32_t)(d + d);
}
#endif

#endif /* XLSTM_UTIL_H_ */
