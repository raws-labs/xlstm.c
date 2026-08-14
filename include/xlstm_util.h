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
 * WHAT IT COSTS, measured rather than assumed (dynamic instructions per
 * mlstm_step_s8 call at H=8, counted under qemu-system-arm):
 *
 *     Cortex-M4    12486 -> 6904   1.81x fewer
 *     Cortex-M7     5417 -> 6381   1.18x more
 *     Cortex-M33    5778 -> 6830   1.18x more
 *
 * The FPv5 cores lose because gcc does NOT fold `a < b ? a : b` back into
 * vminnm (the ternary returns b when b is NaN; vminnm returns the non-NaN
 * operand, so they are not the same function) - it emits vcmp + vmrs + vsel,
 * three instructions instead of one. Nor is there any portable spelling of
 * roundf's half-away-from-zero rule that gcc folds into vrinta. So this is a
 * real trade: a large win on the core that has no such instructions, a modest
 * loss on the cores that do. It is taken deliberately, because a portable
 * kernel whose cost is its own arithmetic is preferable to one whose cost on
 * one target is the quality of that target's libm.
 *
 * These are DELIBERATELY not drop-in fminf/fmaxf/roundf replacements - see the
 * per-function notes on where they differ. They are named for what the
 * kernels need, not for the libm functions they displace.
 * ------------------------------------------------------------------------- */

/* min/max by plain comparison.
 *
 * Equivalent to fminf/fmaxf for every non-NaN input, with one documented
 * exception: on a tie between +0.0 and -0.0, C's fminf/fmaxf are specified to
 * be able to return either operand (glibc returns the positively-signed one;
 * this returns b). Every call site below is either a tie-free comparison
 * against a strictly non-zero bound, or feeds a value whose zero sign is not
 * observable downstream. NaN propagates here rather than being suppressed, so
 * do not use these where an operand can be NaN and the libm "return the
 * non-NaN operand" rule is being relied on - xlstm_round_clamp_i32 below
 * reproduces that rule explicitly for the one place it matters. */
static inline float xlstm_minf(float a, float b) {
    return a < b ? a : b;
}

static inline float xlstm_maxf(float a, float b) {
    return a > b ? a : b;
}

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
 * branch penalty on the cores with no branch predictor. */
static inline int32_t xlstm_round_clamp_i32(float v, float lo, float hi) {
    int32_t t;
    float d;

    v = (v <= hi) ? v : hi;   /* NaN -> hi, matching fminf(hi, NaN) */
    v = (v >= lo) ? v : lo;

    t = (int32_t)v;
    d = v - (float)t;
    return t + (int32_t)(d + d);
}

#endif /* XLSTM_UTIL_H_ */
