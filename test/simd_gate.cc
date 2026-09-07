/* Fast-path gate for the `sse2` and `neon` SIMD backends, run as the fifth
 * binary of `make test` and of `make test-neon`. The other four are the
 * ordinary golden-vector suites.
 *
 * One file for two backends because they are the same kernel twice: four
 * vectorized bodies over the same widths (four floats, eight int8 lanes),
 * each with a scalar remainder, and no dispatch anywhere - the vector loop's
 * own bound decides. Only the intrinsics and the counter names differ.
 *
 * The suites prove the kernels compute the right numbers. They cannot prove
 * WHICH body computed them: a vector loop that never runs a pass passes every
 * golden vector and accelerates nothing. That is not hypothetical here - on
 * the esp backend the accelerated matvec turned out to be reached 6 times in
 * 76 suite calls, by linker accident, with every suite green. So the four
 * checks below assert, per contract function, that the path a call's shape
 * dictates was taken, and that the result matches the shared scalar body.
 *
 * Each kernel reports THREE outcomes, and the third is the opposite of the
 * helium backend's: `tail` counts calls that left work outside the vector
 * body for a scalar remainder, because neither of these two ISAs predicates
 * and a width that is not a multiple of the vector's cannot stay in the
 * vector loop. Asserting it is what separates a width that is fully
 * vectorized from one that is only partly, which is the whole difference
 * between H = 16 and H = 17 here.
 *
 * Three of the four kernels are compared BIT FOR BIT. xlstm_matvec_f32 is
 * not and cannot be - see TolMatvec below - and that is stated rather than
 * hidden behind a tolerance chosen to make it go away.
 * =========================================================================*/

#include "xlstm_simd.h"
/* The scalar bodies themselves, not a copy of them: the checks below compare
 * the accelerated kernels against the same text every backend is defined
 * against. */
#include "xlstm_simd_scalar.h"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>

/* The counters live in src/xlstm_simd_{sse2,neon}.c under
 * XLSTM_{SSE2,NEON}_FASTPATH_COUNTERS, which the Makefile sets for the SIMD
 * object the tests link and for nothing else. They are named
 * xlstm_<backend>_<kernel>_{vector,scalar,tail}; XG() pastes the backend in
 * so that one file gates both. Referenced unconditionally, so that losing the
 * define is a link error rather than a gate that stops checking. */
#if defined(XLSTM_GATE_SSE2)
#define XG(kernel, outcome) xlstm_sse2_##kernel##_##outcome
#define XLSTM_GATE_NAME "sse2"
#define XLSTM_GATE_UNAME "SSE2"
#elif defined(XLSTM_GATE_NEON)
#define XG(kernel, outcome) xlstm_neon_##kernel##_##outcome
#define XLSTM_GATE_NAME "neon"
#define XLSTM_GATE_UNAME "NEON"
#else
#error "test/simd_gate.cc gates one backend per build: compile it with" \
       " -DXLSTM_GATE_SSE2 or -DXLSTM_GATE_NEON."
#endif

#define XG_ALL(kernel)                          \
    extern "C" unsigned long XG(kernel, vector); \
    extern "C" unsigned long XG(kernel, scalar); \
    extern "C" unsigned long XG(kernel, tail)

XG_ALL(matvec_f32);
XG_ALL(matvec_s8);
XG_ALL(rank1_f32);
XG_ALL(vecmat_f32);

namespace {

const int kMaxRows = 20;
const int kMaxCols = 64;
const int kMaxH = 64;

/* 16-byte aligned, so +1, +2 and +3 floats are exactly the other three
 * alignments a 128-bit access can see, and 4 floats longer than the largest
 * shape so those views still end in bounds. Both backends load unaligned by
 * construction, which is what these four cases exist to demonstrate rather
 * than to work around. */
alignas(16) float g_M[kMaxRows * kMaxCols + 4];
alignas(16) float g_v[kMaxCols + 4];
float g_out[kMaxRows + 4];
float g_ref[kMaxRows + 4];

alignas(16) int8_t g_Mi[kMaxRows * kMaxCols + 16];
alignas(16) int8_t g_vi[kMaxCols + 16];
int32_t g_outi[kMaxRows];
int32_t g_refi[kMaxRows];

alignas(16) float g_C[kMaxH * kMaxH + 4];
float g_Cref[kMaxH * kMaxH];
alignas(16) float g_k[kMaxH + 4];
alignas(16) float g_kv[kMaxH + 4];
alignas(16) float g_vout[kMaxCols + 4];
float g_vref[kMaxCols + 4];

/* Seed for out[], so the checks also cover the contract's accumulate
 * semantics (out[i] += row . v) rather than only the product. */
float OutSeed(int i) { return 0.25f * (float)i; }

/* Deterministic, and neither constant nor symmetric - a lane-ordering defect
 * in a 4-wide or 8-wide load has to be able to show up. The int8 seeds reach
 * both extremes: -128 has no positive counterpart. */
void SeedFloats(void) {
    for (int i = 0; i < kMaxRows * kMaxCols + 4; ++i)
        g_M[i] = 0.5f - (float)((i * 37) % 71) / 70.0f;
    for (int j = 0; j < kMaxCols + 4; ++j)
        g_v[j] = (float)((j * 53) % 31) / 15.0f - 1.0f;
}

void SeedInts(void) {
    for (int i = 0; i < kMaxRows * kMaxCols + 16; ++i)
        g_Mi[i] = (int8_t)(((i * 37) % 255) - 128);
    for (int j = 0; j < kMaxCols + 16; ++j)
        g_vi[j] = (int8_t)(((j * 53) % 255) - 128);
}

/* Every check reports its counter split the same way. `want_tail` is the
 * third outcome: 1 when the call's width leaves a scalar remainder. */
bool CheckSplit(const char* what, const char* shape, unsigned long d_vec,
                unsigned long d_scalar, unsigned long d_tail,
                unsigned long want_vec, unsigned long want_tail) {
    if (d_vec == want_vec && d_scalar == 1ul - want_vec && d_tail == want_tail)
        return true;
    std::printf("  FAIL %s %s: expected vector+%lu scalar+%lu tail+%lu, got "
                "vector+%lu scalar+%lu tail+%lu. Which body a call takes is a "
                "property of its shape alone; a gate that cannot prove the "
                "vector instructions ran proves nothing, and one that cannot "
                "prove the SCALAR REMAINDER ran cannot tell a width that is "
                "fully vectorized from one that is only partly.\n",
                what, shape, want_vec, 1ul - want_vec, want_tail, d_vec,
                d_scalar, d_tail);
    return false;
}

/* --- f32 matvec ----------------------------------------------------------
 *
 * out[i] += sum_j M[i][j] * v[j], four columns per pass into a 4-lane
 * accumulator, horizontally summed at the end.
 *
 * This is the one kernel here that is NOT bit-exact against the scalar body,
 * and no rearrangement of the check can make it so: four lanes hold four
 * partial sums of the same terms, which is a different association than the
 * scalar body's single running accumulator seeded from out[i]. Both orders
 * are correct; they round differently.
 *
 * So this check is a tolerance, and the tolerance is the error bound for the
 * two orders rather than a number tuned until the run went green: the error
 * of a floating-point sum of n terms is at most about n * FLT_EPSILON times
 * the sum of the term magnitudes, and both orders sit inside that. Measured
 * over every shape and alignment below, the largest difference actually seen
 * is 2.4 * FLT_EPSILON * absum - so the bound leaves room for rounding and
 * none for a defect. Dropping one column of a 64-column row moves the result
 * by about absum / cols, four orders of magnitude above it.
 */
double TolMatvec(const float* row, const float* v, int cols, float seed) {
    double absum = std::fabs((double)seed);
    for (int j = 0; j < cols; ++j)
        absum += std::fabs((double)row[j] * (double)v[j]);
    return (double)(cols + 4) * (double)FLT_EPSILON * absum;
}

bool CheckMatvecF32(int rows, int cols, int moff, int voff, int ooff,
                    unsigned long want_vec, unsigned long want_tail) {
    const float* M = g_M + moff;
    const float* v = g_v + voff;
    float* out = g_out + ooff;
    const unsigned long v0 = XG(matvec_f32, vector);
    const unsigned long s0 = XG(matvec_f32, scalar);
    const unsigned long t0 = XG(matvec_f32, tail);
    char shape[80];
    bool ok = true;

    for (int i = 0; i < rows; ++i) out[i] = g_ref[i] = OutSeed(i);
    xlstm_matvec_f32(M, v, out, rows, cols);
    xlstm_scalar_matvec_f32(M, v, g_ref, rows, cols);

    std::snprintf(shape, sizeof shape, "rows=%d cols=%d M+%d v+%d out+%d",
                  rows, cols, moff, voff, ooff);
    ok &= CheckSplit("matvec_f32", shape,
                     XG(matvec_f32, vector) - v0,
                     XG(matvec_f32, scalar) - s0,
                     XG(matvec_f32, tail) - t0, want_vec, want_tail);

    for (int i = 0; i < rows; ++i) {
        const double tol = TolMatvec(M + (size_t)i * (size_t)cols, v, cols,
                                     OutSeed(i));
        const double diff = std::fabs((double)out[i] - (double)g_ref[i]);
        if (!(diff <= tol)) {
            std::printf("  FAIL matvec_f32 %s out[%d]: got %.9g, reference "
                        "%.9g (diff %.2e, bound %.2e). The lanes hold four "
                        "partial sums, so this is a rounding bound and not an "
                        "equality - but it is the bound for regrouping the "
                        "same terms, which no lost or duplicated term fits "
                        "inside.\n",
                        shape, i, (double)out[i], (double)g_ref[i], diff, tol);
            ok = false;
            break;
        }
    }
    return ok;
}

bool TestMatvecF32(void) {
    SeedFloats();

    /* Spelled out rather than recomputed from the kernel's own rule: a check
     * that derives the rule the same way the kernel does cannot fail when the
     * rule changes. cols straddles the four-float vector in every residue,
     * including the widths under it that get no acceleration at all. */
    static const struct { int rows, cols; unsigned long vec, tail; } kCases[] =
        {
            {0, 16, 0, 0}, {20, 0, 0, 0},          /* no work: no pass at all */
            {1, 1, 0, 1},  {3, 2, 0, 1},  {20, 3, 0, 1},  /* under the width */
            {1, 4, 1, 0},  {2, 8, 1, 0},  {8, 32, 1, 0},
            {20, 16, 1, 0}, {20, 64, 1, 0},
            {1, 5, 1, 1},  {7, 15, 1, 1}, {17, 17, 1, 1},
            {20, 7, 1, 1}, {20, 31, 1, 1}, {20, 63, 1, 1},
        };
    const int kCaseCount = (int)(sizeof kCases / sizeof kCases[0]);
    bool ok = true;

    for (int s = 0; s < kCaseCount; ++s) {
        for (int moff = 0; moff < 4; ++moff) {
            for (int voff = 0; voff < 4; ++voff) {
                for (int ooff = 0; ooff < 4; ++ooff) {
                    ok &= CheckMatvecF32(kCases[s].rows, kCases[s].cols, moff,
                                         voff, ooff, kCases[s].vec,
                                         kCases[s].tail);
                }
            }
        }
    }
    std::printf("  %d shapes x 64 alignment triples, all within the "
                "regrouping bound against xlstm_scalar_matvec_f32\n",
                kCaseCount);
    return ok;
}

/* --- INT8 matvec ---------------------------------------------------------
 *
 * out[i] = sum_j M[i][j] * (v[j] - v_zp), eight columns per pass: widen both
 * operands to int16, subtract the zero point there, multiply-accumulate into
 * int32.
 *
 * Integers, so summation order is free and the comparison is EQUALITY - any
 * difference at all is a defect. The zero-point sweep stops at +/-32640
 * because that is where this kernel stops being exact rather than where it
 * stops being interesting: the subtraction happens in an int16 lane, and one
 * step further v[j] - v_zp no longer fits it for some int8 v[j]. Unlike the
 * cortexm and esp backends there is no guard that returns those calls to the
 * scalar body, so the bound is a property of the kernel and is recorded here
 * rather than swept past. Every caller passes an int8 zero point.
 */
bool CheckMatvecS8(int rows, int cols, int32_t zp, int moff, int voff,
                   unsigned long want_vec, unsigned long want_tail) {
    const int8_t* M = g_Mi + moff;
    const int8_t* v = g_vi + voff;
    const unsigned long v0 = XG(matvec_s8, vector);
    const unsigned long s0 = XG(matvec_s8, scalar);
    const unsigned long t0 = XG(matvec_s8, tail);
    char shape[96];
    bool ok = true;

    /* A sentinel rather than zero: this contract overwrites out[] instead of
     * accumulating into it, so a row the kernel never wrote has to show up as
     * a mismatch and not as a plausible-looking 0. */
    for (int i = 0; i < rows; ++i) g_outi[i] = g_refi[i] = 0x5A5A5A5A;
    xlstm_matvec_s8(M, v, g_outi, rows, cols, zp);
    xlstm_scalar_matvec_s8(M, v, g_refi, rows, cols, zp);

    std::snprintf(shape, sizeof shape, "rows=%d cols=%d zp=%ld M+%d v+%d",
                  rows, cols, (long)zp, moff, voff);
    ok &= CheckSplit("matvec_s8", shape,
                     XG(matvec_s8, vector) - v0,
                     XG(matvec_s8, scalar) - s0,
                     XG(matvec_s8, tail) - t0, want_vec, want_tail);

    for (int i = 0; i < rows; ++i) {
        if (g_outi[i] != g_refi[i]) {
            std::printf("  FAIL matvec_s8 %s out[%d]: got %ld, reference %ld. "
                        "These are integers - the vector body regroups an "
                        "exact sum of exact products - so any difference at "
                        "all is a defect.\n",
                        shape, i, (long)g_outi[i], (long)g_refi[i]);
            ok = false;
            break;
        }
    }
    return ok;
}

bool TestMatvecS8(void) {
    SeedInts();

    static const struct { int rows, cols; long zp; unsigned long vec, tail; }
        kCases[] = {
            {20, 0, 0, 0, 0}, {0, 16, 0, 0, 0},    /* no work: no pass at all */
            {20, 1, 0, 0, 1},  {20, 7, 0, 0, 1},   /* under the width */
            {20, 8, 0, 1, 0},  {20, 16, 0, 1, 0},  {20, 64, 0, 1, 0},
            {1, 8, 0, 1, 0},   {3, 32, 0, 1, 0},
            {20, 9, 0, 1, 1},  {20, 15, 0, 1, 1},  {20, 17, 0, 1, 1},
            {20, 31, 0, 1, 1}, {20, 63, 0, 1, 1},  {17, 17, 0, 1, 1},
            {20, 17, -128, 1, 1},  /* what a non-negative tensor calibrates to */
            {20, 17, 127, 1, 1},   {20, 17, -127, 1, 1},
            {20, 16, -128, 1, 0},  {20, 64, -128, 1, 0},
            {20, 17, 254, 1, 1},   {20, 17, -254, 1, 1},
            {20, 17, 255, 1, 1},   {20, 17, -255, 1, 1},
            {20, 17, 32640, 1, 1}, {20, 17, -32640, 1, 1},
        };
    const int kCaseCount = (int)(sizeof kCases / sizeof kCases[0]);
    bool ok = true;

    for (int s = 0; s < kCaseCount; ++s) {
        for (int moff = 0; moff < 8; ++moff) {
            for (int voff = 0; voff < 8; ++voff) {
                ok &= CheckMatvecS8(kCases[s].rows, kCases[s].cols,
                                    (int32_t)kCases[s].zp, moff, voff,
                                    kCases[s].vec, kCases[s].tail);
            }
        }
    }
    std::printf("  %d cases x 64 alignment pairings, all bit-exact against "
                "xlstm_scalar_matvec_s8\n", kCaseCount);
    return ok;
}

/* --- rank-1 update -------------------------------------------------------
 *
 * C = f*C + i*k^T v, four elements per pass. Nothing sums across elements
 * here, so exactness is not about accumulation order - it is about
 * CONTRACTION. Two multiplies feed one add, and fusing either into the add
 * changes the last bit, so the vector body must round three times where the
 * scalar body does. That is why both are built -ffp-contract=off and why the
 * comparison is equality: a tolerance would swallow precisely this.
 *
 * The whole H x H is seeded and compared, not just the vectorized part, so a
 * store that ran past the end of a row into the start of the next would be
 * caught here rather than left to the golden vectors.
 */
bool CheckRank1(int rows, int cols, int coff, int koff, int voff, float f_gate,
                float i_gate, unsigned long want_vec, unsigned long want_tail) {
    float* C = g_C + coff;
    const float* k = g_k + koff;
    const float* v = g_kv + voff;
    const unsigned long v0 = XG(rank1_f32, vector);
    const unsigned long s0 = XG(rank1_f32, scalar);
    const unsigned long t0 = XG(rank1_f32, tail);
    char shape[80];
    bool ok = true;

    /* Mantissas with no short binary form, so that f*C + ik*v is not exactly
     * representable and a contraction difference has somewhere to show up. */
    for (int i = 0; i < rows * cols; ++i)
        C[i] = g_Cref[i] = 0.75f - (float)((i * 29) % 97) / 96.0f;
    xlstm_rank1_update_f32(C, f_gate, i_gate, k, v, rows, cols);
    xlstm_scalar_rank1_update_f32(g_Cref, f_gate, i_gate, k, v, rows, cols);

    std::snprintf(shape, sizeof shape, "%dx%d C+%d k+%d v+%d", rows, cols, coff,
                  koff, voff);
    ok &= CheckSplit("rank1_f32", shape,
                     XG(rank1_f32, vector) - v0,
                     XG(rank1_f32, scalar) - s0,
                     XG(rank1_f32, tail) - t0, want_vec, want_tail);

    for (int i = 0; i < rows * cols; ++i) {
        if (C[i] != g_Cref[i]) {
            std::printf("  FAIL rank1_f32 %s C[%d] (row %d col %d): got %.9g, "
                        "reference %.9g (diff %.2e). Nothing here sums across "
                        "elements - a difference is a lane order, a store that "
                        "ran into the next row, or a contraction that stopped "
                        "matching the scalar body.\n",
                        shape, i, i / cols, i % cols, (double)C[i],
                        (double)g_Cref[i], (double)std::fabs(C[i] - g_Cref[i]));
            ok = false;
            break;
        }
    }
    return ok;
}

bool TestRank1(void) {
    for (int i = 0; i < kMaxH + 4; ++i) {
        g_k[i] = 0.5f - (float)((i * 41) % 83) / 82.0f;
        g_kv[i] = (float)((i * 59) % 61) / 30.0f - 1.0f;
    }

    /* vec and tail are keyed on cols alone - it is the vectorised direction -
     * while rows only decides whether any work happens at all. A rectangular
     * shape is therefore not a new counter rule, which is the point: the same
     * rule has to keep holding once the two extents are free to differ. */
    static const struct { int rows, cols; unsigned long vec, tail; } kCases[] = {
        {0, 0, 0, 0},                                        /* no work */
        {1, 1, 0, 1},  {2, 2, 0, 1},  {3, 3, 0, 1},          /* under the width */
        {4, 4, 1, 0},  {8, 8, 1, 0},  {16, 16, 1, 0},
        {32, 32, 1, 0}, {64, 64, 1, 0},
        {5, 5, 1, 1},  {7, 7, 1, 1},  {9, 9, 1, 1},
        {17, 17, 1, 1}, {31, 31, 1, 1},
        /* Rectangular. Both orders, and a narrow one whose cols fall under the
         * vector width while its rows do not - the shape most likely to expose
         * a bound that is still counting the wrong extent. */
        {5, 13, 1, 1},  {13, 5, 1, 1},  {3, 16, 1, 0}, {16, 3, 0, 1},
        {8, 12, 1, 0},  {12, 8, 1, 0},  {1, 7, 1, 1},  {7, 1, 0, 1},
        {17, 4, 1, 0},  {4, 17, 1, 1},  {2, 64, 1, 0}, {64, 2, 0, 1},
    };
    /* One pair with both gates ordinary, one with a tiny i_gate, so the two
     * products differ in exponent by enough that the order they are combined
     * in matters. */
    static const struct { float f, i; } kGates[] = {
        {0.91371f, 0.13793f}, {0.99993f, 3.0517578e-05f},
    };
    const int kCaseCount = (int)(sizeof kCases / sizeof kCases[0]);
    const int kGateCount = (int)(sizeof kGates / sizeof kGates[0]);
    bool ok = true;

    for (int s = 0; s < kCaseCount; ++s) {
        for (int g = 0; g < kGateCount; ++g) {
            for (int coff = 0; coff < 4; ++coff) {
                for (int koff = 0; koff < 4; ++koff) {
                    for (int voff = 0; voff < 4; ++voff) {
                        ok &= CheckRank1(kCases[s].rows, kCases[s].cols, coff,
                                         koff, voff, kGates[g].f, kGates[g].i,
                                         kCases[s].vec, kCases[s].tail);
                    }
                }
            }
        }
    }
    std::printf("  %d shapes x 2 gate pairs x 64 alignment triples, all "
                "bit-exact against xlstm_scalar_rank1_update_f32\n",
                kCaseCount);
    return ok;
}

/* --- vecmat --------------------------------------------------------------
 *
 * q^T C, the other half of an mLSTM timestep, and the one kernel whose
 * natural vector dimension is already the scalar body's: out[j] is the
 * accumulator and j is the contiguous direction, so four columns of out[]
 * ride in one vector across the whole row loop. Every lane still starts at
 * its own out[j] and runs ascending i, so this is exact.
 *
 * Seeding out[] with something other than zero is what makes that checkable:
 * the contract's own callers zero it, so a body that dropped the accumulator
 * seed would pass every suite and fail only here.
 */
bool CheckVecmat(int rows, int cols, int moff, int qoff, int ooff,
                 unsigned long want_vec, unsigned long want_tail) {
    const float* M = g_M + moff;
    const float* q = g_v + qoff;
    float* out = g_vout + ooff;
    const unsigned long v0 = XG(vecmat_f32, vector);
    const unsigned long s0 = XG(vecmat_f32, scalar);
    const unsigned long t0 = XG(vecmat_f32, tail);
    char shape[80];
    bool ok = true;

    for (int j = 0; j < cols; ++j) out[j] = g_vref[j] = OutSeed(j);
    xlstm_vecmat_f32(q, M, out, rows, cols);
    xlstm_scalar_vecmat_f32(q, M, g_vref, rows, cols);

    std::snprintf(shape, sizeof shape, "rows=%d cols=%d M+%d q+%d out+%d",
                  rows, cols, moff, qoff, ooff);
    ok &= CheckSplit("vecmat_f32", shape,
                     XG(vecmat_f32, vector) - v0,
                     XG(vecmat_f32, scalar) - s0,
                     XG(vecmat_f32, tail) - t0, want_vec, want_tail);

    for (int j = 0; j < cols; ++j) {
        if (out[j] != g_vref[j]) {
            std::printf("  FAIL vecmat_f32 %s out[%d]: got %.9g, reference "
                        "%.9g (diff %.2e). Holding out[j] in a lane moves it "
                        "into a register, it does not regroup the adds into "
                        "it - this has to be exact.\n",
                        shape, j, (double)out[j], (double)g_vref[j],
                        (double)std::fabs(out[j] - g_vref[j]));
            ok = false;
            break;
        }
    }
    return ok;
}

bool TestVecmat(void) {
    /* Re-seeded rather than inherited from TestMatvecF32, so this check does
     * not depend on the order the four run in. */
    SeedFloats();

    static const struct { int rows, cols; unsigned long vec, tail; } kCases[] =
        {
            {20, 0, 0, 0}, {0, 8, 0, 0},           /* no work: no pass at all */
            {20, 1, 0, 1}, {20, 2, 0, 1}, {20, 3, 0, 1},  /* under the width */
            {20, 4, 1, 0}, {20, 8, 1, 0}, {20, 16, 1, 0}, {20, 64, 1, 0},
            {1, 8, 1, 0},  {3, 8, 1, 0},           /* fewer rows than lanes */
            {20, 5, 1, 1}, {20, 7, 1, 1}, {20, 17, 1, 1}, {20, 31, 1, 1},
            {1, 17, 1, 1}, {17, 17, 1, 1},
        };
    const int kCaseCount = (int)(sizeof kCases / sizeof kCases[0]);
    bool ok = true;

    for (int s = 0; s < kCaseCount; ++s) {
        for (int moff = 0; moff < 4; ++moff) {
            for (int qoff = 0; qoff < 4; ++qoff) {
                for (int ooff = 0; ooff < 4; ++ooff) {
                    ok &= CheckVecmat(kCases[s].rows, kCases[s].cols, moff,
                                      qoff, ooff, kCases[s].vec,
                                      kCases[s].tail);
                }
            }
        }
    }
    std::printf("  %d shapes x 64 alignment triples, all bit-exact against "
                "xlstm_scalar_vecmat_f32\n", kCaseCount);
    return ok;
}

bool Run(const char* name, bool (*fn)(void)) {
    std::printf("[ RUN      ] " XLSTM_GATE_NAME " %s\n", name);
    if (fn()) {
        std::printf("[       OK ] " XLSTM_GATE_NAME " %s\n", name);
        return true;
    }
    std::printf("[  FAILED  ] " XLSTM_GATE_NAME " %s\n", name);
    return false;
}

} /* namespace */

int main(void) {
    const char* backend = xlstm_simd_backend();
    int rc = 0;

    std::printf("[==========] Running " XLSTM_GATE_NAME " fast-path checks "
                "(backend=%s)\n", backend);

    /* An image that had silently linked src/xlstm_simd_ref.c would pass every
     * golden vector and prove nothing about this backend. The four suites are
     * built from the same SIMD object as this binary, so refusing here
     * refuses for the whole gate - and that failure can never read as a green
     * run. */
    if (std::strcmp(backend, XLSTM_GATE_NAME) != 0) {
        std::printf("FATAL: linked SIMD backend is \"%s\", not \""
                    XLSTM_GATE_NAME "\" - refusing to run. A pass here would "
                    "be a pass for the wrong backend.\n", backend);
        rc = 1;
    } else {
        if (!Run("fast path (matvec f32)", TestMatvecF32)) rc = 1;
        if (!Run("fast path (matvec int8)", TestMatvecS8)) rc = 1;
        if (!Run("fast path (rank-1 update)", TestRank1)) rc = 1;
        if (!Run("fast path (vecmat)", TestVecmat)) rc = 1;

        /* Reported, not asserted - the assertions are the four checks above.
         * This is here so a reader of a green log can see in one line that
         * every vector body really executed, and how many of those calls left
         * a scalar remainder behind. */
        std::printf("XLSTM_" XLSTM_GATE_UNAME "_FASTPATH: "
                    "matvec_f32 %lu vector (%lu with a scalar tail) / %lu "
                    "scalar, matvec_s8 %lu vector (%lu with a scalar tail) / "
                    "%lu scalar, rank1_f32 %lu vector (%lu with a scalar "
                    "tail) / %lu scalar, vecmat_f32 %lu vector (%lu with a "
                    "scalar tail) / %lu scalar.\n",
                    XG(matvec_f32, vector), XG(matvec_f32, tail),
                    XG(matvec_f32, scalar),
                    XG(matvec_s8, vector), XG(matvec_s8, tail),
                    XG(matvec_s8, scalar),
                    XG(rank1_f32, vector), XG(rank1_f32, tail),
                    XG(rank1_f32, scalar),
                    XG(vecmat_f32, vector), XG(vecmat_f32, tail),
                    XG(vecmat_f32, scalar));

        /* A backend that never entered a vector body at all would otherwise
         * leave all four checks trivially satisfied only if their tables also
         * said scalar everywhere. They do not - but assert the total anyway,
         * so that a future edit which weakened every table at once still has
         * to get past one number. */
        if (XG(matvec_f32, vector) == 0ul || XG(matvec_s8, vector) == 0ul ||
            XG(rank1_f32, vector) == 0ul || XG(vecmat_f32, vector) == 0ul ||
            XG(matvec_f32, tail) == 0ul || XG(matvec_s8, tail) == 0ul ||
            XG(rank1_f32, tail) == 0ul || XG(vecmat_f32, tail) == 0ul) {
            std::printf("FATAL: a vector body, or a scalar remainder, never "
                        "ran at all. Every kernel in this backend must reach "
                        "both on this gate's own shapes.\n");
            rc = 1;
        }
    }

    std::printf("[==========] " XLSTM_GATE_NAME " fast-path checks %s\n",
                rc ? "FAILED" : "passed");
    std::fflush(stdout);
    return rc;
}
