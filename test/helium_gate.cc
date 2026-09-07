/* Fast-path gate for the `helium` SIMD backend, run on an emulated Cortex-M55
 * as the fifth binary of `make test-helium`. The other four are the ordinary
 * golden-vector suites, cross-compiled unchanged.
 *
 * The suites prove the kernels compute the right numbers. They cannot prove
 * WHICH body computed them: a dispatch condition stuck at "always scalar"
 * passes every golden vector and accelerates nothing. On the esp backend that
 * was not hypothetical - its accelerated matvec was reached 6 times in 76
 * suite calls, by linker accident, and every suite stayed green. So the four
 * checks below assert, per contract function, that the path a call takes is
 * the one its shape dictates and that the result matches the shared scalar
 * body bit for bit.
 *
 * This backend claims one thing more than esp did, and the counters are built
 * to check it. MVE predicates per lane, so an odd size is meant to stay in the
 * vector body over a narrowed final pass rather than fall out to a scalar
 * remainder. Each kernel therefore reports THREE outcomes - vector, scalar,
 * and whether a narrowed pass ran - and every check below asserts all three.
 * Without the third, a kernel that quietly grew a scalar tail at H = 17 would
 * still look fully accelerated.
 * =========================================================================*/

#include "xlstm_simd.h"
/* The scalar bodies themselves, not a copy of them: the checks below compare
 * the accelerated kernels against the same text every backend is defined
 * against. */
#include "xlstm_simd_scalar.h"

#include <cmath>
#include <cstdio>
#include <cstring>

/* Defined in src/xlstm_simd_helium.c under XLSTM_HELIUM_FASTPATH_COUNTERS,
 * which the Makefile's test-helium target sets. Referenced unconditionally so
 * that losing the define is a link error rather than a gate that stops
 * checking. */
extern "C" unsigned long xlstm_helium_matvec_f32_vector;
extern "C" unsigned long xlstm_helium_matvec_f32_scalar;
extern "C" unsigned long xlstm_helium_matvec_f32_predicated;
extern "C" unsigned long xlstm_helium_matvec_s8_vector;
extern "C" unsigned long xlstm_helium_matvec_s8_scalar;
extern "C" unsigned long xlstm_helium_matvec_s8_predicated;
extern "C" unsigned long xlstm_helium_rank1_f32_vector;
extern "C" unsigned long xlstm_helium_rank1_f32_scalar;
extern "C" unsigned long xlstm_helium_rank1_f32_predicated;
extern "C" unsigned long xlstm_helium_vecmat_f32_vector;
extern "C" unsigned long xlstm_helium_vecmat_f32_scalar;
extern "C" unsigned long xlstm_helium_vecmat_f32_predicated;

namespace {

const int kMaxRows = 20;
const int kMaxCols = 64;
const int kMaxH = 64;

/* 16-byte aligned, so +1, +2 and +3 floats are exactly the other three
 * alignments a 128-bit vector can see, and 4 floats longer than the largest
 * shape so those views still end in bounds. MVE itself needs only 4-byte
 * alignment for a word access, which is what these four cases exist to
 * demonstrate rather than to work around. */
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
 * in a 4-wide gather or a 16-wide byte load has to be able to show up. The
 * int8 seeds reach both extremes: -128 has no positive counterpart. */
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

/* Every check reports its counter split the same way. `pred` is the third
 * outcome: 1 when the call was expected to end in a narrowed vector pass. */
bool CheckSplit(const char* what, const char* shape, unsigned long d_vec,
                unsigned long d_scalar, unsigned long d_pred,
                unsigned long want_vec, unsigned long want_pred) {
    if (d_vec == want_vec && d_scalar == 1ul - want_vec && d_pred == want_pred)
        return true;
    std::printf("  FAIL %s %s: expected vector+%lu scalar+%lu predicated+%lu, "
                "got vector+%lu scalar+%lu predicated+%lu. Which body a call "
                "takes is a property of its shape alone; a gate that cannot "
                "prove the vector instructions ran proves nothing, and one "
                "that cannot prove the NARROWED pass ran cannot tell an odd "
                "size that stayed in the vector body from one that fell out "
                "of it.\n",
                what, shape, want_vec, 1ul - want_vec, want_pred, d_vec,
                d_scalar, d_pred);
    return false;
}

/* --- f32 matvec ----------------------------------------------------------
 *
 * out[i] += sum_j M[i][j] * v[j], four ROWS per vector with a gather load.
 * Three properties are asserted, and the third is the reason for the design:
 *
 *   1. The vector body is entered on rows and cols alone. Alignment is swept
 *      but must never change the answer to "which body", because MVE word
 *      accesses need 4-byte alignment and a float already has it.
 *   2. A rows not divisible by four ends in a narrowed pass - the same loop
 *      with the spare lanes clamped onto the last row and discarded by a
 *      predicated store. There is no column tail at all: cols is not this
 *      kernel's vector dimension.
 *   3. Every result is bit-identical to xlstm_scalar_matvec_f32. This is an
 *      equality and not a tolerance on purpose. Lane r accumulates row r in
 *      ascending j, which is the scalar body's order exactly, so there is
 *      nothing for a tolerance to absorb except a defect - and the f32
 *      goldens have no room for one anyway.
 */

bool CheckMatvecF32(int rows, int cols, int moff, int voff, int ooff,
                    unsigned long want_vec, unsigned long want_pred) {
    const float* M = g_M + moff;
    const float* v = g_v + voff;
    float* out = g_out + ooff;
    const unsigned long v0 = xlstm_helium_matvec_f32_vector;
    const unsigned long s0 = xlstm_helium_matvec_f32_scalar;
    const unsigned long p0 = xlstm_helium_matvec_f32_predicated;
    char shape[80];
    bool ok = true;

    for (int i = 0; i < rows; ++i) out[i] = g_ref[i] = OutSeed(i);
    xlstm_matvec_f32(M, v, out, rows, cols);
    xlstm_scalar_matvec_f32(M, v, g_ref, rows, cols);

    std::snprintf(shape, sizeof shape, "rows=%d cols=%d M+%d v+%d out+%d",
                  rows, cols, moff, voff, ooff);
    ok &= CheckSplit("matvec_f32", shape,
                     xlstm_helium_matvec_f32_vector - v0,
                     xlstm_helium_matvec_f32_scalar - s0,
                     xlstm_helium_matvec_f32_predicated - p0,
                     want_vec, want_pred);

    for (int i = 0; i < rows; ++i) {
        if (out[i] != g_ref[i]) {
            std::printf("  FAIL matvec_f32 %s out[%d]: got %.9g, reference "
                        "%.9g (diff %.2e). Lane r sums row r in ascending j, "
                        "the scalar body's own order - this has to be "
                        "exact.\n",
                        shape, i, (double)out[i], (double)g_ref[i],
                        (double)std::fabs(out[i] - g_ref[i]));
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
     * rule changes. rows straddles the four-row block in every residue, and
     * cols is varied independently to show it is not a vector dimension here
     * - 1 and 17 must be as fully accelerated as 64. */
    static const struct { int rows, cols; unsigned long vec, pred; } kCases[] =
        {
            {0, 16, 0, 0}, {20, 0, 0, 0},          /* no work: scalar body */
            {4, 1, 1, 0},   {4, 3, 1, 0},   {4, 16, 1, 0},  {4, 17, 1, 0},
            {8, 64, 1, 0},  {16, 17, 1, 0}, {20, 64, 1, 0},
            {1, 16, 1, 1},  {2, 16, 1, 1},  {3, 16, 1, 1},  /* under a block */
            {5, 16, 1, 1},  {6, 17, 1, 1},  {7, 1, 1, 1},
            {17, 17, 1, 1}, {17, 64, 1, 1}, {19, 3, 1, 1},
        };
    const int kCaseCount = (int)(sizeof kCases / sizeof kCases[0]);
    bool ok = true;

    for (int s = 0; s < kCaseCount; ++s) {
        for (int moff = 0; moff < 4; ++moff) {
            for (int voff = 0; voff < 4; ++voff) {
                for (int ooff = 0; ooff < 4; ++ooff) {
                    ok &= CheckMatvecF32(kCases[s].rows, kCases[s].cols, moff,
                                         voff, ooff, kCases[s].vec,
                                         kCases[s].pred);
                }
            }
        }
    }
    std::printf("  %d shapes x 64 alignment triples, all bit-exact against "
                "xlstm_scalar_matvec_f32\n", kCaseCount);
    return ok;
}

/* --- INT8 matvec ---------------------------------------------------------
 *
 * out[i] = sum_j M[i][j] * (v[j] - v_zp), 16 columns per VMLADAVA.S8 with
 * VADDVA.S8 carrying the row sum the zero point folds into.
 *
 * Two things differ from the f32 check and both are the point of this
 * backend:
 *
 *   - The zero point is swept far outside int8 and must never change which
 *     body runs. It folds through an identity that holds in int32 for every
 *     zero point, so unlike xlstm_simd_cortexm.c (which leaves its DSP body
 *     above |z| = 32640) and xlstm_simd_esp.c (above |z| = 254) there is no
 *     bound here to be wrong about. The sweep stops at 65535 for the
 *     REFERENCE's sake, not this kernel's: xlstm_scalar_matvec_s8 overflows
 *     its own int32 accumulator somewhere above that at these column counts,
 *     and a gate must not compare against undefined behaviour.
 *   - Bit-exactness needs no rounding argument at all. These are integers.
 *     Any difference is a defect, so the comparison is equality.
 */

bool CheckMatvecS8(int rows, int cols, int32_t zp, int moff, int voff,
                   unsigned long want_vec, unsigned long want_pred) {
    const int8_t* M = g_Mi + moff;
    const int8_t* v = g_vi + voff;
    const unsigned long v0 = xlstm_helium_matvec_s8_vector;
    const unsigned long s0 = xlstm_helium_matvec_s8_scalar;
    const unsigned long p0 = xlstm_helium_matvec_s8_predicated;
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
                     xlstm_helium_matvec_s8_vector - v0,
                     xlstm_helium_matvec_s8_scalar - s0,
                     xlstm_helium_matvec_s8_predicated - p0,
                     want_vec, want_pred);

    for (int i = 0; i < rows; ++i) {
        if (g_outi[i] != g_refi[i]) {
            std::printf("  FAIL matvec_s8 %s out[%d]: got %ld, reference %ld. "
                        "These are integers - the vector body regroups an "
                        "exact sum of exact products and folds the zero point "
                        "through an exact identity, so any difference at all "
                        "is a defect.\n",
                        shape, i, (long)g_outi[i], (long)g_refi[i]);
            ok = false;
            break;
        }
    }
    return ok;
}

bool TestMatvecS8(void) {
    SeedInts();

    /* cols straddles the 16-column group in every interesting residue, and
     * the zero points run from the ones a quantizer produces out to values no
     * int8 lane could hold. */
    static const struct { int rows, cols; long zp; unsigned long vec, pred; }
        kCases[] = {
            {20, 0, 0, 0, 0}, {0, 16, 0, 0, 0},    /* no work: scalar body */
            {20, 16, 0, 1, 0},  {20, 32, 0, 1, 0},  {20, 64, 0, 1, 0},
            {20, 1, 0, 1, 1},   {20, 2, 0, 1, 1},   {20, 8, 0, 1, 1},
            {20, 15, 0, 1, 1},  {20, 17, 0, 1, 1},  {20, 31, 0, 1, 1},
            {20, 33, 0, 1, 1},  {20, 63, 0, 1, 1},
            {1, 17, 0, 1, 1},   {3, 17, 0, 1, 1},   /* odd row counts */
            {20, 17, -128, 1, 1},   /* what a non-negative tensor calibrates to */
            {20, 17, 127, 1, 1},    {20, 17, -127, 1, 1},
            {20, 16, -128, 1, 0},   {20, 64, -128, 1, 0},
            {20, 17, 254, 1, 1},    {20, 17, -254, 1, 1},
            {20, 17, 255, 1, 1},    {20, 17, -255, 1, 1},
            {20, 17, 32640, 1, 1},  {20, 17, -32640, 1, 1},
            {20, 17, 32641, 1, 1},  {20, 17, -32641, 1, 1},
            {20, 17, 65535, 1, 1},  {20, 17, -65535, 1, 1},
        };
    const int kCaseCount = (int)(sizeof kCases / sizeof kCases[0]);
    bool ok = true;

    for (int s = 0; s < kCaseCount; ++s) {
        for (int moff = 0; moff < 16; ++moff) {
            for (int voff = 0; voff < 16; ++voff) {
                ok &= CheckMatvecS8(kCases[s].rows, kCases[s].cols,
                                    (int32_t)kCases[s].zp, moff, voff,
                                    kCases[s].vec, kCases[s].pred);
            }
        }
    }
    std::printf("  %d cases x 256 alignment pairings, all bit-exact against "
                "xlstm_scalar_matvec_s8\n", kCaseCount);
    return ok;
}

/* --- rank-1 update -------------------------------------------------------
 *
 * C = f*C + i*k^T v, four elements per pass. Nothing sums across elements
 * here, so exactness is not about accumulation order - it is about
 * CONTRACTION. Two multiplies feed one add, and fusing either into the add
 * changes the last bit, so the vector body must spell VMUL, VMUL, VADD where
 * the scalar body rounds three times. Element-by-element equality is what
 * detects it if that ever stops holding; a tolerance would swallow precisely
 * this and nothing else.
 *
 * The predicated pass carries a second duty here that it does not elsewhere:
 * C is row-major H x H, so the lanes past column H of one row are the START
 * of the next row. An unpredicated store would overwrite them. The check
 * seeds the whole H x H and compares all of it, so that would be caught.
 */

bool CheckRank1(int H, int coff, int koff, int voff, float f_gate,
                float i_gate, unsigned long want_vec, unsigned long want_pred) {
    float* C = g_C + coff;
    const float* k = g_k + koff;
    const float* v = g_kv + voff;
    const unsigned long v0 = xlstm_helium_rank1_f32_vector;
    const unsigned long s0 = xlstm_helium_rank1_f32_scalar;
    const unsigned long p0 = xlstm_helium_rank1_f32_predicated;
    char shape[80];
    bool ok = true;

    /* Mantissas with no short binary form, so that f*C + ik*v is not exactly
     * representable and a contraction difference has somewhere to show up. */
    for (int i = 0; i < H * H; ++i)
        C[i] = g_Cref[i] = 0.75f - (float)((i * 29) % 97) / 96.0f;
    xlstm_rank1_update_f32(C, f_gate, i_gate, k, v, H, H);
    xlstm_scalar_rank1_update_f32(g_Cref, f_gate, i_gate, k, v, H, H);

    std::snprintf(shape, sizeof shape, "H=%d C+%d k+%d v+%d", H, coff, koff,
                  voff);
    ok &= CheckSplit("rank1_f32", shape,
                     xlstm_helium_rank1_f32_vector - v0,
                     xlstm_helium_rank1_f32_scalar - s0,
                     xlstm_helium_rank1_f32_predicated - p0,
                     want_vec, want_pred);

    for (int i = 0; i < H * H; ++i) {
        if (C[i] != g_Cref[i]) {
            std::printf("  FAIL rank1_f32 %s C[%d] (row %d col %d): got %.9g, "
                        "reference %.9g (diff %.2e). Nothing here sums across "
                        "elements - a difference is a lane order, a predicate "
                        "that let a store run into the next row, or a "
                        "contraction that stopped matching the scalar body.\n",
                        shape, i, i / H, i % H, (double)C[i],
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

    static const struct { int H; unsigned long vec, pred; } kCases[] = {
        {0, 0, 0},                                   /* no work */
        {4, 1, 0},  {8, 1, 0},  {16, 1, 0}, {32, 1, 0}, {64, 1, 0},
        {1, 1, 1},  {2, 1, 1},  {3, 1, 1},  {5, 1, 1},
        {7, 1, 1},  {9, 1, 1},  {17, 1, 1}, {31, 1, 1},
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
                        ok &= CheckRank1(kCases[s].H, coff, koff, voff,
                                         kGates[g].f, kGates[g].i,
                                         kCases[s].vec, kCases[s].pred);
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
 * ride in one vector across the whole row loop.
 *
 * Exactness here IS about summation order, unlike rank1 above. Every lane
 * still starts at its own out[j] and runs ascending i, which is the scalar
 * body's order. Seeding out[] with something other than zero is what makes
 * that checkable: the contract's own callers zero it, so a body that dropped
 * the seed would pass every suite and fail only here.
 */

bool CheckVecmat(int rows, int cols, int moff, int qoff, int ooff,
                 unsigned long want_vec, unsigned long want_pred) {
    const float* M = g_M + moff;
    const float* q = g_v + qoff;
    float* out = g_vout + ooff;
    const unsigned long v0 = xlstm_helium_vecmat_f32_vector;
    const unsigned long s0 = xlstm_helium_vecmat_f32_scalar;
    const unsigned long p0 = xlstm_helium_vecmat_f32_predicated;
    char shape[80];
    bool ok = true;

    for (int j = 0; j < cols; ++j) out[j] = g_vref[j] = OutSeed(j);
    xlstm_vecmat_f32(q, M, out, rows, cols);
    xlstm_scalar_vecmat_f32(q, M, g_vref, rows, cols);

    std::snprintf(shape, sizeof shape, "rows=%d cols=%d M+%d q+%d out+%d",
                  rows, cols, moff, qoff, ooff);
    ok &= CheckSplit("vecmat_f32", shape,
                     xlstm_helium_vecmat_f32_vector - v0,
                     xlstm_helium_vecmat_f32_scalar - s0,
                     xlstm_helium_vecmat_f32_predicated - p0,
                     want_vec, want_pred);

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

    static const struct { int rows, cols; unsigned long vec, pred; } kCases[] =
        {
            {20, 0, 0, 0}, {0, 8, 0, 0},           /* no work: scalar body */
            {20, 4, 1, 0},  {20, 8, 1, 0},  {20, 16, 1, 0}, {20, 64, 1, 0},
            {1, 8, 1, 0},   {3, 8, 1, 0},          /* fewer rows than lanes */
            {20, 1, 1, 1},  {20, 2, 1, 1},  {20, 3, 1, 1},
            {20, 5, 1, 1},  {20, 7, 1, 1},  {20, 17, 1, 1}, {20, 31, 1, 1},
            {1, 17, 1, 1},  {17, 17, 1, 1},
        };
    const int kCaseCount = (int)(sizeof kCases / sizeof kCases[0]);
    bool ok = true;

    for (int s = 0; s < kCaseCount; ++s) {
        for (int moff = 0; moff < 4; ++moff) {
            for (int qoff = 0; qoff < 4; ++qoff) {
                for (int ooff = 0; ooff < 4; ++ooff) {
                    ok &= CheckVecmat(kCases[s].rows, kCases[s].cols, moff,
                                      qoff, ooff, kCases[s].vec,
                                      kCases[s].pred);
                }
            }
        }
    }
    std::printf("  %d shapes x 64 alignment triples, all bit-exact against "
                "xlstm_scalar_vecmat_f32\n", kCaseCount);
    return ok;
}

/* --- no access past an operand's end -------------------------------------
 *
 * Everything above compares values, and there is one class of defect no
 * comparison can reach: a load whose lanes are discarded. matvec_f32 clamps
 * the spare lanes of its gather onto the last row, and the three others end
 * in a predicated pass whose inactive lanes make no access at all. Break any
 * of those and the ANSWERS ARE STILL RIGHT - the extra lanes never reach
 * out[] - while the kernel reads up to three rows, or fifteen bytes, past the
 * end of what the caller gave it. On a part with an MPU, or with a matrix
 * that ends at the end of a TCM, that is a fault in the field and nothing
 * here would have predicted it.
 *
 * So this check does not compare anything. It places one operand at a time
 * hard against the end of mapped memory and runs the kernel: an access past
 * the end is unmapped, which faults, which exits the image non-zero. The
 * shapes are all non-multiples of the vector width, because a shape that
 * divides evenly has no narrowed pass to get wrong.
 */

/* Placed by test/helium.ld so that the last byte of this array is the last
 * byte of mapped SRAM. */
__attribute__((section(".oob_tail"), used, aligned(16))) int8_t g_edge[8192];
const int kEdgeBytes = (int)sizeof g_edge;

/* An operand of `bytes` bytes ending exactly at the edge. Every size passed
 * here is a multiple of 4, so a float operand stays word-aligned. */
void* Edge(int bytes) { return g_edge + kEdgeBytes - bytes; }

/* Destinations that are NOT at the edge. Their own arrays, because the shapes
 * below run wider than the four checks above and so wider than g_out and
 * g_outi, which are sized for those. */
const int kEdgeMax = 32;
float g_eout[kEdgeMax];
int32_t g_eouti[kEdgeMax];

bool TestEdge(void) {
    /* Shapes whose last pass is a narrow one, in every residue. */
    static const int kOdd[] = {1, 2, 3, 5, 7, 15, 17, 31};
    const int kOddCount = (int)(sizeof kOdd / sizeof kOdd[0]);

    for (int i = 0; i < kEdgeBytes; ++i)
        g_edge[i] = (int8_t)(((i * 37) % 255) - 128);
    SeedFloats();

    for (int s = 0; s < kOddCount; ++s) {
        const int n = kOdd[s];
        const int nn = n * n;

        /* f32 matvec: rows is the vector dimension, so an unclamped gather
         * runs off the bottom of M and an unpredicated store runs off the end
         * of out[]. Both operands are placed at the edge in turn. */
        {
            float* eM = (float*)Edge(nn * (int)sizeof(float));
            float* eout = (float*)Edge(n * (int)sizeof(float));

            for (int i = 0; i < nn; ++i) eM[i] = 0.5f - (float)(i % 7);
            for (int i = 0; i < n; ++i) g_eout[i] = OutSeed(i);
            xlstm_matvec_f32(eM, g_v, g_eout, n, n);

            for (int i = 0; i < n; ++i) eout[i] = OutSeed(i);
            xlstm_matvec_f32(g_M, g_v, eout, n, n);
        }

        /* INT8 matvec: the tail group of M, and of v, is 16 columns wide. */
        {
            int8_t* eM = (int8_t*)Edge(nn);
            int8_t* ev = (int8_t*)Edge(n);

            for (int i = 0; i < nn; ++i) eM[i] = (int8_t)(i - 60);
            xlstm_matvec_s8(eM, g_vi, g_eouti, n, n, -128);

            for (int i = 0; i < n; ++i) ev[i] = (int8_t)(i - 60);
            xlstm_matvec_s8(g_Mi, ev, g_eouti, n, n, -128);
        }

        /* Rank-1 update writes as well as reads, so C at the edge covers the
         * store side of the narrowed pass and v at the edge covers its
         * load. */
        {
            float* eC = (float*)Edge(nn * (int)sizeof(float));
            float* ev = (float*)Edge(n * (int)sizeof(float));

            for (int i = 0; i < nn; ++i) eC[i] = 0.25f * (float)(i % 11);
            xlstm_rank1_update_f32(eC, 0.9f, 0.1f, g_k, g_kv, n, n);

            for (int i = 0; i < n; ++i) ev[i] = 0.5f - (float)(i % 5);
            xlstm_rank1_update_f32(g_C, 0.9f, 0.1f, g_k, ev, n, n);
        }

        /* vecmat reads the last columns of every row under a predicate; on
         * the final row those columns are the end of M. */
        {
            float* eM = (float*)Edge(nn * (int)sizeof(float));
            float* eout = (float*)Edge(n * (int)sizeof(float));

            for (int i = 0; i < nn; ++i) eM[i] = 0.5f - (float)(i % 7);
            for (int j = 0; j < n; ++j) g_eout[j] = OutSeed(j);
            xlstm_vecmat_f32(g_v, eM, g_eout, n, n);

            for (int j = 0; j < n; ++j) eout[j] = OutSeed(j);
            xlstm_vecmat_f32(g_v, g_M, eout, n, n);
        }
    }

    std::printf("  %d shapes x 4 kernels x 2 operand placements, no access "
                "past the end of mapped memory\n", kOddCount);
    return true; /* reaching here at all is the result: nothing faulted */
}

bool Run(const char* name, bool (*fn)(void)) {
    std::printf("[ RUN      ] helium %s\n", name);
    if (fn()) {
        std::printf("[       OK ] helium %s\n", name);
        return true;
    }
    std::printf("[  FAILED  ] helium %s\n", name);
    return false;
}

} /* namespace */

int main(void) {
    const char* backend = xlstm_simd_backend();
    int rc = 0;

    /* Unbuffered, because the out-of-bounds check below is meant to fail by
     * faulting, and the fault handler in test/helium_boot.c exits through
     * semihosting rather than through exit(): anything still in a stdio
     * buffer at that point is lost, including the [ RUN ] line that says
     * which check was running. */
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    std::printf("[==========] Running helium fast-path checks (backend=%s)\n",
                backend);

    /* An image that had silently linked src/xlstm_simd_ref.c would pass every
     * golden vector and prove nothing about this backend. The four suites are
     * built from the same xlstm_simd object as this binary, so refusing here
     * refuses for the whole gate - and that failure can never read as a green
     * run. */
    if (std::strcmp(backend, "helium") != 0) {
        std::printf("FATAL: linked SIMD backend is \"%s\", not \"helium\" - "
                    "refusing to run. A pass here would be a pass for the "
                    "wrong backend.\n", backend);
        rc = 1;
    } else {
        if (!Run("fast path (matvec f32, gather)", TestMatvecF32)) rc = 1;
        if (!Run("fast path (matvec int8)", TestMatvecS8)) rc = 1;
        if (!Run("fast path (rank-1 update)", TestRank1)) rc = 1;
        if (!Run("fast path (vecmat)", TestVecmat)) rc = 1;
        if (!Run("no access past an operand's end", TestEdge)) rc = 1;

        /* Reported, not asserted - the assertions are the four checks above.
         * This is here so a reader of a green log can see in one line that
         * every vector body really executed, and how many of those calls
         * ended in a narrowed pass rather than a scalar remainder. */
        std::printf("XLSTM_HELIUM_FASTPATH: matvec_f32 %lu vector (%lu "
                    "predicated) / %lu scalar, matvec_s8 %lu on VMLADAVA.S8 "
                    "(%lu predicated) / %lu scalar, rank1_f32 %lu vector (%lu "
                    "predicated) / %lu scalar, vecmat_f32 %lu vector (%lu "
                    "predicated) / %lu scalar.\n",
                    xlstm_helium_matvec_f32_vector,
                    xlstm_helium_matvec_f32_predicated,
                    xlstm_helium_matvec_f32_scalar,
                    xlstm_helium_matvec_s8_vector,
                    xlstm_helium_matvec_s8_predicated,
                    xlstm_helium_matvec_s8_scalar,
                    xlstm_helium_rank1_f32_vector,
                    xlstm_helium_rank1_f32_predicated,
                    xlstm_helium_rank1_f32_scalar,
                    xlstm_helium_vecmat_f32_vector,
                    xlstm_helium_vecmat_f32_predicated,
                    xlstm_helium_vecmat_f32_scalar);

        /* A backend that never entered a vector body at all would otherwise
         * leave all four checks trivially satisfied only if their tables also
         * said scalar everywhere. They do not - but assert the total anyway,
         * so that a future edit which weakened every table at once still has
         * to get past one number. */
        if (xlstm_helium_matvec_f32_vector == 0ul ||
            xlstm_helium_matvec_s8_vector == 0ul ||
            xlstm_helium_rank1_f32_vector == 0ul ||
            xlstm_helium_vecmat_f32_vector == 0ul ||
            xlstm_helium_matvec_f32_predicated == 0ul ||
            xlstm_helium_matvec_s8_predicated == 0ul ||
            xlstm_helium_rank1_f32_predicated == 0ul ||
            xlstm_helium_vecmat_f32_predicated == 0ul) {
            std::printf("FATAL: a vector body, or a narrowed pass, never ran "
                        "at all. Every kernel in this backend must reach both "
                        "on this gate's own shapes.\n");
            rc = 1;
        }
    }

    std::printf("[==========] helium fast-path checks %s\n",
                rc ? "FAILED" : "passed");
    std::fflush(stdout);
    return rc;
}
