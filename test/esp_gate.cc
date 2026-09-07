/* Fast-path gate for the `esp` SIMD backend, run on an emulated ESP32-S3 as
 * the fifth binary of `make test-esp`. The other four are the ordinary
 * golden-vector suites, cross-compiled unchanged.
 *
 * The suites prove the kernels compute the right numbers. They cannot prove
 * WHICH body computed them: a dispatch condition stuck at "always scalar"
 * passes every golden vector and accelerates nothing. That is what the four
 * checks below assert, one per contract function - the path a call takes is
 * a property of its shape, and every result matches the shared scalar body
 * bit for bit.
 *
 * Deliberately thin: no UART driver takeover, no trigger-byte handshake, no
 * timing pass, no clock guard, no provenance banner. Those belong to a rig
 * driving real silicon and measuring it. Nothing here needs them - stdout
 * arrives on the host through semihosting from the first byte, main's return
 * value becomes the emulator's exit status, and an emulated core has no
 * cycle count worth printing.
 * =========================================================================*/

#include "xlstm_simd.h"
/* The scalar bodies themselves, not a copy of them: the check below compares
 * the accelerated matvec against the same text every backend is defined
 * against. */
#include "xlstm_simd_scalar.h"

#include <cmath>
#include <cstdio>
#include <cstring>

/* Defined in src/xlstm_simd_esp.c under XLSTM_ESP_FASTPATH_COUNTERS, which
 * the Makefile's test-esp target sets. Referenced unconditionally so that
 * losing the define is a link error rather than a gate that stops
 * checking. */
extern "C" unsigned long xlstm_esp_matvec_f32_fast;
extern "C" unsigned long xlstm_esp_matvec_f32_scalar;
extern "C" unsigned long xlstm_esp_matvec_s8_fast;
extern "C" unsigned long xlstm_esp_matvec_s8_scalar;
extern "C" unsigned long xlstm_esp_rank1_f32_fast;
extern "C" unsigned long xlstm_esp_rank1_f32_scalar;
extern "C" unsigned long xlstm_esp_vecmat_f32_wide;
extern "C" unsigned long xlstm_esp_vecmat_f32_blocked;

namespace {

/* --- f32 fast-path check -------------------------------------------------
 *
 * The first of the two bodies this backend accelerates: the four-row,
 * 128-bit load blocked matvec inside xlstm_matvec_f32. Which calls reach it
 * is decided by rows and cols alone: at least 7 columns (a scalar prefix of up
 * to 3, then one whole 16-byte group) and at least one whole block of four
 * rows spaced 4 / gcd(cols, 4) apart. Where M and v landed does not enter
 * into it, and that is the property being checked here. The guard this
 * replaced asked for two 16-byte-aligned operands instead, and got them 6
 * times in 76 suite calls - by linker accident, so a relink could have taken
 * even those away and the suites would have gone on passing with no
 * accelerated coverage at all.
 *
 * So this does not rest on the suites. It runs every shape at all four
 * alignments of M and of v and fails the run unless all three hold:
 *
 *   1. the shapes the rule blocks took the blocked path, at every alignment,
 *   2. the shapes it cannot took the scalar body - without this, a guard
 *      stuck at "always fast" would look identical to a correct one,
 *   3. every result matched xlstm_scalar_matvec_f32 BIT FOR BIT. The blocked
 *      body reorders loads, not additions, so this is an equality and not a
 *      tolerance; a tolerance here would hide the one thing the body is
 *      written to avoid, and the f32 goldens have no room for it.
 */

const int kMaxRows = 20;
const int kMaxCols = 64;

/* 16-byte aligned, so +1, +2 and +3 floats are exactly the other three
 * alignments, and 4 floats longer than the largest shape so those views
 * still end in bounds. */
alignas(16) float g_M[kMaxRows * kMaxCols + 4];
alignas(16) float g_v[kMaxCols + 4];
float g_out[kMaxRows];
float g_ref[kMaxRows];

/* Seed for out[], so the check also covers the contract's accumulate
 * semantics (out[i] += row . v) rather than only the product. */
float OutSeed(int i) { return 0.25f * (float)i; }

bool CheckShape(int rows, int cols, int moff, int voff, bool expect_fast) {
    const float* M = g_M + moff;
    const float* v = g_v + voff;
    const unsigned long fast0 = xlstm_esp_matvec_f32_fast;
    const unsigned long scalar0 = xlstm_esp_matvec_f32_scalar;
    bool ok = true;

    for (int i = 0; i < rows; ++i) g_out[i] = g_ref[i] = OutSeed(i);
    xlstm_matvec_f32(M, v, g_out, rows, cols);
    xlstm_scalar_matvec_f32(M, v, g_ref, rows, cols);

    const unsigned long d_fast = xlstm_esp_matvec_f32_fast - fast0;
    const unsigned long d_scalar = xlstm_esp_matvec_f32_scalar - scalar0;
    const unsigned long want_fast = expect_fast ? 1ul : 0ul;
    if (d_fast != want_fast || d_scalar != 1ul - want_fast) {
        std::printf("  FAIL rows=%d cols=%d M+%d v+%d: expected the %s path, "
                    "got fast+%lu scalar+%lu. Which path a call takes is a "
                    "property of its shape; a gate that cannot prove the "
                    "128-bit load ran proves nothing.\n",
                    rows, cols, moff, voff,
                    expect_fast ? "blocked" : "scalar", d_fast, d_scalar);
        ok = false;
    }

    for (int i = 0; i < rows; ++i) {
        if (g_out[i] != g_ref[i]) {
            std::printf("  FAIL rows=%d cols=%d M+%d v+%d out[%d]: got %.9g, "
                        "reference %.9g (diff %.2e). The blocked body reorders "
                        "loads, not additions - this has to be exact.\n",
                        rows, cols, moff, voff, i, (double)g_out[i],
                        (double)g_ref[i],
                        (double)std::fabs(g_out[i] - g_ref[i]));
            ok = false;
            break;
        }
    }
    return ok;
}

bool TestFastPath(void) {
    /* Deterministic, and neither constant nor symmetric - a lane-ordering
     * defect in the 4-wide load has to be able to show up. */
    for (int i = 0; i < kMaxRows * kMaxCols + 4; ++i)
        g_M[i] = 0.5f - (float)((i * 37) % 71) / 70.0f;
    for (int j = 0; j < kMaxCols + 4; ++j)
        g_v[j] = (float)((j * 53) % 31) / 15.0f - 1.0f;

    /* Spelled out rather than recomputed from the kernel's own formula: a
     * check that derives the rule the same way the kernel does cannot fail
     * when the rule changes. cols straddles the 4-column group both ways,
     * and the last four entries vary the row count rather than the width. */
    static const struct { int rows, cols; bool fast; } kShapes[] = {
        {20, 1, false}, {20, 2, false}, {20, 3, false}, /* under one group */
        {20, 4, false}, {20, 6, false},  /* a group only at some alignments */
        {20, 7, true},  {20, 8, true},  {20, 9, true},
        {20, 16, true}, {20, 17, true}, {20, 64, true},
        {8, 17, false}, /* odd cols blocks 4 rows apart: 16 rows minimum */
        {8, 16, true},  {4, 15, false}, {4, 16, true},
    };
    const int kShapeCount = (int)(sizeof kShapes / sizeof kShapes[0]);
    bool ok = true;

    for (int s = 0; s < kShapeCount; ++s) {
        for (int moff = 0; moff < 4; ++moff) {
            for (int voff = 0; voff < 4; ++voff) {
                ok &= CheckShape(kShapes[s].rows, kShapes[s].cols, moff, voff,
                                 kShapes[s].fast);
            }
        }
    }
    std::printf("  %d shapes x 16 alignments, all bit-exact against "
                "xlstm_scalar_matvec_f32\n", kShapeCount);
    return ok;
}

/* --- INT8 fast-path check ------------------------------------------------
 *
 * Same three assertions as above, for the kernel that assembles each
 * 16-column group out of the two aligned blocks holding it and multiplies it
 * with EE.VMULAS.S8.ACCX. Two things differ from the f32 case and both are
 * worth being explicit about:
 *
 *   - Alignment does not enter the dispatch at all, so the expectation is
 *     the same at all 256 pairings of M and v. The f32 body has to block
 *     rows to share one scalar prefix; this one never seeks alignment, so a
 *     16-byte group does not cost it the odd sizes. Every column of every
 *     row of a vector-body call runs on the 16-lane MAC, H = 17 included.
 *   - What leaves the vector body is a cols of 0 or less, or a zero point
 *     two int8 lanes cannot carry (|v_zp| > 254). The zero point is folded
 *     into a constant vector rather than subtracted from v, so it is part of
 *     the dispatch here in a way it never is for f32.
 *
 * Bit-exactness is again an equality and not a tolerance, and here there is
 * not even a rounding argument to have: these are integers. The vector body
 * regroups an exact sum of exact products, so any difference at all is a
 * defect.
 */

alignas(16) int8_t g_Mi[kMaxRows * kMaxCols + 16];
alignas(16) int8_t g_vi[kMaxCols + 16];
int32_t g_outi[kMaxRows];
int32_t g_refi[kMaxRows];

bool CheckShapeS8(int rows, int cols, int32_t zp, int moff, int voff,
                  bool expect_fast) {
    const int8_t* M = g_Mi + moff;
    const int8_t* v = g_vi + voff;
    const unsigned long fast0 = xlstm_esp_matvec_s8_fast;
    const unsigned long scalar0 = xlstm_esp_matvec_s8_scalar;
    bool ok = true;

    /* A sentinel rather than zero: this contract overwrites out[] instead of
     * accumulating into it, so a row the kernel never wrote has to show up
     * as a mismatch and not as a plausible-looking 0. */
    for (int i = 0; i < rows; ++i) g_outi[i] = g_refi[i] = 0x5A5A5A5A;
    xlstm_matvec_s8(M, v, g_outi, rows, cols, zp);
    xlstm_scalar_matvec_s8(M, v, g_refi, rows, cols, zp);

    const unsigned long d_fast = xlstm_esp_matvec_s8_fast - fast0;
    const unsigned long d_scalar = xlstm_esp_matvec_s8_scalar - scalar0;
    const unsigned long want_fast = expect_fast ? 1ul : 0ul;
    if (d_fast != want_fast || d_scalar != 1ul - want_fast) {
        std::printf("  FAIL s8 rows=%d cols=%d zp=%ld M+%d v+%d: expected the "
                    "%s path, got fast+%lu scalar+%lu. Which path a call takes "
                    "is a property of its cols and zero point; a gate that "
                    "cannot prove EE.VMULAS.S8.ACCX ran proves nothing.\n",
                    rows, cols, (long)zp, moff, voff,
                    expect_fast ? "vector" : "scalar", d_fast, d_scalar);
        ok = false;
    }

    for (int i = 0; i < rows; ++i) {
        if (g_outi[i] != g_refi[i]) {
            std::printf("  FAIL s8 rows=%d cols=%d zp=%ld M+%d v+%d out[%d]: "
                        "got %ld, reference %ld. These are integers - the "
                        "vector body regroups an exact sum of exact products, "
                        "so any difference at all is a defect.\n",
                        rows, cols, (long)zp, moff, voff, i,
                        (long)g_outi[i], (long)g_refi[i]);
            ok = false;
            break;
        }
    }
    return ok;
}

bool TestFastPathS8(void) {
    /* Deterministic, asymmetric, and reaching both int8 extremes: -128 has
     * no positive counterpart, and a lane-ordering or sign defect has to be
     * able to show up. */
    for (int i = 0; i < kMaxRows * kMaxCols + 16; ++i)
        g_Mi[i] = (int8_t)(((i * 37) % 255) - 128);
    for (int j = 0; j < kMaxCols + 16; ++j)
        g_vi[j] = (int8_t)(((j * 53) % 255) - 128);

    /* Spelled out rather than recomputed from the kernel's own rule. cols
     * straddles the 16-column group both ways; the zero points straddle the
     * two-lane fold bound and include -128, which is what a tensor with no
     * negative values calibrates to and the one int8 value needing the
     * second lane. */
    static const struct { int rows, cols; long zp; bool fast; } kCases[] = {
        {20, 0, 0, false},           /* no columns at all */
        {20, 1, 0, true},   {20, 2, 0, true},   {20, 8, 0, true},
        {20, 15, 0, true},  {20, 16, 0, true},  {20, 17, 0, true},
        {20, 31, 0, true},  {20, 32, 0, true},  {20, 64, 0, true},
        {1, 17, 0, true},   {3, 17, 0, true},   /* fewer rows than a block */
        {20, 17, -128, true},        /* the split zero point */
        {20, 17, 127, true},  {20, 17, -127, true},
        {20, 64, -128, true}, {20, 16, -128, true}, {20, 1, -128, true},
        {20, 17, 254, true},  {20, 17, -254, true},  /* the fold bound */
        {20, 17, 255, false}, {20, 17, -255, false}, /* just outside it */
        {20, 17, 1000, false},
    };
    const int kCaseCount = (int)(sizeof kCases / sizeof kCases[0]);
    bool ok = true;

    for (int s = 0; s < kCaseCount; ++s) {
        for (int moff = 0; moff < 16; ++moff) {
            for (int voff = 0; voff < 16; ++voff) {
                ok &= CheckShapeS8(kCases[s].rows, kCases[s].cols,
                                   (int32_t)kCases[s].zp, moff, voff,
                                   kCases[s].fast);
            }
        }
    }
    std::printf("  %d cases x 256 alignment pairings, all bit-exact against "
                "xlstm_scalar_matvec_s8\n", kCaseCount);
    return ok;
}

/* --- rank-1 update fast-path check ---------------------------------------
 *
 * The mLSTM state update, C = f*C + i*k^T v, four rows at a time with
 * EE.LDF.128.IP and EE.STF.128.IP. Same three assertions as the two matvecs,
 * and two things about this one are worth stating outright:
 *
 *   - The dispatch is one number, H >= 7 - a prefix of up to three columns
 *     plus one whole group of four. Every row of every H at or above that
 *     runs on the 128-bit path, because a row a four-row block cannot reach
 *     takes its own prefix as a block of one. So the counters are checked
 *     against a rule with no alignment in it, and H = 0 must tick NEITHER
 *     counter: a fast tick is meant to say a wide move ran, not that a call
 *     with a large enough H arrived.
 *   - Exactness here is not about summation order, because nothing sums:
 *     every element is its own f*C + ik*v. What it IS about is CONTRACTION.
 *     MADD.S does not round the product it adds, so "which of the two
 *     multiplies became the madd" changes the last bit, and the vector body
 *     and the scalar body must make the same choice. Element-by-element
 *     equality is what detects it if they ever stop agreeing - a tolerance
 *     would swallow exactly this and nothing else.
 *
 * The store side gets its own reason to be checked by value: EE.STF.128.IP's
 * operand order is asserted, not documented, and getting it backwards writes
 * the four floats reversed rather than faulting.
 */

const int kMaxH = 64;

alignas(16) float g_C[kMaxH * kMaxH + 4];
float g_Cref[kMaxH * kMaxH];
alignas(16) float g_k[kMaxH + 4];
alignas(16) float g_kv[kMaxH + 4];

/* Mantissas with no short binary form, so that f*C + ik*v is not exactly
 * representable and a contraction difference has somewhere to show up. */
float CSeed(int i) { return 0.75f - (float)((i * 29) % 97) / 96.0f; }

bool CheckRank1(int H, int coff, int koff, int voff, float f_gate,
                float i_gate, unsigned long want_fast,
                unsigned long want_scalar) {
  float* C = g_C + coff;
  const float* k = g_k + koff;
  const float* v = g_kv + voff;
  const unsigned long fast0 = xlstm_esp_rank1_f32_fast;
  const unsigned long scalar0 = xlstm_esp_rank1_f32_scalar;
  bool ok = true;

  for (int i = 0; i < H * H; ++i) C[i] = g_Cref[i] = CSeed(i);
  xlstm_rank1_update_f32(C, f_gate, i_gate, k, v, H, H);
  xlstm_scalar_rank1_update_f32(g_Cref, f_gate, i_gate, k, v, H, H);

  const unsigned long d_fast = xlstm_esp_rank1_f32_fast - fast0;
  const unsigned long d_scalar = xlstm_esp_rank1_f32_scalar - scalar0;
  if (d_fast != want_fast || d_scalar != want_scalar) {
    std::printf("  FAIL rank1 H=%d C+%d k+%d v+%d: expected fast+%lu "
                "scalar+%lu, got fast+%lu scalar+%lu. Which path a call takes "
                "is a property of H alone; a gate that cannot prove "
                "EE.STF.128.IP ran proves nothing.\n",
                H, coff, koff, voff, want_fast, want_scalar, d_fast, d_scalar);
    ok = false;
  }

  for (int i = 0; i < H * H; ++i) {
    if (C[i] != g_Cref[i]) {
      std::printf("  FAIL rank1 H=%d C+%d k+%d v+%d C[%d] (row %d col %d): "
                  "got %.9g, reference %.9g (diff %.2e). Nothing here sums "
                  "across elements - a difference is a lane order or a "
                  "contraction that stopped matching the scalar body.\n",
                  H, coff, koff, voff, i, i / H, i % H, (double)C[i],
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

  /* Spelled out, not recomputed from the kernel's own rule. H straddles the
   * seven-column dispatch both ways and covers the three step classes an
   * odd, an even and a multiple-of-four H put the four-row block into. */
  static const struct { int H; unsigned long fast, scalar; } kCases[] = {
      {0, 0, 0},                                    /* neither body runs */
      {1, 0, 1}, {2, 0, 1}, {3, 0, 1}, {6, 0, 1},   /* under one group */
      {7, 1, 0},  {8, 1, 0},  {9, 1, 0},  {10, 1, 0},
      {15, 1, 0}, {16, 1, 0}, {17, 1, 0}, {20, 1, 0},
      {32, 1, 0}, {64, 1, 0},
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
            ok &= CheckRank1(kCases[s].H, coff, koff, voff, kGates[g].f,
                             kGates[g].i, kCases[s].fast, kCases[s].scalar);
          }
        }
      }
    }
  }
  std::printf("  %d shapes x 2 gate pairs x 64 alignment triples, all "
              "bit-exact against xlstm_scalar_rank1_update_f32\n", kCaseCount);
  return ok;
}

/* --- vecmat fast-path check ----------------------------------------------
 *
 * q^T C, the other half of an mLSTM timestep. This body has two spellings
 * rather than a fast one and a fallback, so the counters name what they are:
 *
 *   wide     four columns of out[] in registers across the row loop AND four
 *            of M per EE.LDF.128.IP. Reachable when one column boundary can
 *            be aligned for every row at once, which is a cols divisible by
 *            four - and at least seven of them, so a group is certain to fit
 *            behind a prefix of up to three.
 *   blocked  the same column blocking with scalar loads of M. Every other
 *            shape, at any alignment. It is not the shared scalar body and
 *            is not a fallback: it does the same work in fewer instructions,
 *            it just does not use a vector instruction to do it.
 *
 * An empty call - no rows or no columns - ticks neither, so a wide tick
 * always means a 128-bit load executed.
 *
 * Exactness here IS about summation order, unlike rank1 above: out[] is
 * read-modify-write and the blocking moves it into a register for the whole
 * row loop. Every accumulator still starts at its own out[j] and runs
 * ascending i, which is the scalar body's order exactly. Seeding out[] with
 * something other than zero is what makes that checkable - the contract's
 * own callers zero it, so a body that dropped the seed would pass every
 * suite and fail only here.
 */

alignas(16) float g_vout[kMaxCols + 4];
float g_vref[kMaxCols + 4];

bool CheckVecmat(int rows, int cols, int moff, int qoff, int ooff,
                 unsigned long want_wide, unsigned long want_blocked) {
  const float* M = g_M + moff;
  const float* qv = g_v + qoff;
  float* out = g_vout + ooff;
  const unsigned long wide0 = xlstm_esp_vecmat_f32_wide;
  const unsigned long blocked0 = xlstm_esp_vecmat_f32_blocked;
  bool ok = true;

  for (int j = 0; j < cols; ++j) out[j] = g_vref[j] = OutSeed(j);
  xlstm_vecmat_f32(qv, M, out, rows, cols);
  xlstm_scalar_vecmat_f32(qv, M, g_vref, rows, cols);

  const unsigned long d_wide = xlstm_esp_vecmat_f32_wide - wide0;
  const unsigned long d_blocked = xlstm_esp_vecmat_f32_blocked - blocked0;
  if (d_wide != want_wide || d_blocked != want_blocked) {
    std::printf("  FAIL vecmat rows=%d cols=%d M+%d q+%d out+%d: expected "
                "wide+%lu blocked+%lu, got wide+%lu blocked+%lu. Which "
                "spelling a call takes is a property of cols alone; a gate "
                "that cannot prove EE.LDF.128.IP ran proves nothing.\n",
                rows, cols, moff, qoff, ooff, want_wide, want_blocked,
                d_wide, d_blocked);
    ok = false;
  }

  for (int j = 0; j < cols; ++j) {
    if (out[j] != g_vref[j]) {
      std::printf("  FAIL vecmat rows=%d cols=%d M+%d q+%d out+%d out[%d]: "
                  "got %.9g, reference %.9g (diff %.2e). Blocking moves out[j] "
                  "into a register, it does not regroup the adds into it - "
                  "this has to be exact.\n",
                  rows, cols, moff, qoff, ooff, j, (double)out[j],
                  (double)g_vref[j], (double)std::fabs(out[j] - g_vref[j]));
      ok = false;
      break;
    }
  }
  return ok;
}

bool TestVecmat(void) {
  /* Re-seeded rather than inherited from TestFastPath, so this check does not
   * depend on the order the three run in. */
  for (int i = 0; i < kMaxRows * kMaxCols + 4; ++i)
    g_M[i] = 0.5f - (float)((i * 37) % 71) / 70.0f;
  for (int j = 0; j < kMaxCols + 4; ++j)
    g_v[j] = (float)((j * 53) % 31) / 15.0f - 1.0f;

  static const struct { int rows, cols; unsigned long wide, blocked; }
      kCases[] = {
      {20, 0, 0, 0}, {0, 8, 0, 0},   /* empty: neither spelling runs */
      {20, 1, 0, 1},  {20, 2, 0, 1},  {20, 3, 0, 1},
      {20, 4, 0, 1},  {20, 5, 0, 1},  {20, 6, 0, 1},  /* mult of 4, under 7 */
      {20, 7, 0, 1},  {20, 17, 0, 1}, {20, 31, 0, 1}, /* not a mult of 4 */
      {20, 8, 1, 0},  {20, 12, 1, 0}, {20, 16, 1, 0},
      {20, 32, 1, 0}, {20, 64, 1, 0},
      {1, 8, 1, 0},   {3, 8, 1, 0},   {1, 17, 0, 1},  /* fewer rows than 4 */
  };
  const int kCaseCount = (int)(sizeof kCases / sizeof kCases[0]);
  bool ok = true;

  for (int s = 0; s < kCaseCount; ++s) {
    for (int moff = 0; moff < 4; ++moff) {
      for (int qoff = 0; qoff < 4; ++qoff) {
        for (int ooff = 0; ooff < 4; ++ooff) {
          ok &= CheckVecmat(kCases[s].rows, kCases[s].cols, moff, qoff, ooff,
                            kCases[s].wide, kCases[s].blocked);
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
 * comparison can reach. The INT8 body assembles each 16-column window from
 * the two aligned blocks holding it, and its last group loads the upper block
 * only when the row's final byte falls in it. Load it unconditionally and the
 * ANSWERS ARE STILL RIGHT - the lanes above the row are multiplied by the
 * zero-padded constants and contribute nothing - while the kernel reads up to
 * 16 bytes past the end of what the caller gave it. That is the claim under
 * "NOTHING IS READ THAT CANNOT BE READ" in src/xlstm_simd_esp.c, and until
 * this check existed nothing tested it.
 *
 * So this check does not compare anything. It places one operand at a time
 * hard against the end of mapped memory and runs the kernel: an access past
 * the end is unmapped, which faults, which the toolchain's handler turns into
 * PANIC and a non-zero exit. Ported from test/helium_gate.cc, where the same
 * technique found two defects nothing else could.
 *
 * kEdgeTop is where the emulator's RAM ends, not where the chip's does:
 * Espressif's qemu-system-xtensa maps the esp32s3's data bus up to
 * 0x3FDF0000 and faults at or above it. The linker's DRAM window ends at
 * 0x3FCF0000 (its script's own limit, widened to the S3's 448 KB by
 * --defsym=entire_dram_seg=1), a megabyte below, so the top of that RAM holds
 * nothing this image put there and operands can be butted against it.
 *
 * What proves the constant is right is the mutation battery, not this file:
 * a kEdgeTop below the true end would leave every call below reading ordinary
 * RAM and passing, and test/mutants.py injects exactly that unconditional
 * upper-block load and fails if this check does not catch it.
 */

const uintptr_t kEdgeTop = 0x3FDF0000u;

void* Edge(int bytes) {
  return (void*)(kEdgeTop - (uintptr_t)bytes);
}

/* Destinations that are NOT at the edge. Their own arrays rather than the
 * checks' g_out and g_outi, which are sized for those checks' shapes. */
const int kEdgeMax = 32;
float g_eout[kEdgeMax];
int32_t g_eouti[kEdgeMax];

bool TestEdge(void) {
  /* Sizes that straddle the 16-column group in every residue, plus the two
   * multiples of it: a row whose last byte lands exactly on a block boundary
   * is the case where the upper block is the one place a load here could
   * leave the caller's buffer. */
  static const int kSizes[] = {1, 2, 3, 5, 7, 15, 16, 17, 31, 32};
  const int kSizeCount = (int)(sizeof kSizes / sizeof kSizes[0]);

  for (int i = 0; i < kMaxRows * kMaxCols + 4; ++i)
    g_M[i] = 0.5f - (float)((i * 37) % 71) / 70.0f;
  for (int j = 0; j < kMaxCols + 4; ++j)
    g_v[j] = (float)((j * 53) % 31) / 15.0f - 1.0f;
  for (int i = 0; i < kMaxRows * kMaxCols + 16; ++i)
    g_Mi[i] = (int8_t)(((i * 37) % 255) - 128);
  for (int j = 0; j < kMaxCols + 16; ++j)
    g_vi[j] = (int8_t)(((j * 53) % 255) - 128);

  for (int s = 0; s < kSizeCount; ++s) {
    const int n = kSizes[s];
    const int nn = n * n;

    /* INT8 matvec: M's last group and v's last group are both 16 columns
     * wide, and each is assembled from two blocks. */
    {
      int8_t* eM = (int8_t*)Edge(nn);
      int8_t* ev = (int8_t*)Edge(n);

      for (int i = 0; i < nn; ++i) eM[i] = (int8_t)(i - 60);
      xlstm_matvec_s8(eM, g_vi, g_eouti, n, n, -128);

      for (int i = 0; i < n; ++i) ev[i] = (int8_t)(i - 60);
      xlstm_matvec_s8(g_Mi, ev, g_eouti, n, n, -128);

      xlstm_matvec_s8(g_Mi, g_vi, (int32_t*)Edge(n * (int)sizeof(int32_t)),
                      n, n, -128);
    }

    /* f32 matvec: a scalar prefix to a 16-byte boundary and then whole
     * groups, so the last group is the one that could overrun. */
    {
      float* eM = (float*)Edge(nn * (int)sizeof(float));
      float* eout = (float*)Edge(n * (int)sizeof(float));

      for (int i = 0; i < nn; ++i) eM[i] = 0.5f - (float)(i % 7);
      for (int i = 0; i < n; ++i) g_eout[i] = OutSeed(i);
      xlstm_matvec_f32(eM, g_v, g_eout, n, n);

      for (int i = 0; i < n; ++i) eout[i] = OutSeed(i);
      xlstm_matvec_f32(g_M, g_v, eout, n, n);
    }

    /* Rank-1 update reads AND writes C in whole groups, so C at the edge
     * covers the store side and v at the edge the load. */
    {
      float* eC = (float*)Edge(nn * (int)sizeof(float));
      float* ev = (float*)Edge(n * (int)sizeof(float));

      for (int i = 0; i < nn; ++i) eC[i] = 0.25f * (float)(i % 11);
      xlstm_rank1_update_f32(eC, 0.9f, 0.1f, g_k, g_kv, n, n);

      for (int i = 0; i < n; ++i) ev[i] = 0.5f - (float)(i % 5);
      xlstm_rank1_update_f32(g_C, 0.9f, 0.1f, g_k, ev, n, n);
    }

    /* vecmat holds four columns of out[] in registers across the row loop,
     * and on the final row those columns are the end of M. */
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

  std::printf("  %d sizes x 4 kernels, each operand in turn at the edge, no "
              "access past the end of mapped memory\n", kSizeCount);
  return true; /* reaching here at all is the result: nothing faulted */
}

} /* namespace */

int main(void) {
    const char* backend = xlstm_simd_backend();
    int rc = 0;

    std::printf("[==========] Running esp fast-path checks (backend=%s)\n",
                backend);

    /* An image that had silently linked src/xlstm_simd_ref.c would pass every
     * golden vector and prove nothing about this backend. The four suites are
     * built from the same xlstm_simd object as this binary, so refusing here
     * refuses for the whole gate - and that failure can never read as a green
     * run. */
    if (std::strcmp(backend, "esp") != 0) {
        std::printf("FATAL: linked SIMD backend is \"%s\", not \"esp\" - "
                    "refusing to run. A pass here would be a pass for the "
                    "wrong backend.\n", backend);
        rc = 1;
    } else {
        std::printf("[ RUN      ] esp fast path\n");
        if (TestFastPath()) {
            std::printf("[       OK ] esp fast path\n");
        } else {
            std::printf("[  FAILED  ] esp fast path\n");
            rc = 1;
        }

        std::printf("[ RUN      ] esp fast path (int8)\n");
        if (TestFastPathS8()) {
            std::printf("[       OK ] esp fast path (int8)\n");
        } else {
            std::printf("[  FAILED  ] esp fast path (int8)\n");
            rc = 1;
        }

        std::printf("[ RUN      ] esp fast path (rank-1 update)\n");
        if (TestRank1()) {
            std::printf("[       OK ] esp fast path (rank-1 update)\n");
        } else {
            std::printf("[  FAILED  ] esp fast path (rank-1 update)\n");
            rc = 1;
        }

        std::printf("[ RUN      ] esp fast path (vecmat)\n");
        if (TestVecmat()) {
            std::printf("[       OK ] esp fast path (vecmat)\n");
        } else {
            std::printf("[  FAILED  ] esp fast path (vecmat)\n");
            rc = 1;
        }

        std::printf("[ RUN      ] esp no access past an operand's end\n");
        std::fflush(stdout);
        if (TestEdge()) {
            std::printf("[       OK ] esp no access past an operand's end\n");
        } else {
            std::printf("[  FAILED  ] esp no access past an operand's end\n");
            rc = 1;
        }

        /* Reported, not asserted - the assertions are the four checks above.
         * This is here so a reader of a green log can see, in one line, that
         * every one of the four vector bodies really executed, and how often
         * each check drove it rather than the scalar body it is compared
         * against. */
        std::printf("XLSTM_ESP_FASTPATH: matvec_f32 %lu blocked / %lu scalar, "
                    "matvec_s8 %lu on EE.VMULAS.S8.ACCX / %lu scalar, "
                    "rank1_f32 %lu on EE.LDF.128.IP + EE.STF.128.IP / %lu "
                    "scalar, vecmat_f32 %lu with a 128-bit load of M / %lu "
                    "blocked with scalar loads.\n",
                    xlstm_esp_matvec_f32_fast, xlstm_esp_matvec_f32_scalar,
                    xlstm_esp_matvec_s8_fast, xlstm_esp_matvec_s8_scalar,
                    xlstm_esp_rank1_f32_fast, xlstm_esp_rank1_f32_scalar,
                    xlstm_esp_vecmat_f32_wide, xlstm_esp_vecmat_f32_blocked);
    }

    std::printf("[==========] esp fast-path checks %s\n", rc ? "FAILED" : "passed");
    std::fflush(stdout);
    return rc;
}
