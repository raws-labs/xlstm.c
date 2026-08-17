/* Entry point for the emulated ESP32-S3 gate (`make test-esp`).
 *
 * Runs the four golden-vector suites against the `esp` SIMD backend on a
 * QEMU-emulated ESP32-S3, plus the two fast-path checks below, and prints
 * one sentinel line that ../qemu_gate.sh turns into the container's exit
 * code.
 *
 * Deliberately thin: no UART driver takeover, no trigger-byte handshake, no
 * timing pass, no clock guard, no provenance banner. Those belong to a rig
 * driving real silicon and measuring it. Nothing here needs them - QEMU's
 * serial output is captured from the first byte, and an emulated core has
 * no cycle count worth printing.
 * =========================================================================*/

#include "test_config.h"
#include "xlstm_simd.h"
/* The scalar bodies themselves, not a copy of them: the check below compares
 * the accelerated matvec against the same text every backend is defined
 * against. */
#include "xlstm_simd_scalar.h"

#include <cmath>
#include <cstdio>
#include <cstring>

/* The four suite entry points, renamed per translation unit by
 * main/CMakeLists.txt. All five files are C++, so these link straight
 * against the renamed definitions with no header in between. */
extern int slstm_test_main(void);
extern int mlstm_test_main(void);
extern int slstm_s8_test_main(void);
extern int mlstm_s8_test_main(void);

/* Defined in src/xlstm_simd_esp.c under XLSTM_ESP_FASTPATH_COUNTERS, which
 * main/CMakeLists.txt sets. Referenced unconditionally so that losing the
 * define is a link error rather than a gate that stops checking. */
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
  xlstm_rank1_update_f32(C, f_gate, i_gate, k, v, H);
  xlstm_scalar_rank1_update_f32(g_Cref, f_gate, i_gate, k, v, H);

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

} /* namespace */

extern "C" void app_main(void) {
    const char* backend = xlstm_simd_backend();
    int rc = 0;

    std::printf("XLSTM_ESP_GATE: backend=%s test_max_h=%d\n",
                backend, XLSTM_TEST_MAX_H);

    /* An image that had silently linked src/xlstm_simd_ref.c would pass
     * every suite below and prove nothing about this backend. Refuse before
     * running anything, so that failure can never read as a green run. */
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

        const unsigned long fast0 = xlstm_esp_matvec_f32_fast;
        const unsigned long scalar0 = xlstm_esp_matvec_f32_scalar;
        const unsigned long qfast0 = xlstm_esp_matvec_s8_fast;
        const unsigned long qscalar0 = xlstm_esp_matvec_s8_scalar;
        const unsigned long rfast0 = xlstm_esp_rank1_f32_fast;
        const unsigned long rscalar0 = xlstm_esp_rank1_f32_scalar;
        const unsigned long vwide0 = xlstm_esp_vecmat_f32_wide;
        const unsigned long vblocked0 = xlstm_esp_vecmat_f32_blocked;

        rc |= slstm_test_main();
        rc |= mlstm_test_main();
        rc |= slstm_s8_test_main();
        rc |= mlstm_s8_test_main();

        /* Reported, not asserted. This number is now a property of the case
         * list - every call with 7 or more columns is blocked, and the rest
         * are the H and I of 1 to 4 - so asserting it would only pin down
         * reference_data.h, which is not what this gate is for. TestFastPath
         * above is the assertion; this is here so a reader of a green log
         * can see how much of the run was accelerated, and what the calls
         * that were not have in common. */
        const unsigned long fast = xlstm_esp_matvec_f32_fast - fast0;
        const unsigned long scalar = xlstm_esp_matvec_f32_scalar - scalar0;
        const unsigned long qfast = xlstm_esp_matvec_s8_fast - qfast0;
        const unsigned long qscalar = xlstm_esp_matvec_s8_scalar - qscalar0;
        const unsigned long rfast = xlstm_esp_rank1_f32_fast - rfast0;
        const unsigned long rscalar = xlstm_esp_rank1_f32_scalar - rscalar0;
        const unsigned long vwide = xlstm_esp_vecmat_f32_wide - vwide0;
        const unsigned long vblocked = xlstm_esp_vecmat_f32_blocked - vblocked0;
        std::printf("XLSTM_ESP_FASTPATH: the suites called xlstm_matvec_f32 "
                    "%lu times, %lu of them blocked (the rest are under 7 "
                    "columns wide), and xlstm_matvec_s8 %lu times, %lu of "
                    "them on EE.VMULAS.S8.ACCX - and a vector-body INT8 call "
                    "runs every column of every row there, at any alignment. "
                    "xlstm_rank1_update_f32 ran %lu times, %lu of them on "
                    "EE.LDF.128.IP + EE.STF.128.IP (the rest are H under 7), "
                    "and xlstm_vecmat_f32 %lu times, %lu of them with a "
                    "128-bit load of M - the other %lu keep the same column "
                    "blocking with scalar loads, which is every width not "
                    "divisible by four.\n",
                    fast + scalar, fast, qfast + qscalar, qfast,
                    rfast + rscalar, rfast, vwide + vblocked, vwide, vblocked);
    }

    std::printf("##xlstm-esp-gate:%d##\n", rc ? 1 : 0);
    std::fflush(stdout);
}
