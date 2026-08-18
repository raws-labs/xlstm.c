/* Fast-path and out-of-bounds gate for the `cortexm` SIMD backend, run as the
 * fifth binary of `make test-cortexm`. The other four are the ordinary
 * golden-vector suites, cross-compiled unchanged.
 *
 * --- which body ran -------------------------------------------------------
 *
 * The suites prove the kernels compute the right numbers. They cannot prove
 * WHICH body computed them: a dispatch stuck at "always scalar", or an
 * eight-row block that no call ever reaches, passes every golden vector and
 * accelerates nothing. On the esp backend that was not hypothetical - its
 * accelerated matvec was reached 6 times in 76 suite calls, by linker
 * accident, and every suite stayed green. So the two accelerated kernels here
 * are checked against the counters in src/xlstm_simd_cortexm.c:
 *
 *   xlstm_matvec_s8   three outcomes, because the dispatch has three arms:
 *                     the word-load instance, the byte-assembled one, and the
 *                     scalar body it leaves for a zero point SXTAB16 cannot
 *                     fold. Which arm is a property of cols, the zero point
 *                     and the two operands' alignment, and of nothing else.
 *   xlstm_matvec_f32  whether an eight-row block ran, and whether anything was
 *                     left for the two-row and one-row tiers. Neither is a
 *                     scalar body - all three use fmaf - so what this asserts
 *                     is that the widest block is reached at the row counts
 *                     whose shape says it must be.
 *
 * The INT8 comparison is bit-exact: integers, so any difference is a defect.
 * The f32 comparison is NOT, and cannot be - fmaf rounds once per term where
 * the scalar body rounds twice, which is a deliberate numeric difference and
 * the source of that kernel's win. See TolMatvec below for the bound.
 *
 * --- no access past an operand's end --------------------------------------
 *
 * There is one class of defect no value comparison can reach: a load whose
 * bytes are then discarded. This backend reads its INT8 operands four bytes
 * at a time, and a group load that stepped one word too far would leave every
 * answer right - the extra bytes never reach out[] - while reading up to four
 * bytes past the end of what the caller gave it. On a part with an MPU, or
 * with a matrix that ends at the end of a TCM, that is a fault in the field
 * and no golden vector would have predicted it.
 *
 * So this check does not compare anything. It places one operand at a time
 * hard against a guard page and runs the kernel: an access past the end is
 * unmapped, which faults, which the handler below reports before exiting
 * non-zero. Same technique as test/helium_gate.cc, spelled for a hosted
 * target - there the edge is the end of mapped SRAM, placed by the linker
 * script; here it is mmap and mprotect.
 *
 * The guard is proved before it is trusted. A first probe reads one byte past
 * it and the run fails if that does NOT fault, because a guard page that
 * turned out to be ordinary memory would make every check below vacuous while
 * still reporting green.
 * =========================================================================*/

#include "xlstm_simd.h"
/* The scalar bodies themselves, not a copy of them: the checks below compare
 * the accelerated kernels against the same text every backend is defined
 * against. */
#include "xlstm_simd_scalar.h"

#include <cfloat>
#include <cmath>
#include <csetjmp>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/mman.h>
#include <unistd.h>

/* Defined in src/xlstm_simd_cortexm.c under XLSTM_CORTEXM_FASTPATH_COUNTERS,
 * which the Makefile sets for the SIMD object the tests link. Referenced
 * unconditionally so that losing the define is a link error rather than a
 * gate that stops checking. */
extern "C" unsigned long xlstm_cortexm_matvec_s8_aligned;
extern "C" unsigned long xlstm_cortexm_matvec_s8_unaligned;
extern "C" unsigned long xlstm_cortexm_matvec_s8_scalar;
extern "C" unsigned long xlstm_cortexm_matvec_f32_blocked;
extern "C" unsigned long xlstm_cortexm_matvec_f32_unblocked;
extern "C" unsigned long xlstm_cortexm_matvec_f32_narrow;

namespace {

/* --- the guard page ------------------------------------------------------
 *
 * kPages of readable scratch followed by one unmapped page. Edge(n) returns
 * the address of an n-byte operand whose last byte is the last readable one.
 */

/* 32 KB at the usual page size; the largest operand below is 20 x 64
 * floats. */
const size_t kPages = 8;

uint8_t* g_scratch = nullptr;
size_t g_span = 0;

/* Which operand was at the edge, for the handler to name. */
const char* volatile g_where = "(nothing)";

/* The handler has two jobs and they need different exits. While the guard
 * itself is being proved a fault is the expected outcome and has to be
 * survivable; during the checks a fault IS the failure, and reporting it from
 * the handler rather than unwinding keeps the report honest about where it
 * happened. Nothing but globals crosses the siglongjmp, and they are
 * volatile, so the recovering path has no indeterminate locals. */
volatile sig_atomic_t g_recover = 0;
volatile sig_atomic_t g_faulted = 0;
sigjmp_buf g_unwind;

/* write(2) rather than printf: a fault handler must not touch stdio. */
void SayFault(void) {
    static const char a[] = "  FAIL edge ";
    static const char b[] = ": the kernel accessed memory past the end of that"
                            " operand. No value changed - discarded bytes"
                            " never reach out[] - so nothing but an unmapped"
                            " page can see this.\n";
    ssize_t n;
    n = write(1, a, sizeof a - 1);
    n = write(1, g_where, std::strlen(g_where));
    n = write(1, b, sizeof b - 1);
    (void)n;
}

void OnFault(int sig) {
    (void)sig;
    g_faulted = 1;
    if (g_recover) {
        siglongjmp(g_unwind, 1);
    }
    SayFault();
    _exit(1);
}

bool MapScratch(void) {
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    void* p = mmap(nullptr, (kPages + 1) * page, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    struct sigaction sa;

    if (p == MAP_FAILED) {
        std::printf("  FAIL edge: mmap of the scratch region failed\n");
        return false;
    }
    g_scratch = (uint8_t*)p;
    g_span = kPages * page;
    if (mprotect(g_scratch + g_span, page, PROT_NONE) != 0) {
        std::printf("  FAIL edge: the guard page could not be unmapped\n");
        return false;
    }
    for (size_t i = 0; i < g_span; ++i)
        g_scratch[i] = (uint8_t)((i * 37) % 255);

    std::memset(&sa, 0, sizeof sa);
    sa.sa_handler = OnFault;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGSEGV, &sa, nullptr) != 0 ||
        sigaction(SIGBUS, &sa, nullptr) != 0) {
        std::printf("  FAIL edge: the fault handler could not be installed\n");
        return false;
    }
    return true;
}

void* Edge(size_t bytes) { return g_scratch + g_span - bytes; }

bool TestGuard(void) {
    g_where = "the guard page itself";
    g_faulted = 0;
    g_recover = 1;
    if (sigsetjmp(g_unwind, 1) == 0) {
        volatile uint8_t sink = *(const volatile uint8_t*)(g_scratch + g_span);
        (void)sink;
    }
    g_recover = 0;
    if (!g_faulted) {
        std::printf("  FAIL edge: reading one byte past the scratch region "
                    "did not fault, so every check below would pass whatever "
                    "the kernels read. The guard is the whole mechanism.\n");
        return false;
    }
    std::printf("  guard proved: one byte past the scratch region faults\n");
    return true;
}

/* --- operands that are NOT at the edge -----------------------------------
 *
 * Word-aligned with four bytes of slack, so that +1, +2 and +3 are the other
 * three alignments the INT8 dispatch can see and the shifted views still end
 * in bounds. The float arrays keep four elements of slack for the same
 * reason, though alignment decides nothing in the f32 kernel - showing that
 * it does not is part of what the check below is for.
 */

const int kMaxRows = 20;
const int kMaxCols = 64;
const int kMaxH = 32;

alignas(16) float g_M[kMaxRows * kMaxCols + 4];
alignas(16) float g_v[kMaxCols + 4];
float g_out[(kMaxRows > kMaxCols ? kMaxRows : kMaxCols) + 4];
float g_ref[(kMaxRows > kMaxCols ? kMaxRows : kMaxCols) + 4];
alignas(16) int8_t g_Mi[kMaxRows * kMaxCols + 4];
alignas(16) int8_t g_vi[kMaxCols + 4];
int32_t g_outi[kMaxRows];
int32_t g_refi[kMaxRows];
float g_C[kMaxH * kMaxH];
float g_k[kMaxH];
float g_kv[kMaxH];

void Seed(void) {
    for (int i = 0; i < kMaxRows * kMaxCols + 4; ++i) {
        g_M[i] = 0.5f - (float)((i * 37) % 71) / 70.0f;
        g_Mi[i] = (int8_t)(((i * 37) % 255) - 128);
    }
    for (int j = 0; j < kMaxCols + 4; ++j) {
        g_v[j] = (float)((j * 53) % 31) / 15.0f - 1.0f;
        g_vi[j] = (int8_t)(((j * 53) % 255) - 128);
    }
    for (int i = 0; i < kMaxH * kMaxH; ++i) g_C[i] = 0.25f * (float)(i % 11);
    for (int i = 0; i < kMaxH; ++i) {
        g_k[i] = 0.5f - (float)((i * 41) % 83) / 82.0f;
        g_kv[i] = (float)((i * 59) % 61) / 30.0f - 1.0f;
    }
}

/* --- which body ran: INT8 matvec -----------------------------------------
 *
 * Three arms, and the check asserts which one a call takes as well as what it
 * computed. The comparison is equality: these are integers, and the DSP body
 * folds the zero point through SXTAB16's halfword lanes exactly for every
 * zero point it does not hand back to the scalar body, so any difference is a
 * defect. The sweep runs past that hand-back point on purpose - the arm the
 * kernel picks there is part of what is being asserted.
 */

enum S8Arm { kS8Scalar = 0, kS8Aligned = 1, kS8Unaligned = 2 };

bool CheckSplitS8(const char* shape, unsigned long d_al, unsigned long d_un,
                  unsigned long d_sc, S8Arm want) {
    const unsigned long w_al = want == kS8Aligned ? 1ul : 0ul;
    const unsigned long w_un = want == kS8Unaligned ? 1ul : 0ul;
    const unsigned long w_sc = want == kS8Scalar ? 1ul : 0ul;

    if (d_al == w_al && d_un == w_un && d_sc == w_sc) return true;
    std::printf("  FAIL matvec_s8 %s: expected aligned+%lu unaligned+%lu "
                "scalar+%lu, got aligned+%lu unaligned+%lu scalar+%lu. Which "
                "body a call takes is a property of cols, the zero point and "
                "the two operands' alignment alone; a gate that cannot prove "
                "the SXTAB16 + SMLAD loop ran proves nothing.\n",
                shape, w_al, w_un, w_sc, d_al, d_un, d_sc);
    return false;
}

bool CheckMatvecS8(int rows, int cols, int32_t zp, int moff, int voff,
                   S8Arm want) {
    const int8_t* M = g_Mi + moff;
    const int8_t* v = g_vi + voff;
    const unsigned long a0 = xlstm_cortexm_matvec_s8_aligned;
    const unsigned long u0 = xlstm_cortexm_matvec_s8_unaligned;
    const unsigned long s0 = xlstm_cortexm_matvec_s8_scalar;
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
    ok &= CheckSplitS8(shape, xlstm_cortexm_matvec_s8_aligned - a0,
                       xlstm_cortexm_matvec_s8_unaligned - u0,
                       xlstm_cortexm_matvec_s8_scalar - s0, want);

    for (int i = 0; i < rows; ++i) {
        if (g_outi[i] != g_refi[i]) {
            std::printf("  FAIL matvec_s8 %s out[%d]: got %ld, reference %ld. "
                        "These are integers - the DSP body sums exact products "
                        "in a different order and folds an exact zero point - "
                        "so any difference at all is a defect.\n",
                        shape, i, (long)g_outi[i], (long)g_refi[i]);
            ok = false;
            break;
        }
    }
    return ok;
}

bool TestMatvecS8(void) {
    /* Spelled out rather than recomputed from the kernel's own rule: a check
     * that derives the rule the same way the kernel does cannot fail when the
     * rule changes. Each case names the arm its shape dictates when both
     * operands sit on a word boundary; the loop below applies the one further
     * fact that an operand off a word boundary cannot take the word-load
     * instance. Zero points run out past +/-32640, where SXTAB16's halfword
     * lanes would wrap and the kernel hands the call back to the scalar body;
     * the sweep stops at 65535 for the REFERENCE's sake, since
     * xlstm_scalar_matvec_s8 overflows its own int32 accumulator somewhere
     * above that at these column counts. */
    static const struct { int rows, cols; long zp; S8Arm arm; } kCases[] = {
        {20, 0, 0, kS8Scalar}, {20, -1, 0, kS8Scalar},   /* no columns */
        {0, 16, 0, kS8Aligned}, {1, 16, 0, kS8Aligned},
        {20, 4, 0, kS8Aligned}, {20, 16, 0, kS8Aligned},
        {20, 32, 0, kS8Aligned}, {20, 64, 0, kS8Aligned},
        {20, 1, 0, kS8Unaligned}, {20, 2, 0, kS8Unaligned},
        {20, 3, 0, kS8Unaligned}, {20, 7, 0, kS8Unaligned},
        {20, 17, 0, kS8Unaligned}, {20, 31, 0, kS8Unaligned},
        {17, 17, 0, kS8Unaligned}, {1, 63, 0, kS8Unaligned},
        {20, 17, -128, kS8Unaligned},  /* what a non-negative tensor gives */
        {20, 16, -128, kS8Aligned}, {20, 64, 127, kS8Aligned},
        {20, 17, 127, kS8Unaligned}, {20, 17, -127, kS8Unaligned},
        {20, 17, 254, kS8Unaligned}, {20, 17, -255, kS8Unaligned},
        {20, 16, 32640, kS8Aligned}, {20, 17, -32640, kS8Unaligned},
        {20, 16, 32641, kS8Scalar}, {20, 17, -32641, kS8Scalar},
        {20, 16, 65535, kS8Scalar}, {20, 17, -65535, kS8Scalar},
    };
    const int kCaseCount = (int)(sizeof kCases / sizeof kCases[0]);
    bool ok = true;

    for (int s = 0; s < kCaseCount; ++s) {
        for (int moff = 0; moff < 4; ++moff) {
            for (int voff = 0; voff < 4; ++voff) {
                S8Arm want = kCases[s].arm;
                if (want == kS8Aligned && ((moff | voff) & 3) != 0)
                    want = kS8Unaligned;
                ok &= CheckMatvecS8(kCases[s].rows, kCases[s].cols,
                                    (int32_t)kCases[s].zp, moff, voff, want);
            }
        }
    }
    std::printf("  %d cases x 16 alignment pairings, all bit-exact against "
                "xlstm_scalar_matvec_s8\n", kCaseCount);
    return ok;
}

/* --- which body ran: f32 matvec ------------------------------------------
 *
 * out[i] += sum_j M[i][j] * v[j], eight rows per block so that v[j] is loaded
 * once and consumed by all eight. Two outcomes are counted: whether a block
 * ran at all, and whether anything was left over for the two-row and one-row
 * tiers. Neither is a scalar body - all three use fmaf - so what is asserted
 * is that the widest block is reached wherever the row count allows it.
 *
 * The comparison is a tolerance and not an equality, and that is a property
 * of the kernel rather than a concession by the gate: fmaf does not round the
 * product before adding it, so this body rounds once per term where the
 * scalar body rounds twice. That difference IS the win. The bound below is
 * the error bound for a sum of n terms - about n * FLT_EPSILON times the sum
 * of the term magnitudes - which both orders sit inside, and which no lost or
 * duplicated term fits inside: dropping one column of a 64-column row moves
 * the result by about absum / cols, four orders of magnitude above it.
 */
double TolMatvec(const float* row, const float* v, int cols, float seed) {
    double absum = std::fabs((double)seed);
    for (int j = 0; j < cols; ++j)
        absum += std::fabs((double)row[j] * (double)v[j]);
    return (double)(cols + 4) * (double)FLT_EPSILON * absum;
}

float OutSeed(int i) { return 0.25f * (float)i; }

bool CheckMatvecF32(int rows, int cols, int moff, int voff, int ooff,
                    unsigned long want_blocked, unsigned long want_narrow) {
    const float* M = g_M + moff;
    const float* v = g_v + voff;
    float* out = g_out + ooff;
    const unsigned long b0 = xlstm_cortexm_matvec_f32_blocked;
    const unsigned long u0 = xlstm_cortexm_matvec_f32_unblocked;
    const unsigned long n0 = xlstm_cortexm_matvec_f32_narrow;
    char shape[80];
    bool ok = true;

    for (int i = 0; i < rows; ++i) out[i] = g_ref[i] = OutSeed(i);
    xlstm_matvec_f32(M, v, out, rows, cols);
    xlstm_scalar_matvec_f32(M, v, g_ref, rows, cols);

    std::snprintf(shape, sizeof shape, "rows=%d cols=%d M+%d v+%d out+%d",
                  rows, cols, moff, voff, ooff);
    {
        const unsigned long d_b = xlstm_cortexm_matvec_f32_blocked - b0;
        const unsigned long d_u = xlstm_cortexm_matvec_f32_unblocked - u0;
        const unsigned long d_n = xlstm_cortexm_matvec_f32_narrow - n0;
        if (d_b != want_blocked || d_u != 1ul - want_blocked ||
            d_n != want_narrow) {
            std::printf("  FAIL matvec_f32 %s: expected blocked+%lu "
                        "unblocked+%lu narrow+%lu, got blocked+%lu "
                        "unblocked+%lu narrow+%lu. Whether the eight-row block "
                        "runs is a property of rows and cols alone; a gate "
                        "that cannot prove it ran proves nothing about the "
                        "one thing this kernel is blocked for.\n",
                        shape, want_blocked, 1ul - want_blocked, want_narrow,
                        d_b, d_u, d_n);
            ok = false;
        }
    }

    for (int i = 0; i < rows; ++i) {
        const double tol = TolMatvec(M + (size_t)i * (size_t)cols, v, cols,
                                     OutSeed(i));
        const double diff = std::fabs((double)out[i] - (double)g_ref[i]);
        if (!(diff <= tol)) {
            std::printf("  FAIL matvec_f32 %s out[%d]: got %.9g, reference "
                        "%.9g (diff %.2e, bound %.2e). fmaf rounds once per "
                        "term where the scalar body rounds twice, so this is "
                        "a rounding bound and not an equality - but it is the "
                        "bound for the same terms, which no lost or duplicated "
                        "term fits inside.\n",
                        shape, i, (double)out[i], (double)g_ref[i], diff, tol);
            ok = false;
            break;
        }
    }
    return ok;
}

bool TestMatvecF32(void) {
    /* rows straddles the eight-row block in every residue that matters, and
     * cols is varied independently to show it is not a blocking dimension
     * here - only a cols of 0 or less keeps the block from running. */
    static const struct { int rows, cols; unsigned long blocked, narrow; }
        kCases[] = {
            {0, 16, 0, 0},                          /* no rows: neither tier */
            {20, 0, 0, 1}, {20, -1, 0, 1},          /* no columns: no block */
            {1, 16, 0, 1}, {2, 17, 0, 1}, {7, 64, 0, 1},   /* under a block */
            {8, 1, 1, 0},  {8, 16, 1, 0}, {16, 17, 1, 0}, {16, 64, 1, 0},
            {9, 17, 1, 1}, {15, 3, 1, 1}, {17, 17, 1, 1}, {20, 64, 1, 1},
        };
    const int kCaseCount = (int)(sizeof kCases / sizeof kCases[0]);
    bool ok = true;

    for (int s = 0; s < kCaseCount; ++s) {
        for (int moff = 0; moff < 4; ++moff) {
            for (int voff = 0; voff < 4; ++voff) {
                for (int ooff = 0; ooff < 4; ++ooff) {
                    ok &= CheckMatvecF32(kCases[s].rows, kCases[s].cols, moff,
                                         voff, ooff, kCases[s].blocked,
                                         kCases[s].narrow);
                }
            }
        }
    }
    std::printf("  %d shapes x 64 alignment triples, all within the fmaf "
                "rounding bound against xlstm_scalar_matvec_f32\n",
                kCaseCount);
    return ok;
}

/* --- the edge placements -------------------------------------------------
 *
 * Shapes cover both instances of the INT8 body: a cols divisible by four with
 * word-aligned operands takes the word-typed group load and has no scalar
 * remainder at all, which is the instance that can walk off the end of a row;
 * anything else takes the byte-assembled load and a tail of 1 to 3 columns.
 * Edge(n) is page-aligned minus n, so a multiple of four leaves an operand
 * word-aligned and a non-multiple does not - both are wanted.
 */

struct Shape { int rows, cols; };

const Shape kShapes[] = {
    {1, 1},  {2, 3},   {3, 4},   {4, 7},   {5, 8},   {7, 16},
    {8, 17}, {16, 16}, {17, 17}, {3, 31},  {20, 64}, {2, 32},
};
const int kShapeCount = (int)(sizeof kShapes / sizeof kShapes[0]);

const int kH[] = {1, 2, 3, 4, 7, 8, 15, 16, 17, 31};
const int kHCount = (int)(sizeof kH / sizeof kH[0]);

/* A fault inside `call` never returns here - the handler reports and exits -
 * so reaching the end of this function IS the result. */
#define EDGE_RUN(what, call) do { g_where = (what); call; } while (0)

void RunEdgeCases(void) {
    for (int s = 0; s < kShapeCount; ++s) {
        const int rows = kShapes[s].rows;
        const int cols = kShapes[s].cols;
        const size_t nf = (size_t)rows * (size_t)cols * sizeof(float);

        /* INT8 matvec: M by group, v by group, one int32 of out[] per row. */
        EDGE_RUN("matvec_s8 M",
                 xlstm_matvec_s8((const int8_t*)Edge((size_t)rows *
                                                     (size_t)cols),
                                 g_vi, g_outi, rows, cols, -128));
        EDGE_RUN("matvec_s8 v",
                 xlstm_matvec_s8(g_Mi, (const int8_t*)Edge((size_t)cols),
                                 g_outi, rows, cols, -128));
        EDGE_RUN("matvec_s8 out",
                 xlstm_matvec_s8(g_Mi, g_vi,
                                 (int32_t*)Edge((size_t)rows *
                                                sizeof(int32_t)),
                                 rows, cols, -128));

        /* f32 matvec: eight rows at a time, v walked once per block. */
        for (int i = 0; i < rows; ++i) g_out[i] = 0.25f * (float)i;
        EDGE_RUN("matvec_f32 M",
                 xlstm_matvec_f32((const float*)Edge(nf), g_v, g_out, rows,
                                  cols));
        EDGE_RUN("matvec_f32 v",
                 xlstm_matvec_f32(g_M,
                                  (const float*)Edge((size_t)cols *
                                                     sizeof(float)),
                                  g_out, rows, cols));
        {
            float* eout = (float*)Edge((size_t)rows * sizeof(float));
            for (int i = 0; i < rows; ++i) eout[i] = 0.25f * (float)i;
            EDGE_RUN("matvec_f32 out",
                     xlstm_matvec_f32(g_M, g_v, eout, rows, cols));
        }

        /* vecmat: out[] accumulates over rows, so its last columns are read
         * and written on every row of the call. */
        for (int j = 0; j < cols; ++j) g_out[j] = 0.25f * (float)j;
        EDGE_RUN("vecmat_f32 M",
                 xlstm_vecmat_f32(g_v, (const float*)Edge(nf), g_out, rows,
                                  cols));
        {
            float* eout = (float*)Edge((size_t)cols * sizeof(float));
            for (int j = 0; j < cols; ++j) eout[j] = 0.25f * (float)j;
            EDGE_RUN("vecmat_f32 out",
                     xlstm_vecmat_f32(g_v, g_M, eout, rows, cols));
        }
    }

    /* Rank-1 update writes as well as reads, so C at the edge covers the
     * store side and v at the edge the load. */
    for (int h = 0; h < kHCount; ++h) {
        const int H = kH[h];
        float* eC = (float*)Edge((size_t)H * (size_t)H * sizeof(float));

        for (int i = 0; i < H * H; ++i) eC[i] = 0.25f * (float)(i % 11);
        EDGE_RUN("rank1_f32 C",
                 xlstm_rank1_update_f32(eC, 0.9f, 0.1f, g_k, g_kv, H));
        EDGE_RUN("rank1_f32 v",
                 xlstm_rank1_update_f32(g_C, 0.9f, 0.1f, g_k,
                                        (const float*)Edge((size_t)H *
                                                           sizeof(float)),
                                        H));
    }

    std::printf("  %d shapes x 8 placements + %d widths x 2, no access past "
                "the end of any operand\n", kShapeCount, kHCount);
}

bool Run(const char* name, bool (*fn)(void)) {
    std::printf("[ RUN      ] cortexm %s\n", name);
    if (fn()) {
        std::printf("[       OK ] cortexm %s\n", name);
        return true;
    }
    std::printf("[  FAILED  ] cortexm %s\n", name);
    return false;
}

} /* namespace */

int main(void) {
    const char* backend = xlstm_simd_backend();
    int rc = 0;

    /* Unbuffered, because the fault handler reports through write(2) and then
     * _exit(2): anything still sitting in a stdio buffer at that point is
     * lost, including the banner that says the image ran at all. */
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    std::printf("[==========] Running cortexm fast-path and out-of-bounds "
                "checks (backend=%s)\n", backend);

    /* An image that had silently linked src/xlstm_simd_ref.c would read
     * nothing past anything and prove nothing about this backend. The four
     * suites are built from the same SIMD object as this binary, so refusing
     * here refuses for the whole gate. */
    if (std::strcmp(backend, "cortexm") != 0) {
        std::printf("FATAL: linked SIMD backend is \"%s\", not \"cortexm\" - "
                    "refusing to run. A pass here would be a pass for the "
                    "wrong backend.\n", backend);
        rc = 1;
    } else {
        Seed();
        if (!Run("fast path (matvec int8)", TestMatvecS8)) rc = 1;
        if (!Run("fast path (matvec f32)", TestMatvecF32)) rc = 1;

        std::printf("[ RUN      ] cortexm no access past an operand's end\n");
        if (MapScratch() && TestGuard()) {
            RunEdgeCases();
            std::printf("[       OK ] cortexm no access past an operand's "
                        "end\n");
        } else {
            std::printf("[  FAILED  ] cortexm no access past an operand's "
                        "end\n");
            rc = 1;
        }

        /* Reported, not asserted - the assertions are the two checks above.
         * This is here so a reader of a green log can see in one line that
         * the DSP loop and the eight-row block really executed, and how the
         * calls divided between the arms. */
        std::printf("XLSTM_CORTEXM_FASTPATH: matvec_s8 %lu on SXTAB16 + SMLAD "
                    "with word loads / %lu with byte-assembled loads / %lu "
                    "scalar, matvec_f32 %lu of %lu calls ran an eight-row "
                    "block and %lu left rows for a narrower tier.\n",
                    xlstm_cortexm_matvec_s8_aligned,
                    xlstm_cortexm_matvec_s8_unaligned,
                    xlstm_cortexm_matvec_s8_scalar,
                    xlstm_cortexm_matvec_f32_blocked,
                    xlstm_cortexm_matvec_f32_blocked +
                        xlstm_cortexm_matvec_f32_unblocked,
                    xlstm_cortexm_matvec_f32_narrow);

        /* A backend that never entered an accelerated body at all would
         * otherwise leave both checks trivially satisfied only if their tables
         * also said scalar everywhere. They do not - but assert the totals
         * anyway, so that a future edit which weakened both tables at once
         * still has to get past these numbers. */
        if (xlstm_cortexm_matvec_s8_aligned == 0ul ||
            xlstm_cortexm_matvec_s8_unaligned == 0ul ||
            xlstm_cortexm_matvec_s8_scalar == 0ul ||
            xlstm_cortexm_matvec_f32_blocked == 0ul ||
            xlstm_cortexm_matvec_f32_narrow == 0ul) {
            std::printf("FATAL: an arm of the INT8 dispatch, or the eight-row "
                        "f32 block, never ran at all. Every one of them must "
                        "be reached on this gate's own shapes.\n");
            rc = 1;
        }
    }

    std::printf("[==========] cortexm fast-path and out-of-bounds checks %s\n",
                rc ? "FAILED" : "passed");
    std::fflush(stdout);
    return rc;
}
