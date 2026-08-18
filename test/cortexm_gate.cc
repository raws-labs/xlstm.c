/* Out-of-bounds gate for the `cortexm` SIMD backend, run as the fifth binary
 * of `make test-cortexm`. The other four are the ordinary golden-vector
 * suites, cross-compiled unchanged.
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
 *
 * Deliberately narrow: this gate asserts nothing about WHICH body a call
 * took. The esp and helium backends carry counters for that; adding them here
 * would be a change to a shipping kernel rather than to its gate.
 * =========================================================================*/

#include "xlstm_simd.h"

#include <csetjmp>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/mman.h>
#include <unistd.h>

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

/* --- operands that are NOT at the edge ----------------------------------- */

const int kMaxRows = 20;
const int kMaxCols = 64;
const int kMaxH = 32;

float g_M[kMaxRows * kMaxCols];
float g_v[kMaxCols];
float g_out[kMaxRows > kMaxCols ? kMaxRows : kMaxCols];
int8_t g_Mi[kMaxRows * kMaxCols];
int8_t g_vi[kMaxCols];
int32_t g_outi[kMaxRows];
float g_C[kMaxH * kMaxH];
float g_k[kMaxH];
float g_kv[kMaxH];

void Seed(void) {
    for (int i = 0; i < kMaxRows * kMaxCols; ++i) {
        g_M[i] = 0.5f - (float)((i * 37) % 71) / 70.0f;
        g_Mi[i] = (int8_t)(((i * 37) % 255) - 128);
    }
    for (int j = 0; j < kMaxCols; ++j) {
        g_v[j] = (float)((j * 53) % 31) / 15.0f - 1.0f;
        g_vi[j] = (int8_t)(((j * 53) % 255) - 128);
    }
    for (int i = 0; i < kMaxH * kMaxH; ++i) g_C[i] = 0.25f * (float)(i % 11);
    for (int i = 0; i < kMaxH; ++i) {
        g_k[i] = 0.5f - (float)((i * 41) % 83) / 82.0f;
        g_kv[i] = (float)((i * 59) % 61) / 30.0f - 1.0f;
    }
}

/* --- the checks ----------------------------------------------------------
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

} /* namespace */

int main(void) {
    const char* backend = xlstm_simd_backend();
    int rc = 0;

    /* Unbuffered, because the fault handler reports through write(2) and then
     * _exit(2): anything still sitting in a stdio buffer at that point is
     * lost, including the banner that says the image ran at all. */
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    std::printf("[==========] Running cortexm out-of-bounds checks "
                "(backend=%s)\n", backend);

    /* An image that had silently linked src/xlstm_simd_ref.c would read
     * nothing past anything and prove nothing about this backend. The four
     * suites are built from the same xlstm_simd object as this binary, so
     * refusing here refuses for the whole gate. */
    if (std::strcmp(backend, "cortexm") != 0) {
        std::printf("FATAL: linked SIMD backend is \"%s\", not \"cortexm\" - "
                    "refusing to run. A pass here would be a pass for the "
                    "wrong backend.\n", backend);
        rc = 1;
    } else {
        Seed();
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
    }

    std::printf("[==========] cortexm out-of-bounds checks %s\n",
                rc ? "FAILED" : "passed");
    std::fflush(stdout);
    return rc;
}
