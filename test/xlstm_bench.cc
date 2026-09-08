/* xlstm.c benchmark - measures throughput of all 4 kernels across sizes.
 *
 * Build & run:
 *   make bench          # auto-detect SIMD backend
 *   make bench-ref      # force scalar reference
 *   make bench-sse2     # force SSE2
 *
 * Fixed-work mode, used by the performance gate (`make perf`):
 *   xlstm_bench <kernel> <H> <steps>
 * runs exactly one case for exactly <steps> steps and prints nothing. No
 * calibration, no clock: the work done is a function of the arguments alone,
 * so an instruction count taken over it is reproducible.
 * =========================================================================*/

#include "xlstm.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <time.h>

// ============================================================================
// LCG pseudo-random (deterministic, no <random>)
// ============================================================================

static uint32_t g_lcg = 42;

static uint32_t lcg_next() {
    g_lcg = g_lcg * 1664525u + 1013904223u;
    return g_lcg;
}

static float rand_f32(float lo, float hi) {
    return lo + (hi - lo) * (float)(lcg_next() & 0xFFFF) / 65535.0f;
}

static void fill_f32(float* buf, int n, float lo, float hi) {
    for (int i = 0; i < n; i++) buf[i] = rand_f32(lo, hi);
}



// ============================================================================
// Timing
// ============================================================================

static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// ============================================================================
// Auto-scale: find iteration count that takes ~200ms
// ============================================================================

static const double TARGET_MS = 200.0;

template <typename Fn>
int calibrate(Fn fn) {
    int iters = 64;
    for (;;) {
        double t0 = now_ms();
        for (int i = 0; i < iters; i++) fn();
        double elapsed = now_ms() - t0;
        if (elapsed >= TARGET_MS * 0.5) {
            int scaled = (int)(iters * TARGET_MS / elapsed);
            return scaled < 64 ? 64 : scaled;
        }
        iters *= 4;
    }
}

template <typename Fn>
double measure(Fn fn, int iters) {
    double t0 = now_ms();
    for (int i = 0; i < iters; i++) fn();
    return now_ms() - t0;
}

// ============================================================================
// Kernel benchmarks
// ============================================================================

static void bench_slstm_f32(int H, int steps) {
    const int I = H;
    float* W = (float*)malloc(4 * H * I * sizeof(float));
    float* R = (float*)malloc(4 * H * H * sizeof(float));
    float* b = (float*)malloc(4 * H * sizeof(float));
    float* x = (float*)malloc(I * sizeof(float));
    float* y = (float*)calloc(H, sizeof(float));
    float* c = (float*)calloc(H, sizeof(float));
    float* n = (float*)calloc(H, sizeof(float));
    float* m = (float*)calloc(H, sizeof(float));
    float* scratch = (float*)calloc(4 * H, sizeof(float));

    fill_f32(W, 4 * H * I, -0.5f, 0.5f);
    fill_f32(R, 4 * H * H, -0.5f, 0.5f);
    fill_f32(b, 4 * H, -0.1f, 0.1f);
    fill_f32(x, I, -1.0f, 1.0f);

    SlstmParams params = {0.0f};

    auto step = [&]() {
        slstm_step_f32(x, W, R, b, y, c, n, m, scratch, I, H, &params);
    };

    int iters = steps > 0 ? steps : calibrate(step);
    // Reset state for clean measurement
    memset(y, 0, H * sizeof(float));
    memset(c, 0, H * sizeof(float));
    memset(n, 0, H * sizeof(float));
    memset(m, 0, H * sizeof(float));

    double total = measure(step, iters);
    if (steps <= 0)
        printf("slstm_f32   %5d  %8d  %10.1f  %9.3f\n",
               H, iters, total, total * 1000.0 / iters);

    free(W); free(R); free(b); free(x);
    free(y); free(c); free(n); free(m); free(scratch);
}

static void bench_mlstm_f32(int H, int steps) {
    const int I = H;
    const int Wrows = 4 * H + 2;
    float* W = (float*)malloc(Wrows * I * sizeof(float));
    float* b = (float*)malloc(Wrows * sizeof(float));
    float* x = (float*)malloc(I * sizeof(float));
    float* y = (float*)calloc(H, sizeof(float));
    float* C = (float*)calloc(H * H, sizeof(float));
    float* n = (float*)calloc(H, sizeof(float));
    float  m_state = 0.0f;
    float* scratch = (float*)calloc(Wrows, sizeof(float));

    fill_f32(W, Wrows * I, -0.5f, 0.5f);
    fill_f32(b, Wrows, -0.1f, 0.1f);
    fill_f32(x, I, -1.0f, 1.0f);

    MlstmParams params = {};

    auto step = [&]() {
        mlstm_step_f32(x, W, b, y, C, n, &m_state, scratch, I, H, H, &params);
    };

    int iters = steps > 0 ? steps : calibrate(step);
    memset(y, 0, H * sizeof(float));
    memset(C, 0, H * H * sizeof(float));
    memset(n, 0, H * sizeof(float));
    m_state = 0.0f;

    double total = measure(step, iters);
    if (steps <= 0)
        printf("mlstm_f32   %5d  %8d  %10.1f  %9.3f\n",
               H, iters, total, total * 1000.0 / iters);

    free(W); free(b); free(x);
    free(y); free(C); free(n); free(scratch);
}

/* ---------------------------------------------------------------------------
 * Calibrated INT8 fixtures.
 *
 * These used to hardcode every quantization scale and quantize nothing: random
 * int8 weights straight from the RNG, a fixed 0.001 for the INT16 state. That
 * is not a deployment's numbers and it is not even a stable regime. Measured on
 * the old fixture: 99-100% of the mLSTM C sat on its INT16 rails at H=64, and
 * at H=128 the input gate was pinned shut so the state never left zero. A
 * saturated state puts the kernel in xlstm_round_clamp_i32 instead of its
 * arithmetic, and the clamp is the CHEAPER path - so the fixture reported the
 * kernel faster than a calibrated one would be, which is the wrong direction
 * for a number anyone might quote.
 *
 * It also meant the f32 and INT8 benches ran on unrelated numbers, so an
 * INT8-versus-f32 ratio from this harness compared two different problems.
 *
 * Both now start from the SAME float tensors as their f32 counterpart and
 * derive every scale the way a deployment would: weights and input from their
 * own ranges, output and state from a float warm-up's observed trajectory.
 * Same pattern as test/{slstm,mlstm}_s8_test.cc's Prepare* and the hardware
 * harness, both of which calibrate rather than assume.
 * ------------------------------------------------------------------------ */

/* Matches XLSTM_GENERATOR_HEADROOM in the generated reference data: the INT16
 * state is calibrated with room above the trajectory the warm-up saw, because
 * the INT8 trajectory is not bit-identical to the f32 one it was measured on. */
#define BENCH_STATE_HEADROOM 4.0f

/* The warm-up runs to the trajectory's fixed point rather than for a fixed
 * number of steps. The measured run is thousands of iterations of a CONSTANT
 * input, which drives the state to a fixed point a short warm-up does not
 * reach - a 64-step calibration left the H=128 sLSTM state on its rails, and
 * require_unsaturated() below caught it. Stop when a whole chunk adds no new
 * extreme; the cap is a backstop, not the expected exit. */
static const int kCalibChunk = 512;
static const int kCalibMaxChunks = 64;

struct Range { float lo, hi, absmax; };

static void range_init(Range* r) { r->lo = 0.0f; r->hi = 0.0f; r->absmax = 0.0f; }

static void range_add(Range* r, const float* v, int n) {
    for (int i = 0; i < n; ++i) {
        if (v[i] < r->lo) r->lo = v[i];
        if (v[i] > r->hi) r->hi = v[i];
        float a = fabsf(v[i]);
        if (a > r->absmax) r->absmax = a;
    }
}

/* Both quant helpers read only the extremes of what they are handed, so a
 * two-element summary calibrates identically to the whole trajectory. */
static int range_same(const Range* a, const Range* b) {
    return a->lo == b->lo && a->hi == b->hi && a->absmax == b->absmax;
}

static void range_asym(const Range* r, XlstmQuantParam* q) {
    float mm[2] = { r->lo, r->hi };
    xlstm_quant_asymmetric(mm, 2, q);
}
static void range_s16(const Range* r, XlstmQuantParam* q) {
    float a[1] = { r->absmax };
    xlstm_quant_symmetric_s16(a, 1, BENCH_STATE_HEADROOM, q);
}

/* The post-condition the old fixture silently violated. A saturated state means
 * the measurement is of the clamp, not the kernel, so this exits rather than
 * printing a number that would be wrong in a flattering direction. */
static void require_unsaturated(const char* what, int H,
                                const int16_t* a, int na,
                                const int16_t* b, int nb) {
    int sat = 0;
    for (int i = 0; i < na; ++i) if (a[i] == 32767 || a[i] == -32768) sat++;
    for (int i = 0; i < nb; ++i) if (b[i] == 32767 || b[i] == -32768) sat++;
    if (sat) {
        std::fprintf(stderr,
            "bench: %s H=%d left %d INT16 state element(s) saturated after the "
            "measured run. The calibration did not hold, so this timing would "
            "be of xlstm_round_clamp_i32 rather than the kernel.\n",
            what, H, sat);
        std::exit(1);
    }
}

static void bench_slstm_s8(int H, int steps) {
    const int I = H;
    int8_t*  W_q = (int8_t*)malloc(4 * H * I);
    int8_t*  R_q = (int8_t*)malloc(4 * H * H);
    int32_t* b_q = (int32_t*)malloc(4 * H * sizeof(int32_t));
    int8_t*  x   = (int8_t*)malloc(I);
    int8_t*  y   = (int8_t*)calloc(H, 1);
    int16_t* c   = (int16_t*)calloc(H, sizeof(int16_t));
    int16_t* n   = (int16_t*)calloc(H, sizeof(int16_t));
    float*   m   = (float*)calloc(H, sizeof(float));
    int32_t* scratch = (int32_t*)calloc(4 * H, sizeof(int32_t));

    /* Same float fixture as bench_slstm_f32, so the two time one problem. */
    float* Wf = (float*)malloc(4 * H * I * sizeof(float));
    float* Rf = (float*)malloc(4 * H * H * sizeof(float));
    float* bf = (float*)malloc(4 * H * sizeof(float));
    float* xf = (float*)malloc(I * sizeof(float));
    fill_f32(Wf, 4 * H * I, -0.5f, 0.5f);
    fill_f32(Rf, 4 * H * H, -0.5f, 0.5f);
    fill_f32(bf, 4 * H, -0.1f, 0.1f);
    fill_f32(xf, I, -1.0f, 1.0f);

    Range ry, rc, rn;
    range_init(&ry); range_init(&rc); range_init(&rn);
    {
        float* fy = (float*)calloc(H, sizeof(float));
        float* fc = (float*)calloc(H, sizeof(float));
        float* fn = (float*)calloc(H, sizeof(float));
        float* fm = (float*)calloc(H, sizeof(float));
        float* fs = (float*)calloc(4 * H, sizeof(float));
        SlstmParams fp = {0.0f};
        for (int k = 0; k < kCalibMaxChunks; ++k) {
            Range py = ry, pc = rc, pn = rn;
            for (int t = 0; t < kCalibChunk; ++t) {
                slstm_step_f32(xf, Wf, Rf, bf, fy, fc, fn, fm, fs, I, H, &fp);
                range_add(&ry, fy, H); range_add(&rc, fc, H); range_add(&rn, fn, H);
            }
            if (range_same(&py, &ry) && range_same(&pc, &rc) &&
                range_same(&pn, &rn)) break;
        }
        free(fy); free(fc); free(fn); free(fm); free(fs);
    }

    SlstmS8Params params = {};
    params.cell_clip = 0.0f;
    XlstmQuantParam w_qp, r_qp, b_qp;
    xlstm_quant_symmetric(Wf, 4 * H * I, &w_qp);
    xlstm_quant_symmetric(Rf, 4 * H * H, &r_qp);
    xlstm_quant_asymmetric(xf, I, &params.x_quant);
    xlstm_quantize_f32_to_s8(Wf, W_q, 4 * H * I, &w_qp);
    xlstm_quantize_f32_to_s8(Rf, R_q, 4 * H * H, &r_qp);
    xlstm_quantize_f32_to_s8(xf, x, I, &params.x_quant);
    b_qp.scale = w_qp.scale * params.x_quant.scale;
    b_qp.zero_point = 0;
    xlstm_quantize_f32_to_s32(bf, b_q, 4 * H, &b_qp);
    params.W_scale = w_qp.scale;
    params.R_scale = r_qp.scale;
    range_asym(&ry, &params.y_quant);
    range_s16(&rc, &params.c_quant);
    range_s16(&rn, &params.n_quant);
    free(Wf); free(Rf); free(bf); free(xf);

    auto step = [&]() {
        slstm_step_s8(x, W_q, R_q, b_q, y, c, n, m, scratch, I, H, &params);
    };

    int iters = steps > 0 ? steps : calibrate(step);
    memset(y, 0, H);
    memset(c, 0, H * sizeof(int16_t));
    memset(n, 0, H * sizeof(int16_t));
    memset(m, 0, H * sizeof(float));

    double total = measure(step, iters);
    require_unsaturated("slstm_s8", H, c, H, n, H);
    if (steps <= 0)
        printf("slstm_s8    %5d  %8d  %10.1f  %9.3f\n",
               H, iters, total, total * 1000.0 / iters);

    free(W_q); free(R_q); free(b_q); free(x);
    free(y); free(c); free(n); free(m); free(scratch);
}

static void bench_mlstm_s8(int H, int steps) {
    const int I = H;
    const int Wrows = 4 * H + 2;
    int8_t*  W_q = (int8_t*)malloc(Wrows * I);
    int32_t* b_q = (int32_t*)malloc(Wrows * sizeof(int32_t));
    int8_t*  x   = (int8_t*)malloc(I);
    int8_t*  y   = (int8_t*)calloc(H, 1);
    int16_t* C   = (int16_t*)calloc(H * H, sizeof(int16_t));
    int16_t* n   = (int16_t*)calloc(H, sizeof(int16_t));
    float    m_state = 0.0f;
    int32_t* scratch = (int32_t*)calloc(Wrows, sizeof(int32_t));

    /* Same float fixture as bench_mlstm_f32, so the two time one problem. */
    float* Wf = (float*)malloc(Wrows * I * sizeof(float));
    float* bf = (float*)malloc(Wrows * sizeof(float));
    float* xf = (float*)malloc(I * sizeof(float));
    fill_f32(Wf, Wrows * I, -0.5f, 0.5f);
    fill_f32(bf, Wrows, -0.1f, 0.1f);
    fill_f32(xf, I, -1.0f, 1.0f);

    Range ry, rC, rn;
    range_init(&ry); range_init(&rC); range_init(&rn);
    {
        float* fy = (float*)calloc(H, sizeof(float));
        float* fC = (float*)calloc(H * H, sizeof(float));
        float* fn = (float*)calloc(H, sizeof(float));
        float  fm = 0.0f;
        float* fs = (float*)calloc(Wrows, sizeof(float));
        MlstmParams fp = {};
        for (int k = 0; k < kCalibMaxChunks; ++k) {
            Range py = ry, pC = rC, pn = rn;
            for (int t = 0; t < kCalibChunk; ++t) {
                mlstm_step_f32(xf, Wf, bf, fy, fC, fn, &fm, fs, I, H, H, &fp);
                range_add(&ry, fy, H); range_add(&rC, fC, H * H); range_add(&rn, fn, H);
            }
            if (range_same(&py, &ry) && range_same(&pC, &rC) &&
                range_same(&pn, &rn)) break;
        }
        free(fy); free(fC); free(fn); free(fs);
    }

    MlstmS8Params params = {};
    params.cell_clip = 0.0f;
    XlstmQuantParam w_qp, b_qp;
    xlstm_quant_symmetric(Wf, Wrows * I, &w_qp);
    xlstm_quant_asymmetric(xf, I, &params.x_quant);
    xlstm_quantize_f32_to_s8(Wf, W_q, Wrows * I, &w_qp);
    xlstm_quantize_f32_to_s8(xf, x, I, &params.x_quant);
    b_qp.scale = w_qp.scale * params.x_quant.scale;
    b_qp.zero_point = 0;
    xlstm_quantize_f32_to_s32(bf, b_q, Wrows, &b_qp);
    params.W_scale = w_qp.scale;
    range_asym(&ry, &params.y_quant);
    range_s16(&rC, &params.C_quant);
    range_s16(&rn, &params.n_quant);
    free(Wf); free(bf); free(xf);

    auto step = [&]() {
        mlstm_step_s8(x, W_q, b_q, y, C, n, &m_state, scratch, I, H, H, &params);
    };

    int iters = steps > 0 ? steps : calibrate(step);
    memset(y, 0, H);
    memset(C, 0, H * H * sizeof(int16_t));
    memset(n, 0, H * sizeof(int16_t));
    m_state = 0.0f;

    double total = measure(step, iters);
    require_unsaturated("mlstm_s8", H, C, H * H, n, H);
    if (steps <= 0)
        printf("mlstm_s8    %5d  %8d  %10.1f  %9.3f\n",
               H, iters, total, total * 1000.0 / iters);

    free(W_q); free(b_q); free(x);
    free(y); free(C); free(n); free(scratch);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    // Fixed-work mode: `xlstm_bench <kernel> <H> <steps>`.
    if (argc == 4) {
        const char* k = argv[1];
        const int H = atoi(argv[2]);
        const int steps = atoi(argv[3]);
        if (H < 1 || steps < 1) { fprintf(stderr, "bad H or steps\n"); return 2; }
        if      (!strcmp(k, "slstm_f32")) bench_slstm_f32(H, steps);
        else if (!strcmp(k, "mlstm_f32")) bench_mlstm_f32(H, steps);
        else if (!strcmp(k, "slstm_s8"))  bench_slstm_s8(H, steps);
        else if (!strcmp(k, "mlstm_s8"))  bench_mlstm_s8(H, steps);
        else { fprintf(stderr, "unknown kernel: %s\n", k); return 2; }
        return 0;
    }

    const int sizes[] = {16, 32, 64, 128};
    const int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("xlstm.c benchmark | backend: %s\n", xlstm_simd_backend());
    printf("%-12s %5s  %8s  %10s  %9s\n",
           "kernel", "H", "iters", "total_ms", "us/step");
    printf("------------------------------------------------------\n");

    for (int i = 0; i < nsizes; i++) bench_slstm_f32(sizes[i], 0);
    for (int i = 0; i < nsizes; i++) bench_mlstm_f32(sizes[i], 0);
    for (int i = 0; i < nsizes; i++) bench_slstm_s8(sizes[i], 0);
    for (int i = 0; i < nsizes; i++) bench_mlstm_s8(sizes[i], 0);

    return 0;
}
