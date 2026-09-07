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

static void fill_s8(int8_t* buf, int n) {
    for (int i = 0; i < n; i++) buf[i] = (int8_t)((lcg_next() % 256) - 128);
}

static void fill_s32(int32_t* buf, int n) {
    for (int i = 0; i < n; i++) buf[i] = (int32_t)(lcg_next() % 2048) - 1024;
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

    fill_s8(W_q, 4 * H * I);
    fill_s8(R_q, 4 * H * H);
    fill_s32(b_q, 4 * H);
    fill_s8(x, I);

    SlstmS8Params params = {};
    params.cell_clip = 0.0f;
    params.W_scale = 0.01f;
    params.R_scale = 0.01f;
    params.x_quant = {0.05f, 0};
    params.y_quant = {0.05f, 0};
    params.c_quant = {0.001f, 0};
    params.n_quant = {0.001f, 0};

    auto step = [&]() {
        slstm_step_s8(x, W_q, R_q, b_q, y, c, n, m, scratch, I, H, &params);
    };

    int iters = steps > 0 ? steps : calibrate(step);
    memset(y, 0, H);
    memset(c, 0, H * sizeof(int16_t));
    memset(n, 0, H * sizeof(int16_t));
    memset(m, 0, H * sizeof(float));

    double total = measure(step, iters);
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

    fill_s8(W_q, Wrows * I);
    fill_s32(b_q, Wrows);
    fill_s8(x, I);

    MlstmS8Params params = {};
    params.cell_clip = 0.0f;
    params.W_scale = 0.01f;
    params.x_quant = {0.05f, 0};
    params.y_quant = {0.05f, 0};
    params.C_quant = {0.001f, 0};
    params.n_quant = {0.001f, 0};

    auto step = [&]() {
        mlstm_step_s8(x, W_q, b_q, y, C, n, &m_state, scratch, I, H, H, &params);
    };

    int iters = steps > 0 ? steps : calibrate(step);
    memset(y, 0, H);
    memset(C, 0, H * H * sizeof(int16_t));
    memset(n, 0, H * sizeof(int16_t));
    m_state = 0.0f;

    double total = measure(step, iters);
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
