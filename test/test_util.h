/* Shared test utilities for xlstm.c kernel tests.
 * =========================================================================*/

#ifndef TEST_UTIL_H_
#define TEST_UTIL_H_

#include <cmath>
#include <cstdio>

static int g_tests_run = 0;
static int g_tests_passed = 0;

/* Relative term for float comparisons, used by EVERY comparison in this
 * suite - the f32 runners' ExpectNear below, and equally the INT8 runners'
 * per-channel checks, which spell out `tol + kRelTol * |expected|` inline
 * (slstm_s8_test.cc, mlstm_s8_test.cc). It is not an f32-only constant.
 *
 * Why a relative term at all: kernel outputs accumulate over the hidden
 * dimension with exponential gating and a normalizer division, so error
 * scales with magnitude - a pure absolute tolerance demands accuracy that
 * float32 cannot deliver at larger values. In the INT8 runners it plays
 * the same role on top of the per-channel INT8 bound: the bound covers
 * quantization error, this term covers the float arithmetic that the
 * dequantized comparison is still carried out in. Mirrors numpy.allclose,
 * which the Python adapter tests already rely on. */
static const float kRelTol = 2e-6f;

/* [[maybe_unused]] applies to ExpectNear, not to kRelTol: ExpectNear is
 * called only by the f32 runners (slstm_test.cc, mlstm_test.cc), because
 * the INT8 runners need a different bound per channel
 * (tc->tol_s8_per_channel) and so open-code their comparison instead.
 * kRelTol itself is used in all four translation units. */
[[maybe_unused]] static bool ExpectNear(const char* name, const float* expected,
                       const float* actual, int len, float tol) {
    for (int i = 0; i < len; ++i) {
        float diff = std::abs(expected[i] - actual[i]);
        if (diff > tol + kRelTol * std::abs(expected[i])) {
            std::printf("  FAIL %s[%d]: expected %.8f, got %.8f (diff %.2e)\n",
                        name, i, expected[i], actual[i], diff);
            return false;
        }
    }
    return true;
}

static bool ExpectFinite(const char* name, const float* vals, int len) {
    for (int i = 0; i < len; ++i) {
        if (!std::isfinite(vals[i])) {
            std::printf("  FAIL %s[%d]: not finite (%.8f)\n", name, i, vals[i]);
            return false;
        }
    }
    return true;
}

/* Absolute slack added to the floor-consistency bound in the two INT8
 * runners (slstm_s8_test.cc, mlstm_s8_test.cc).
 *
 * Four channels in reference_data.h have a replica-predicted floor of
 * exactly 0.0 (kTest3_tol_s8_floor_per_channel, kMTest3_...), which turns
 * `err > floor * 1.5f` into `err > 0.0` - a demand for bit-exact float
 * equality that a single ULP breaks. Measured: -O2 -ffast-math failed with
 * err = 1.192e-07 (2 ULP at |y| ~ 1) before this term existed. A different
 * libm, a contracted FMA or a CMSIS-NN LUT would do the same, which is
 * precisely the class of backend this gate is meant to admit. This is the
 * same defect an earlier round fixed for tol_s8_per_channel; it survived
 * in the floor check.
 *
 * 1e-6 is ~8x FLT_EPSILON, so it absorbs last-bit noise at |y| ~ 1, while
 * staying far below one output LSB: the two cases where a zero floor
 * actually binds quantize y at 3.92e-03 (Test3) and 5.88e-02 (MTest3), so
 * this is 0.026% and 0.0017% of a step there, and 9.4% of the smallest
 * y_quant.scale anywhere in the table (1.07e-05, MTest1, whose own floors
 * are non-zero and dominate). An off-by-one requantization - which moves
 * the dequantized value by a full LSB - therefore still fails. The
 * runners add kRelTol * channel_ref on top so the same reasoning holds at
 * MTest3's |y| ~ 15, where one ULP is already 1.9e-06. */
static const float kFloorEps = 1e-6f;

/* Absolute floor added to a state-tensor bound so that a tensor whose
 * golden values are all zero (range 0) is not silently asked for bit-exact
 * float equality - the same defect class as kFloorEps in the two INT8
 * runners. No case in reference_data.h has a zero-range state today; this
 * is a guard, not a working tolerance. At 1e-6 it is ~8x FLT_EPSILON and
 * several orders below every state magnitude in the table, so it never
 * moves a real bound. */
static const float kStateTolAbs = 1e-6f;

/* Value assertion for an INT8 exit state (sLSTM c/n/m, mLSTM C/n/m).
 *
 * reference_data.h carries per-channel INT8 bounds for the OUTPUT path
 * only - generate_reference.py derives nothing for the state tensors. A
 * per-channel relative bound is not usable for them either: the measured
 * INT8-vs-f32 deviation reaches 3.75x a channel's own magnitude (sLSTM
 * Test1's c[0], the same sign-flip channel .docs/SCOPE.md section 8
 * documents for y). So each state tensor is held to one case-wide
 * absolute bound: `frac` x the largest golden magnitude that tensor takes
 * in this case.
 *
 * That bound is non-vacuous by construction for any frac < 1 - a kernel
 * that returns an all-zero state errs by exactly the range and fails -
 * and the check below turns that into a verified invariant rather than a
 * comment, so raising frac past 1.0 fails loudly instead of quietly
 * disabling the assertion.
 *
 * What it does NOT do: a case-wide bound cannot see a small-magnitude
 * channel hiding behind a large one in the same tensor. That is the same
 * limitation the per-channel output bounds exist to avoid, and it is
 * accepted here because no per-channel state data exists to do better.
 * See each caller's kStateTolFrac comment for the measured ratios. */
[[maybe_unused]] static bool ExpectStateNear(const char* name, const float* expected,
                                             const float* actual, int len, float frac) {
    float range = 0.0f;
    for (int i = 0; i < len; ++i) {
        float a = std::abs(expected[i]);
        if (a > range) range = a;
    }
    float tol = frac * range + kStateTolAbs;
    if (range > 0.0f && tol >= range) {
        std::printf("  FAIL %s: bound %.8e is vacuous - it is >= the tensor's own "
                    "range %.8e, so an all-zero state would pass\n", name, tol, range);
        return false;
    }
    for (int i = 0; i < len; ++i) {
        float diff = std::abs(expected[i] - actual[i]);
        if (diff > tol) {
            std::printf("  FAIL %s[%d]: expected %.8f, got %.8f (diff %.2e, "
                        "bound %.2e = %.3f x range %.6f)\n",
                        name, i, expected[i], actual[i], diff, tol, frac, range);
            return false;
        }
    }
    return true;
}

#define RUN_TEST(test_fn)                                  \
    do {                                                   \
        g_tests_run++;                                     \
        std::printf("[ RUN      ] %s\n", #test_fn);       \
        if (test_fn()) {                                   \
            g_tests_passed++;                              \
            std::printf("[       OK ] %s\n", #test_fn);    \
        } else {                                           \
            std::printf("[  FAILED  ] %s\n", #test_fn);    \
        }                                                  \
    } while (0)

#endif /* TEST_UTIL_H_ */
