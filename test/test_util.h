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

/* Value assertion for an INT8 exit state (sLSTM c/n/m, mLSTM C/n/m),
 * one bound per ELEMENT.
 *
 * `tol` is the case's tol_s8_state_per_elem / _n_per_elem / _m_per_elem
 * array from reference_data.h, produced by generate_reference.py's
 * compute_state_tol_per_elem from that element's own measured replica
 * error and its own golden magnitude. Read that function for the
 * derivation; the two properties it guarantees, and that this function
 * re-checks rather than trusts, are:
 *
 *   1. NON-VACUOUS. bound < |golden[i]|, so zeroing element i (error
 *      exactly |golden[i]|) always fails. An earlier round bounded each
 *      state tensor at a fraction of the whole tensor's maximum instead;
 *      that passed a mutant which zeroed 4063 of SweepM64's 4096 C
 *      elements, because each one individually sat under the tensor-wide
 *      bound. The invariant is checked here, per element, so a bound that
 *      cannot fail fails the build instead of passing silently.
 *   2. NOT BIT-EXACT. bound >= the replica's measured error for that
 *      element, including a 1.001x activation perturbation. A bound
 *      derived from |golden[i]| by a fixed relative factor would be
 *      unusable: sLSTM Test1's c[0] deviates 3.749x its own golden.
 *
 * A negative entry is XLSTM_STATE_TOL_UNASSERTABLE: no bound for that
 * element can satisfy both at once (its golden is exactly zero, so
 * zeroing it is undetectable, or its honest measured error already covers
 * its whole dynamic range). Those are skipped and counted into
 * *unasserted, which the callers report, so an unasserted element shows
 * up in the test output rather than quietly not existing. */
[[maybe_unused]] static bool ExpectStatePerElem(const char* name, const float* expected,
                                                const float* actual, const float* tol,
                                                int len, int* unasserted) {
    if (!tol) {
        std::printf("  note: %s has no per-element bounds in reference_data.h - "
                    "not asserted (regenerate with `make reference`)\n", name);
        *unasserted += len;
        return true;
    }
    for (int i = 0; i < len; ++i) {
        if (tol[i] < 0.0f) { ++*unasserted; continue; }
        float rng = std::abs(expected[i]);
        if (tol[i] >= rng) {
            std::printf("  FAIL %s[%d]: bound %.8e is vacuous - it is >= the element's "
                        "own magnitude %.8e, so zeroing this element would pass\n",
                        name, i, tol[i], rng);
            return false;
        }
        float diff = std::abs(expected[i] - actual[i]);
        if (diff > tol[i]) {
            std::printf("  FAIL %s[%d]: expected %.8f, got %.8f (diff %.2e, bound %.2e, "
                        "element magnitude %.2e)\n",
                        name, i, expected[i], actual[i], diff, tol[i], rng);
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
