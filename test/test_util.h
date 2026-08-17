/* Shared test utilities for xlstm.c kernel tests.
 *
 * Standing rule for every tolerance below, and for the floor factors the
 * INT8 runners spell out inline: re-run `make mutants` after changing one.
 * A loosened bound fails by making a gate quietly stop failing, which a
 * green `make test` cannot show you - the mutation battery injects the
 * defects these bounds exist to catch and fails if one now passes.
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

/* Multiple of a state element's replica-predicted floor that its real
 * measured error is allowed to reach before the drift detector fires.
 *
 * The output path's equivalent is the literal 1.5 in the two INT8
 * runners' floor-consistency loops. That factor does NOT transfer, and
 * this was measured rather than assumed: across all 5198 state elements
 * the worst real-kernel-error / replica-floor ratio is 2.1187 (sLSTM
 * Test1's m[1], floor 1.32e-03 vs error 2.80e-03), where the output
 * path's worst is 1.0187. A 1.5 factor would false-fire on the correct,
 * unmodified kernel.
 *
 * The tail is short: only 3 of 5198 elements exceed 1.01, all in Test1
 * (m[1] 2.119, m[0] 1.038, c[0] 1.024). 2509 sit in 1.000-1.010 and 2683
 * at or below 1.0 - the state is requantized to INT16, so replica and
 * kernel usually land on the same integer and the ratio is 1.0 plus float
 * noise. 3.0 clears the measured worst by 1.42x, which is the same margin
 * the output path's 1.5 keeps over its own worst of 1.0187 (1.47x). Do
 * not tighten it toward 2.2 to "make the test stricter": the margin is
 * what stops a legitimate backend difference from being reported as
 * replica drift. */
static const float kStateFloorFactor = 3.0f;

/* Exit-state drift detector - the twin of the output path's
 * floor-consistency check, which has caught every kernel mutant tried
 * against it, and which the state path had no equivalent of.
 *
 * `floor` is tol_s8_{state,n,m}_floor_per_elem from reference_data.h:
 * what generate_reference.py's numpy replica measured as that element's
 * own worst-case error. The replica and the real kernel are two
 * hand-synchronized implementations of the same math, and a new SIMD
 * backend is exactly what stresses that coupling. Without this check, a
 * kernel change that moves the exit state but not y is invisible as a
 * divergence: it passes if it stays inside the per-element bound, or it
 * surfaces as a bound violation with nothing pointing at the replica.
 *
 * This covers EVERY element, including the 142 whose bound is
 * XLSTM_STATE_TOL_UNASSERTABLE. Drift detection does not need a
 * non-vacuous bound to exist, so those elements are unguarded for
 * correctness but still guarded against divergence.
 *
 * kFloorEps + kRelTol carry the same job here as in the output check: 3
 * elements have a floor of exactly 0.0, which without them would demand
 * bit-exact float equality (all 3 currently match exactly, so the guard
 * is what stops one ULP from a different libm failing the suite), and
 * the relative term keeps that true at mLSTM MTest3's C ~= 159 where one
 * ULP is already 1.5e-05. */
[[maybe_unused]] static bool ExpectStateFloorConsistent(const char* name,
                                                        const float* expected,
                                                        const float* actual,
                                                        const float* floor, int len) {
    if (!floor) return true;
    for (int i = 0; i < len; ++i) {
        float err = std::abs(expected[i] - actual[i]);
        float bound = floor[i] * kStateFloorFactor + kFloorEps
                    + kRelTol * std::abs(expected[i]);
        if (err > bound) {
            std::printf("  FAIL state-floor-consistency %s[%d]: measured error %.8g "
                        "exceeds bound %.8g (%.1fx the numpy replica's predicted floor "
                        "%.8g, plus float slack) - either the C kernel regressed or "
                        "generate_reference.py's replica drifted away from it; check the "
                        "kernel first if you just changed one\n",
                        name, i, err, bound, kStateFloorFactor, floor[i]);
            return false;
        }
    }
    return true;
}

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
