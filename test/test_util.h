/* Shared test utilities for xlstm.c kernel tests.
 * =========================================================================*/

#ifndef TEST_UTIL_H_
#define TEST_UTIL_H_

#include <cmath>
#include <cstdio>

static int g_tests_run = 0;
static int g_tests_passed = 0;

static bool ExpectNear(const char* name, const float* expected,
                       const float* actual, int len, float tol) {
    for (int i = 0; i < len; ++i) {
        float diff = std::abs(expected[i] - actual[i]);
        if (diff > tol) {
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
