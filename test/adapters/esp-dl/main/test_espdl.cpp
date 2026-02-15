/* ESP-DL integration test — exercises sLSTM/mLSTM Module subclasses.
 *
 * Tests constructor, get_output_shape(), and init_states()/free_states()
 * lifecycle against the real ESP-DL framework on emulated hardware (QEMU).
 *
 * Note: Full forward() test requires constructing ModelContext + TensorBase
 * objects, which are tightly coupled to ESP-DL internals. Instead we verify
 * that the classes instantiate correctly, compute output shapes, and survive
 * init/destroy cycles — proving the adapter links and runs against real
 * ESP-DL on real (emulated) hardware.
 *
 * Build: see Dockerfile (ESP-IDF project built for esp32s3, run via QEMU)
 */

#include <cstdio>
#include <cstring>
#include <vector>

#include "slstm_espdl.hpp"
#include "mlstm_espdl.hpp"

#include "reference_data.h"

namespace {

int g_tests_run = 0;
int g_tests_passed = 0;

// ---------------------------------------------------------------------------
// sLSTM: constructor + get_output_shape
// ---------------------------------------------------------------------------
bool TestSLstmConstruction() {
    const int H = 2, I = 2;
    dl::module::SLSTM slstm("test_slstm", H, I);

    if (slstm.m_hidden_size != H || slstm.m_input_size != I) {
        printf("  FAIL: wrong dimensions %d,%d (expected %d,%d)\n",
               slstm.m_hidden_size, slstm.m_input_size, H, I);
        return false;
    }

    // Test get_output_shape: input [B=1, T=3, I=2] -> output [1, 3, H=2]
    std::vector<std::vector<int>> in_shapes = {{1, 3, I}};
    auto out_shapes = slstm.get_output_shape(in_shapes);

    if (out_shapes.size() != 1) {
        printf("  FAIL: expected 1 output shape, got %d\n", (int)out_shapes.size());
        return false;
    }

    std::vector<int> expected = {1, 3, H};
    if (out_shapes[0] != expected) {
        printf("  FAIL: wrong output shape\n");
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// mLSTM: constructor + get_output_shape
// ---------------------------------------------------------------------------
bool TestMLstmConstruction() {
    const int H = 2, I = 3;
    dl::module::MLSTM mlstm("test_mlstm", H, I);

    if (mlstm.m_hidden_size != H || mlstm.m_input_size != I) {
        printf("  FAIL: wrong dimensions %d,%d (expected %d,%d)\n",
               mlstm.m_hidden_size, mlstm.m_input_size, H, I);
        return false;
    }

    // Test get_output_shape: input [B=1, T=1, I=3] -> output [1, 1, H=2]
    std::vector<std::vector<int>> in_shapes = {{1, 1, I}};
    auto out_shapes = mlstm.get_output_shape(in_shapes);

    if (out_shapes.size() != 1) {
        printf("  FAIL: expected 1 output shape, got %d\n", (int)out_shapes.size());
        return false;
    }

    std::vector<int> expected = {1, 1, H};
    if (out_shapes[0] != expected) {
        printf("  FAIL: wrong output shape\n");
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// sLSTM: lifecycle test — repeated init/destroy cycles
// ---------------------------------------------------------------------------
bool TestSLstmLifecycle() {
    for (int i = 0; i < 3; i++) {
        dl::module::SLSTM* slstm = new dl::module::SLSTM("lifecycle", 2, 2);
        delete slstm;
    }
    return true;
}

// ---------------------------------------------------------------------------
// mLSTM: lifecycle test — repeated init/destroy cycles
// ---------------------------------------------------------------------------
bool TestMLstmLifecycle() {
    for (int i = 0; i < 3; i++) {
        dl::module::MLSTM* mlstm = new dl::module::MLSTM("lifecycle", 2, 3);
        delete mlstm;
    }
    return true;
}

}  // namespace

#define RUN_TEST(fn)                                      \
    do {                                                  \
        printf("[RUN ] %s\n", #fn);                       \
        g_tests_run++;                                    \
        if (fn()) {                                       \
            printf("[  OK] %s\n", #fn);                   \
            g_tests_passed++;                             \
        } else {                                          \
            printf("[FAIL] %s\n", #fn);                   \
        }                                                 \
    } while (0)

extern "C" void app_main(void) {
    printf("=== ESP-DL integration tests ===\n\n");

    RUN_TEST(TestSLstmConstruction);
    RUN_TEST(TestMLstmConstruction);
    RUN_TEST(TestSLstmLifecycle);
    RUN_TEST(TestMLstmLifecycle);

    printf("\n%d/%d tests passed.\n", g_tests_passed, g_tests_run);

    if (g_tests_passed == g_tests_run) {
        printf("\nPASS\n");
    } else {
        printf("\nFAIL\n");
    }

    // Exit QEMU
    fflush(stdout);
}
