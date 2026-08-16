/* ESP-DL integration test - exercises sLSTM/mLSTM Module subclasses.
 *
 * READ THIS BEFORE TRUSTING A GREEN RUN. This suite is COMPILED but never
 * EXECUTED. ESP-DL needs esp32s3 or newer; the QEMU shipped with ESP-IDF
 * v5.3 emulates esp32 only (`qemu-system-xtensa -machine help` lists no
 * s3), so the Dockerfile builds the image and stops. What that proves is
 * real but narrow: the adapter compiles and links against genuine ESP-DL
 * headers on a genuine cross-compilation target, which catches a wrong
 * TensorBase accessor, a missing symbol or a type error. None of the
 * assertions below have ever run. Do not read this file as evidence that
 * the ESP-DL numerical path is correct.
 *
 * What the INT8 forward tests would check when a runner exists: they build
 * real TensorBase objects, run the module, and require the result to equal
 * what slstm_eval_s8 / mlstm_eval_s8 produce when called directly with the
 * scales DL_SCALE derives from the same exponents. That is exact and fails
 * on any tensor, exponent or state wired to the wrong place. It is not a
 * check against the PyTorch-derived reference: ESP-DL quantization is
 * symmetric power-of-two with no zero-point, so it cannot express the
 * reference calibration (arbitrary scales, asymmetric input zero-point)
 * that reference_data.json's expected integers were produced with. The INT8
 * kernels themselves are gated by test/{slstm,mlstm}_s8_test.cc and by the
 * ONNX Runtime, microTVM and TFLM adapter suites, all of which do execute.
 *
 * Build: see Dockerfile (ESP-IDF project built for esp32s3)
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

// Deterministic pseudo-random int8 fill - no dependency on rand()'s
// implementation, so the two sides of every comparison see identical bytes.
template <typename T>
void FillPattern(T* dst, int n, int seed, int lo, int hi) {
    uint32_t s = (uint32_t)seed * 2654435761u + 1u;
    for (int i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        dst[i] = (T)(lo + (int)((s >> 16) % (uint32_t)(hi - lo + 1)));
    }
}

template <typename T>
bool ExpectEqual(const char* name, const T* expected, const T* actual, int n) {
    for (int i = 0; i < n; i++) {
        if (expected[i] != actual[i]) {
            printf("  FAIL %s[%d]: expected %d, got %d\n",
                   name, i, (int)expected[i], (int)actual[i]);
            return false;
        }
    }
    return true;
}

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
// sLSTM: lifecycle test - repeated init/destroy cycles
// ---------------------------------------------------------------------------
bool TestSLstmLifecycle() {
    for (int i = 0; i < 3; i++) {
        dl::module::SLSTM* slstm = new dl::module::SLSTM("lifecycle", 2, 2);
        delete slstm;
    }
    return true;
}

// ---------------------------------------------------------------------------
// mLSTM: lifecycle test - repeated init/destroy cycles
// ---------------------------------------------------------------------------
bool TestMLstmLifecycle() {
    for (int i = 0; i < 3; i++) {
        dl::module::MLSTM* mlstm = new dl::module::MLSTM("lifecycle", 2, 3);
        delete mlstm;
    }
    return true;
}

// ---------------------------------------------------------------------------
// sLSTM INT8: forward() through real TensorBase objects, checked against the
// core kernel driven directly with the same scales.
// ---------------------------------------------------------------------------
bool TestSLstmInt8Forward() {
    const int H = 4, I = 4, T = 2, B = 1;
    // Exponents are ESP-DL's scale representation: scale = 2^exponent.
    const int x_exp = -6, W_exp = -7, R_exp = -7, y_exp = -7;
    const int c_exp = -10, n_exp = -10;

    int8_t x[T * I], W[4 * H * I], R[4 * H * H];
    int32_t b[4 * H];
    FillPattern(x, T * I, 1, -100, 100);
    FillPattern(W, 4 * H * I, 2, -120, 120);
    FillPattern(R, 4 * H * H, 3, -120, 120);
    FillPattern(b, 4 * H, 4, -50, 50);

    dl::TensorBase t_x({B, T, I}, x, x_exp, dl::DATA_TYPE_INT8);
    dl::TensorBase t_W({4 * H, I}, W, W_exp, dl::DATA_TYPE_INT8);
    dl::TensorBase t_R({4 * H, H}, R, R_exp, dl::DATA_TYPE_INT8);
    dl::TensorBase t_b({4 * H}, b, 0, dl::DATA_TYPE_INT32);
    dl::TensorBase t_out({B, T, H}, nullptr, y_exp, dl::DATA_TYPE_INT8);

    dl::module::SLSTM mod("slstm_s8", H, I, dl::MODULE_NON_INPLACE,
                          dl::QUANT_TYPE_SYMM_8BIT, c_exp, n_exp);
    std::vector<dl::TensorBase*> tensors = {&t_x, &t_W, &t_R, &t_b, &t_out};
    mod.forward(tensors);

    // Same computation, kernel driven directly.
    SlstmS8Params params;
    params.cell_clip = 0.0f;
    params.W_scale = DL_SCALE(W_exp);
    params.R_scale = DL_SCALE(R_exp);
    params.x_quant.scale = DL_SCALE(x_exp); params.x_quant.zero_point = 0;
    params.y_quant.scale = DL_SCALE(y_exp); params.y_quant.zero_point = 0;
    params.c_quant.scale = DL_SCALE(c_exp); params.c_quant.zero_point = 0;
    params.n_quant.scale = DL_SCALE(n_exp); params.n_quant.zero_point = 0;

    int8_t ref_y[H] = {0}, ref_out[T * H] = {0};
    int16_t ref_c[H] = {0}, ref_n[H] = {0};
    float ref_m[H] = {0};
    int32_t scratch[4 * H] = {0};
    slstm_eval_s8(x, W, R, b, ref_y, ref_c, ref_n, ref_m, ref_out, scratch,
                  B, T, I, H, &params);

    bool ok = ExpectEqual("output", ref_out, t_out.get_element_ptr<int8_t>(), T * H);

    // A module that recomputed from zero state each call would pass the
    // check above and fail this one: run both a second time and require
    // they still agree, which they only can if the module carried its
    // state forward the way the kernel did.
    mod.forward(tensors);
    slstm_eval_s8(x, W, R, b, ref_y, ref_c, ref_n, ref_m, ref_out, scratch,
                  B, T, I, H, &params);
    ok &= ExpectEqual("output (2nd call, state carried forward)",
                      ref_out, t_out.get_element_ptr<int8_t>(), T * H);
    return ok;
}

// ---------------------------------------------------------------------------
// mLSTM INT8: as above.
// ---------------------------------------------------------------------------
bool TestMLstmInt8Forward() {
    const int H = 4, I = 4, T = 2, B = 1;
    const int x_exp = -6, W_exp = -7, y_exp = -7;
    const int C_exp = -10, n_exp = -10;
    const int total = 4 * H + 2;

    int8_t x[T * I], W[(4 * 4 + 2) * 4];
    int32_t b[4 * 4 + 2];
    FillPattern(x, T * I, 5, -100, 100);
    FillPattern(W, total * I, 6, -120, 120);
    FillPattern(b, total, 7, -50, 50);

    dl::TensorBase t_x({B, T, I}, x, x_exp, dl::DATA_TYPE_INT8);
    dl::TensorBase t_W({total, I}, W, W_exp, dl::DATA_TYPE_INT8);
    dl::TensorBase t_b({total}, b, 0, dl::DATA_TYPE_INT32);
    dl::TensorBase t_out({B, T, H}, nullptr, y_exp, dl::DATA_TYPE_INT8);

    dl::module::MLSTM mod("mlstm_s8", H, I, dl::MODULE_NON_INPLACE,
                          dl::QUANT_TYPE_SYMM_8BIT, C_exp, n_exp);
    std::vector<dl::TensorBase*> tensors = {&t_x, &t_W, &t_b, &t_out};
    mod.forward(tensors);

    MlstmS8Params params;
    params.cell_clip = 0.0f;
    params.W_scale = DL_SCALE(W_exp);
    params.x_quant.scale = DL_SCALE(x_exp); params.x_quant.zero_point = 0;
    params.y_quant.scale = DL_SCALE(y_exp); params.y_quant.zero_point = 0;
    params.C_quant.scale = DL_SCALE(C_exp); params.C_quant.zero_point = 0;
    params.n_quant.scale = DL_SCALE(n_exp); params.n_quant.zero_point = 0;

    int8_t ref_y[H] = {0}, ref_out[T * H] = {0};
    int16_t ref_C[H * H] = {0}, ref_n[H] = {0};
    float ref_m[1] = {0};
    int32_t scratch[4 * H + 2] = {0};
    mlstm_eval_s8(x, W, b, ref_y, ref_C, ref_n, ref_m, ref_out, scratch,
                  B, T, I, H, &params);

    bool ok = ExpectEqual("output", ref_out, t_out.get_element_ptr<int8_t>(), T * H);

    mod.forward(tensors);
    mlstm_eval_s8(x, W, b, ref_y, ref_C, ref_n, ref_m, ref_out, scratch,
                  B, T, I, H, &params);
    ok &= ExpectEqual("output (2nd call, state carried forward)",
                      ref_out, t_out.get_element_ptr<int8_t>(), T * H);
    return ok;
}

// ---------------------------------------------------------------------------
// Quantized modules: construction, shape and lifecycle, as for the f32 ones.
// The INT8 path allocates a different set of state buffers, so this is not
// redundant with the f32 lifecycle tests.
// ---------------------------------------------------------------------------
bool TestInt8Lifecycle() {
    for (int i = 0; i < 3; i++) {
        auto* s = new dl::module::SLSTM("lifecycle_s8", 2, 2,
                                        dl::MODULE_NON_INPLACE,
                                        dl::QUANT_TYPE_SYMM_8BIT, -10, -10);
        std::vector<std::vector<int>> in_shapes = {{1, 3, 2}};
        std::vector<int> expected = {1, 3, 2};
        if (s->get_output_shape(in_shapes)[0] != expected) {
            printf("  FAIL: wrong sLSTM INT8 output shape\n");
            delete s;
            return false;
        }
        delete s;

        auto* m = new dl::module::MLSTM("lifecycle_s8", 2, 3,
                                        dl::MODULE_NON_INPLACE,
                                        dl::QUANT_TYPE_SYMM_8BIT, -10, -10);
        delete m;
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
    RUN_TEST(TestInt8Lifecycle);
    RUN_TEST(TestSLstmInt8Forward);
    RUN_TEST(TestMLstmInt8Forward);

    printf("\n%d/%d tests passed.\n", g_tests_passed, g_tests_run);

    if (g_tests_passed == g_tests_run) {
        printf("\nPASS\n");
    } else {
        printf("\nFAIL\n");
    }

    // Exit QEMU
    fflush(stdout);
}
