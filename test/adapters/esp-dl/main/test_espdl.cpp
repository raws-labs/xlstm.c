/* ESP-DL integration test - exercises sLSTM/mLSTM Module subclasses.
 *
 * WHAT THIS GATES, precisely. It runs on an emulated ESP32-S3 against real
 * ESP-DL, and it checks this adapter's own dispatch: that forward() reads
 * the tensors it was handed in the order the module documents, turns each
 * tensor's exponent into the scale the kernel expects, and carries its
 * states across calls. The INT8 tests build real TensorBase objects, run
 * the module, and require the result to equal what slstm_eval_s8 /
 * mlstm_eval_s8 produce when called directly with the scales DL_SCALE
 * derives from the same exponents. That is exact - integers, no tolerance -
 * and fails on any tensor, exponent or state wired to the wrong place.
 *
 * WHAT IT IS NOT: a check against the PyTorch-derived golden vectors, the
 * way the ONNX Runtime, microTVM and TFLM adapter suites are. It cannot be
 * one. ESP-DL quantization is power-of-two symmetric with no zero-point,
 * and that is structural rather than a setting: the .espdl FlatBuffers
 * schema carries an `exponents` field and has no scale or zero-point field
 * at all, and the exporter discards the float scale as
 * exponent = int(log2(scale)). So the framework cannot express the
 * reference calibration - arbitrary scales, asymmetric input zero-point -
 * that reference_data.json's expected integers were produced with. Checking
 * the adapter against the core kernels driven with matching scales is
 * exactly the claim Espressif make for their own operators, and it is the
 * claim here.
 *
 * The kernels underneath, and the esp SIMD backend they call, are a
 * separate question and are gated separately: `make test-esp` runs the full
 * golden vector set on this same emulated part.
 *
 * Bit-exactness here is per-target, not per-framework: ESP-DL rounds
 * ROUND_HALF_UP on ESP32 and ESP32-S3 and ROUND_HALF_EVEN on ESP32-P4.
 *
 * Build and run: see Dockerfile (ESP-IDF project, esp32s3, under QEMU).
 */

#include <cstdio>
#include <cstring>
#include <vector>

#include "slstm_espdl.hpp"
#include "mlstm_espdl.hpp"

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

// A fixture that computes all zeros compares equal to almost any wrong
// wiring, so the equality checks below are only worth what this guard says
// they are.
template <typename T>
bool ExpectNonTrivial(const T* v, int n) {
    for (int i = 0; i < n; i++) {
        if (v[i] != 0) return true;
    }
    printf("  FAIL: fixture output is all zero - the check is vacuous\n");
    return false;
}

// ---------------------------------------------------------------------------
// sLSTM: constructor + get_output_shape
// ---------------------------------------------------------------------------
bool TestSLstmConstruction() {
    // H != I deliberately: with both 2, an output shape built from the input
    // width instead of the hidden width is the same shape, and the check
    // below passes it.
    const int H = 2, I = 3;
    dl::module::SLSTM slstm("test_slstm", H, I);

    if (slstm.m_hidden_size != H || slstm.m_input_size != I) {
        printf("  FAIL: wrong dimensions %d,%d (expected %d,%d)\n",
               slstm.m_hidden_size, slstm.m_input_size, H, I);
        return false;
    }

    // Test get_output_shape: input [B=1, T=3, I=3] -> output [1, 3, H=2]
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
// sLSTM: construct and destroy, repeatedly. Narrow on purpose and narrower
// than it looks: the module allocates its states on the first forward(), and
// this never calls one, so all it can catch is a constructor or destructor
// that traps. No wiring mistake fails it. Same for the mLSTM one below.
// ---------------------------------------------------------------------------
bool TestSLstmLifecycle() {
    for (int i = 0; i < 3; i++) {
        dl::module::SLSTM* slstm = new dl::module::SLSTM("lifecycle", 2, 3);
        delete slstm;
    }
    return true;
}

// ---------------------------------------------------------------------------
// mLSTM: as above.
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
    // Exponents are ESP-DL's scale representation: scale = 2^exponent. No two
    // of them are equal, so that reading the wrong one is visible: with
    // W_exp == R_exp a swapped scale computes the same numbers and the check
    // below passes a module that is wired backwards.
    const int x_exp = -6, W_exp = -7, R_exp = -5, y_exp = -7;
    // c and n are far enough apart that swapping them saturates n rather than
    // just rounding it differently. Closer together the swap is invisible:
    // both sides dequantize with whatever scale they quantized with, so it
    // survives as rounding error and no int8 output moves.
    const int c_exp = -14, n_exp = -10;

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

    // TensorBase deep-copies what it is handed, so the module reads ESP-DL's
    // copy rather than these arrays. Check the copy first: if it is not the
    // source, every comparison below is a numeric mystery rather than a
    // verdict on the adapter. Not hypothetical - one emulator build got this
    // wrong; see the Dockerfile on which one and why the version is pinned.
    bool ok = ExpectEqual("x tensor", x, t_x.get_element_ptr<int8_t>(), T * I);
    ok &= ExpectEqual("W tensor", W, t_W.get_element_ptr<int8_t>(), 4 * H * I);
    ok &= ExpectEqual("R tensor", R, t_R.get_element_ptr<int8_t>(), 4 * H * H);
    ok &= ExpectEqual("b tensor", b, t_b.get_element_ptr<int32_t>(), 4 * H);

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

    ok &= ExpectNonTrivial(ref_out, T * H);
    ok &= ExpectEqual("output", ref_out, t_out.get_element_ptr<int8_t>(), T * H);

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
    // Distinct, as above. y_exp is two steps coarser than the sLSTM case
    // because at 2^-7 this fixture's outputs clip at +-128, and a clipped
    // output hides a wrong one.
    const int x_exp = -6, W_exp = -7, y_exp = -5;
    const int C_exp = -10, n_exp = -8;
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

    // ESP-DL's copy of the inputs, checked before the module reads it, for the
    // reason given in the sLSTM case above.
    bool ok = ExpectEqual("x tensor", x, t_x.get_element_ptr<int8_t>(), T * I);
    ok &= ExpectEqual("W tensor", W, t_W.get_element_ptr<int8_t>(), total * I);
    ok &= ExpectEqual("b tensor", b, t_b.get_element_ptr<int32_t>(), total);

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

    ok &= ExpectNonTrivial(ref_out, T * H);
    ok &= ExpectEqual("output", ref_out, t_out.get_element_ptr<int8_t>(), T * H);

    mod.forward(tensors);
    mlstm_eval_s8(x, W, b, ref_y, ref_C, ref_n, ref_m, ref_out, scratch,
                  B, T, I, H, &params);
    ok &= ExpectEqual("output (2nd call, state carried forward)",
                      ref_out, t_out.get_element_ptr<int8_t>(), T * H);
    return ok;
}

// ---------------------------------------------------------------------------
// Quantized modules: construction, shape and destruction. The shape half is
// the part that can fail - a module built with a quant_type still has to
// report the hidden width, not the input width.
// ---------------------------------------------------------------------------
bool TestInt8Lifecycle() {
    for (int i = 0; i < 3; i++) {
        // H = 2, I = 3, distinct for the reason TestSLstmConstruction gives.
        auto* s = new dl::module::SLSTM("lifecycle_s8", 2, 3,
                                        dl::MODULE_NON_INPLACE,
                                        dl::QUANT_TYPE_SYMM_8BIT, -10, -8);
        std::vector<std::vector<int>> in_shapes = {{1, 3, 3}};
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

// A verdict a human reads is not a gate. QEMU's -semihosting implements the
// Xtensa simcall ABI - a2 = 1 (SYS_exit), a3 = status - and exits the
// emulator with that status, which is how newlib's own _exit reaches the
// host on this target. So the process running QEMU sees red as non-zero.
void qemu_exit(int status) {
    register int nr __asm__("a2") = 1;
    register int arg __asm__("a3") = status;
    __asm__ volatile("simcall" : "+r"(nr), "+r"(arg) :: "memory");
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
    printf("=== ESP-DL adapter dispatch tests (esp32s3, QEMU) ===\n");
    printf("Scope: this adapter's own dispatch - tensor order, exponent to\n");
    printf("scale, state carried across calls - against the core kernels\n");
    printf("driven with matching scales. NOT a golden-value check against\n");
    printf("the PyTorch reference: ESP-DL quantization is power-of-two\n");
    printf("symmetric with no zero-point and cannot express that\n");
    printf("calibration. The kernels themselves: make test-esp.\n\n");

    RUN_TEST(TestSLstmConstruction);
    RUN_TEST(TestMLstmConstruction);
    RUN_TEST(TestSLstmLifecycle);
    RUN_TEST(TestMLstmLifecycle);
    RUN_TEST(TestInt8Lifecycle);
    RUN_TEST(TestSLstmInt8Forward);
    RUN_TEST(TestMLstmInt8Forward);

    printf("\n%d/%d tests passed.\n", g_tests_passed, g_tests_run);

    bool ok = (g_tests_passed == g_tests_run);
    printf("\n%s\n", ok ? "PASS" : "FAIL");
    fflush(stdout);
    qemu_exit(ok ? 0 : 1);
}
