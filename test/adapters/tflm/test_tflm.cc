/* TFLM integration test - runs sLSTM and mLSTM custom ops through a real
 * MicroInterpreter with generated .tflite FlatBuffer models.
 *
 * Build: see Dockerfile (compiled against tflite-micro source tree)
 */

#include <cmath>
#include <cstdio>
#include <cstring>

#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "slstm_tflm.h"
#include "mlstm_tflm.h"

#include "slstm_model_data.h"
#include "mlstm_model_data.h"
#include "mlstm_rect_model_data.h"
#include "slstm_s8_model_data.h"
#include "mlstm_s8_model_data.h"
#include "slstm_s8_big_model_data.h"
#include "mlstm_s8_big_model_data.h"
#include "reference_data.h"
#include "s8_case_data.h"

namespace {

constexpr int kArenaSize = 32 * 1024;
alignas(16) uint8_t arena[kArenaSize];

int g_tests_run = 0;
int g_tests_passed = 0;

bool ExpectNear(const char* name, const float* expected,
                const float* actual, int len, float tol) {
    for (int i = 0; i < len; i++) {
        float diff = std::fabs(expected[i] - actual[i]);
        if (diff > tol || std::isnan(actual[i]) || std::isinf(actual[i])) {
            printf("  FAIL %s[%d]: expected %.8f, got %.8f (diff %.8f)\n",
                   name, i, expected[i], actual[i], diff);
            return false;
        }
    }
    return true;
}

// INT8 assertions are exact. The expected integers in s8_case_data.h come
// from generate_reference.py's numpy replica of the quantized kernel, and
// the replica reproduces the C kernel's integers bit for bit on every case
// in reference_data.h. Any drift in the gate math, the requantization or
// the tensor plumbing moves an integer and fails here.
template <typename T>
bool ExpectExact(const char* name, const T* expected, const T* actual, int len) {
    for (int i = 0; i < len; i++) {
        if (expected[i] != actual[i]) {
            printf("  FAIL %s[%d]: expected %d, got %d\n",
                   name, i, (int)expected[i], (int)actual[i]);
            return false;
        }
    }
    return true;
}

void FillTensor(TfLiteTensor* tensor, const float* data, int count) {
    std::memcpy(tensor->data.f, data, count * sizeof(float));
}

void ZeroTensor(TfLiteTensor* tensor, int count) {
    std::memset(tensor->data.f, 0, count * sizeof(float));
}

template <typename T>
void FillQuantTensor(TfLiteTensor* tensor, const T* data, int count) {
    std::memcpy(tensor->data.data, data, count * sizeof(T));
}

void ZeroQuantTensor(TfLiteTensor* tensor, int count, size_t elem) {
    std::memset(tensor->data.data, 0, count * elem);
}

// ---------------------------------------------------------------------------
// sLSTM test: single timestep, zero initial state (Test 1)
// ---------------------------------------------------------------------------
bool TestSLstmSingleTimestep() {
    const tflite::Model* model = tflite::GetModel(slstm_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("  Model schema version mismatch\n");
        return false;
    }

    tflite::MicroMutableOpResolver<1> resolver;
    TFLMRegistration slstm_reg = tflite::Register_SLSTM();
    resolver.AddCustom("SLSTM", &slstm_reg);

    tflite::MicroInterpreter interpreter(model, resolver, arena, kArenaSize);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("  AllocateTensors failed\n");
        return false;
    }

    // B=1, T=1, I=2, H=2
    const int H = 2;

    // Fill inputs: [0]=input, [1]=W, [2]=R, [3]=b, [4]=y, [5]=c, [6]=n, [7]=m
    FillTensor(interpreter.input(0), kTest1_input, 1 * 1 * 2);
    FillTensor(interpreter.input(1), kTest1_W, 8 * 2);
    FillTensor(interpreter.input(2), kTest1_R, 8 * 2);
    FillTensor(interpreter.input(3), kTest1_b, 8);
    ZeroTensor(interpreter.input(4), 1 * H);
    ZeroTensor(interpreter.input(5), 1 * H);
    ZeroTensor(interpreter.input(6), 1 * H);
    ZeroTensor(interpreter.input(7), 1 * H);

    if (interpreter.Invoke() != kTfLiteOk) {
        printf("  Invoke failed\n");
        return false;
    }

    // Output: [B, T, H] - last timestep hidden state should match expected_y
    const float* output = interpreter.output(0)->data.f;

    // For T=1 the output is the same as y
    bool ok = ExpectNear("output", kTest1_expected_y, output, H, 1e-5f);

    // State tensors are updated in-place (they are "inputs" in TFLM)
    ok &= ExpectNear("y", kTest1_expected_y, interpreter.input(4)->data.f, H, 1e-5f);
    ok &= ExpectNear("c", kTest1_expected_c, interpreter.input(5)->data.f, H, 1e-5f);
    ok &= ExpectNear("n", kTest1_expected_n, interpreter.input(6)->data.f, H, 1e-5f);
    ok &= ExpectNear("m", kTest1_expected_m, interpreter.input(7)->data.f, H, 1e-5f);

    return ok;
}

// ---------------------------------------------------------------------------
// mLSTM test: single timestep, zero initial state (Test 1)
// ---------------------------------------------------------------------------
bool TestMLstmSingleTimestep() {
    const tflite::Model* model = tflite::GetModel(mlstm_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("  Model schema version mismatch\n");
        return false;
    }

    tflite::MicroMutableOpResolver<1> resolver;
    TFLMRegistration mlstm_reg = tflite::Register_MLSTM();
    resolver.AddCustom("MLSTM", &mlstm_reg);

    tflite::MicroInterpreter interpreter(model, resolver, arena, kArenaSize);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("  AllocateTensors failed\n");
        return false;
    }

    // B=1, T=1, I=3, H=2
    const int H = 2;

    // Fill inputs: [0]=input, [1]=W, [2]=b, [3]=y, [4]=C, [5]=n, [6]=m
    FillTensor(interpreter.input(0), kMTest1_input, 1 * 1 * 3);
    FillTensor(interpreter.input(1), kMTest1_W, 10 * 3);
    FillTensor(interpreter.input(2), kMTest1_b, 10);
    ZeroTensor(interpreter.input(3), 1 * H);
    ZeroTensor(interpreter.input(4), 1 * H * H);
    ZeroTensor(interpreter.input(5), 1 * H);
    ZeroTensor(interpreter.input(6), 1 * 1);

    if (interpreter.Invoke() != kTfLiteOk) {
        printf("  Invoke failed\n");
        return false;
    }

    const float* output = interpreter.output(0)->data.f;

    bool ok = ExpectNear("output", kMTest1_expected_y, output, H, 1e-5f);
    ok &= ExpectNear("y", kMTest1_expected_y, interpreter.input(3)->data.f, H, 1e-5f);
    ok &= ExpectNear("C", kMTest1_expected_C, interpreter.input(4)->data.f, H * H, 1e-5f);
    ok &= ExpectNear("n", kMTest1_expected_n, interpreter.input(5)->data.f, H, 1e-5f);
    ok &= ExpectNear("m", kMTest1_expected_m, interpreter.input(6)->data.f, 1, 1e-5f);

    return ok;
}

// ---------------------------------------------------------------------------
// mLSTM with the two widths different (DQ=4, DV=12). The adapter reads DV
// off y and DQ off n; a square model cannot tell the two apart, so this is
// the only case that catches taking one where the other belongs.
// ---------------------------------------------------------------------------
bool TestMLstmRectangular() {
    const tflite::Model* model = tflite::GetModel(mlstm_rect_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("  Model schema version mismatch\n");
        return false;
    }

    tflite::MicroMutableOpResolver<1> resolver;
    TFLMRegistration mlstm_reg = tflite::Register_MLSTM();
    resolver.AddCustom("MLSTM", &mlstm_reg);

    tflite::MicroInterpreter interpreter(model, resolver, arena, kArenaSize);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("  AllocateTensors failed\n");
        return false;
    }

    // B=1, T=3, I=12, DQ=4, DV=12
    const int T = 3, I = 12, DQ = 4, DV = 12;
    const int R = 2 * DQ + 2 * DV + 2;

    FillTensor(interpreter.input(0), kRectM4x12_input, 1 * T * I);
    FillTensor(interpreter.input(1), kRectM4x12_W, R * I);
    FillTensor(interpreter.input(2), kRectM4x12_b, R);
    ZeroTensor(interpreter.input(3), 1 * DV);
    ZeroTensor(interpreter.input(4), 1 * DQ * DV);
    ZeroTensor(interpreter.input(5), 1 * DQ);
    ZeroTensor(interpreter.input(6), 1 * 1);

    if (interpreter.Invoke() != kTfLiteOk) {
        printf("  Invoke failed\n");
        return false;
    }

    bool ok = ExpectNear("output", kRectM4x12_expected_output,
                         interpreter.output(0)->data.f, T * DV, 1e-5f);
    ok &= ExpectNear("y", kRectM4x12_expected_y,
                     interpreter.input(3)->data.f, DV, 1e-5f);
    ok &= ExpectNear("C", kRectM4x12_expected_C,
                     interpreter.input(4)->data.f, DQ * DV, 1e-5f);
    ok &= ExpectNear("n", kRectM4x12_expected_n,
                     interpreter.input(5)->data.f, DQ, 1e-5f);
    ok &= ExpectNear("m", kRectM4x12_expected_m,
                     interpreter.input(6)->data.f, 1, 1e-5f);
    return ok;
}

// ---------------------------------------------------------------------------
// INT8 cases. Both cells run test1 (H=2, T=1) and test5 (H=8, T=3): test1
// alone leaves the gate blind to small numerical drift, since a change too
// small to move a 2-channel single-timestep integer still moves test5's.
// ---------------------------------------------------------------------------
struct SLstmS8Case {
    const char* name;
    const unsigned char* model;
    int T, I, H;
    const int8_t *x_q, *W_q, *R_q;
    const int32_t* b_q;
    const int8_t *out_q, *y_q;
    const int16_t *c_q, *n_q;
    const float* m;
};

struct MLstmS8Case {
    const char* name;
    const unsigned char* model;
    int T, I, H;
    const int8_t *x_q, *W_q;
    const int32_t* b_q;
    const int8_t *out_q, *y_q;
    const int16_t *C_q, *n_q;
    const float* m;
};

bool RunSLstmS8(const SLstmS8Case& tc) {
    const tflite::Model* model = tflite::GetModel(tc.model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("  %s: model schema version mismatch\n", tc.name);
        return false;
    }

    tflite::MicroMutableOpResolver<1> resolver;
    TFLMRegistration slstm_reg = tflite::Register_SLSTM();
    resolver.AddCustom("SLSTM", &slstm_reg);

    tflite::MicroInterpreter interpreter(model, resolver, arena, kArenaSize);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("  %s: AllocateTensors failed\n", tc.name);
        return false;
    }
    if (interpreter.input(0)->type != kTfLiteInt8) {
        printf("  %s: input tensor is not INT8 - model is not quantized\n", tc.name);
        return false;
    }

    const int H = tc.H;
    FillQuantTensor(interpreter.input(0), tc.x_q, tc.T * tc.I);
    FillQuantTensor(interpreter.input(1), tc.W_q, 4 * H * tc.I);
    FillQuantTensor(interpreter.input(2), tc.R_q, 4 * H * H);
    FillQuantTensor(interpreter.input(3), tc.b_q, 4 * H);
    ZeroQuantTensor(interpreter.input(4), H, sizeof(int8_t));
    ZeroQuantTensor(interpreter.input(5), H, sizeof(int16_t));
    ZeroQuantTensor(interpreter.input(6), H, sizeof(int16_t));
    ZeroQuantTensor(interpreter.input(7), H, sizeof(float));

    if (interpreter.Invoke() != kTfLiteOk) {
        printf("  %s: Invoke failed\n", tc.name);
        return false;
    }

    bool ok = ExpectExact("output", tc.out_q, interpreter.output(0)->data.int8, tc.T * H);
    ok &= ExpectExact("y", tc.y_q, interpreter.input(4)->data.int8, H);
    ok &= ExpectExact("c", tc.c_q, interpreter.input(5)->data.i16, H);
    ok &= ExpectExact("n", tc.n_q, interpreter.input(6)->data.i16, H);
    // m is the log-space stabilizer: float32 even on the INT8 path.
    ok &= ExpectNear("m", tc.m, interpreter.input(7)->data.f, H, 1e-5f);
    if (!ok) printf("  (case %s)\n", tc.name);
    return ok;
}

bool RunMLstmS8(const MLstmS8Case& tc) {
    const tflite::Model* model = tflite::GetModel(tc.model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("  %s: model schema version mismatch\n", tc.name);
        return false;
    }

    tflite::MicroMutableOpResolver<1> resolver;
    TFLMRegistration mlstm_reg = tflite::Register_MLSTM();
    resolver.AddCustom("MLSTM", &mlstm_reg);

    tflite::MicroInterpreter interpreter(model, resolver, arena, kArenaSize);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("  %s: AllocateTensors failed\n", tc.name);
        return false;
    }
    if (interpreter.input(0)->type != kTfLiteInt8) {
        printf("  %s: input tensor is not INT8 - model is not quantized\n", tc.name);
        return false;
    }

    const int H = tc.H;
    FillQuantTensor(interpreter.input(0), tc.x_q, tc.T * tc.I);
    FillQuantTensor(interpreter.input(1), tc.W_q, (4 * H + 2) * tc.I);
    FillQuantTensor(interpreter.input(2), tc.b_q, 4 * H + 2);
    ZeroQuantTensor(interpreter.input(3), H, sizeof(int8_t));
    ZeroQuantTensor(interpreter.input(4), H * H, sizeof(int16_t));
    ZeroQuantTensor(interpreter.input(5), H, sizeof(int16_t));
    ZeroQuantTensor(interpreter.input(6), 1, sizeof(float));

    if (interpreter.Invoke() != kTfLiteOk) {
        printf("  %s: Invoke failed\n", tc.name);
        return false;
    }

    bool ok = ExpectExact("output", tc.out_q, interpreter.output(0)->data.int8, tc.T * H);
    ok &= ExpectExact("y", tc.y_q, interpreter.input(3)->data.int8, H);
    ok &= ExpectExact("C", tc.C_q, interpreter.input(4)->data.i16, H * H);
    ok &= ExpectExact("n", tc.n_q, interpreter.input(5)->data.i16, H);
    ok &= ExpectNear("m", tc.m, interpreter.input(6)->data.f, 1, 1e-5f);
    if (!ok) printf("  (case %s)\n", tc.name);
    return ok;
}

#define SLSTM_S8_CASE(nm, model, T, I, H, p)                              \
    SLstmS8Case{nm, model, T, I, H, p##_x_q, p##_W_q, p##_R_q, p##_b_q,   \
                p##_expected_output_q, p##_expected_y_q,                  \
                p##_expected_c_q, p##_expected_n_q, p##_expected_m}

#define MLSTM_S8_CASE(nm, model, T, I, H, p)                              \
    MLstmS8Case{nm, model, T, I, H, p##_x_q, p##_W_q, p##_b_q,            \
                p##_expected_output_q, p##_expected_y_q,                  \
                p##_expected_C_q, p##_expected_n_q, p##_expected_m}

bool TestSLstmInt8() {
    bool ok = RunSLstmS8(SLSTM_S8_CASE("slstm test1", slstm_s8_model_data,
                                       1, 2, 2, kS8Test1));
    ok &= RunSLstmS8(SLSTM_S8_CASE("slstm test5", slstm_s8_big_model_data,
                                   3, 8, 8, kS8Test5));
    return ok;
}

bool TestMLstmInt8() {
    bool ok = RunMLstmS8(MLSTM_S8_CASE("mlstm test1", mlstm_s8_model_data,
                                       1, 3, 2, kMS8Test1));
    ok &= RunMLstmS8(MLSTM_S8_CASE("mlstm test5", mlstm_s8_big_model_data,
                                   3, 8, 8, kMS8Test5));
    return ok;
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

int main() {
    printf("=== TFLM integration tests ===\n\n");

    RUN_TEST(TestSLstmSingleTimestep);
    RUN_TEST(TestMLstmSingleTimestep);
    RUN_TEST(TestMLstmRectangular);
    RUN_TEST(TestSLstmInt8);
    RUN_TEST(TestMLstmInt8);

    printf("\n%d/%d tests passed.\n", g_tests_passed, g_tests_run);
    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
