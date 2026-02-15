/* TFLM integration test — runs sLSTM and mLSTM custom ops through a real
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
#include "reference_data.h"

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

void FillTensor(TfLiteTensor* tensor, const float* data, int count) {
    std::memcpy(tensor->data.f, data, count * sizeof(float));
}

void ZeroTensor(TfLiteTensor* tensor, int count) {
    std::memset(tensor->data.f, 0, count * sizeof(float));
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

    // Output: [B, T, H] — last timestep hidden state should match expected_y
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

    printf("\n%d/%d tests passed.\n", g_tests_passed, g_tests_run);
    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
