/* Copyright 2026 RAWS labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =========================================================================
 * sLSTM TFLM adapter — tensor indices, OpData, and registration.
 * ===========================================================================*/

#ifndef SLSTM_TFLM_H_
#define SLSTM_TFLM_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_common.h"

namespace tflite {

// Register sLSTM operator for TFLM
TFLMRegistration Register_SLSTM();

// Input tensor indices for sLSTM
enum SLstmTensorIndex {
    kSLstmInputTensor = 0,              // [batch, time, features]
    kSLstmInputWeightsTensor = 1,       // [4*hidden, input]
    kSLstmRecurrentWeightsTensor = 2,   // [4*hidden, hidden]
    kSLstmBiasTensor = 3,              // [4*hidden]
    kSLstmHiddenStateTensor = 4,       // y: [batch, hidden]
    kSLstmCellStateTensor = 5,         // c: [batch, hidden]
    kSLstmNormalizerStateTensor = 6,   // n: [batch, hidden]
    kSLstmStabilizerStateTensor = 7,   // m: [batch, hidden]
    kSLstmNumInputs = 8
};

// Output tensor indices
enum SLstmOutputIndex {
    kSLstmOutputTensor = 0,
    kSLstmNumOutputs = 1
};

// OpData structure for scratch buffers and precomputed values
struct OpDataSLstm {
    int scratch_buffer_index;  // For gate computations [4 * hidden]

    int batch_size;
    int time_steps;
    int input_size;
    int hidden_size;

    float cell_clip;
};

}  // namespace tflite

#endif  // SLSTM_TFLM_H_
