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
 * mLSTM TFLM adapter — tensor indices, OpData, and registration.
 * ===========================================================================*/

#ifndef MLSTM_TFLM_H_
#define MLSTM_TFLM_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_common.h"

namespace tflite {

// Register mLSTM operator for TFLM
TFLMRegistration Register_MLSTM();

// Input tensor indices for mLSTM (no recurrent weights)
enum MLstmTensorIndex {
    kMLstmInputTensor = 0,              // [batch, time, features]
    kMLstmInputWeightsTensor = 1,       // [(4*hidden+2), input]
    kMLstmBiasTensor = 2,              // [4*hidden+2]
    kMLstmHiddenStateTensor = 3,       // y: [batch, hidden]
    kMLstmCellStateTensor = 4,         // C: [batch, hidden*hidden]
    kMLstmNormalizerStateTensor = 5,   // n: [batch, hidden]
    kMLstmStabilizerStateTensor = 6,   // m: [batch, 1]
    kMLstmNumInputs = 7
};

// Output tensor indices
enum MLstmOutputIndex {
    kMLstmOutputTensor = 0,
    kMLstmNumOutputs = 1
};

// OpData structure for scratch buffers and precomputed values
struct OpDataMLstm {
    int scratch_buffer_index;  // For gate computations [4*hidden+2]

    int batch_size;
    int time_steps;
    int input_size;
    int hidden_size;

    float cell_clip;
};

}  // namespace tflite

#endif  // MLSTM_TFLM_H_
