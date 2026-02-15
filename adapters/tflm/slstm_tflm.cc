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
 * sLSTM TFLM adapter — thin wrapper that unpacks tensors and calls core.
 * ===========================================================================*/

#include "slstm_tflm.h"

#include "slstm.h"

#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

void* SLstmInit(TfLiteContext* context, const char* buffer, size_t length) {
    TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
    return context->AllocatePersistentBuffer(context, sizeof(OpDataSLstm));
}

TfLiteStatus SLstmPrepare(TfLiteContext* context, TfLiteNode* node) {
    TFLITE_DCHECK(node->user_data != nullptr);
    OpDataSLstm* op_data = static_cast<OpDataSLstm*>(node->user_data);

    TF_LITE_ENSURE_EQ(context, NumInputs(node), kSLstmNumInputs);
    TF_LITE_ENSURE_EQ(context, NumOutputs(node), kSLstmNumOutputs);

    MicroContext* micro_context = GetMicroContext(context);
    TfLiteTensor* input =
        micro_context->AllocateTempInputTensor(node, kSLstmInputTensor);
    TF_LITE_ENSURE(context, input != nullptr);
    TF_LITE_ENSURE_EQ(context, NumDimensions(input), 3);

    TfLiteTensor* hidden_state =
        micro_context->AllocateTempInputTensor(node, kSLstmHiddenStateTensor);
    TF_LITE_ENSURE(context, hidden_state != nullptr);

    op_data->batch_size = input->dims->data[0];
    op_data->time_steps = input->dims->data[1];
    op_data->input_size = input->dims->data[2];
    op_data->hidden_size = hidden_state->dims->data[1];

    micro_context->DeallocateTempTfLiteTensor(input);
    micro_context->DeallocateTempTfLiteTensor(hidden_state);

    op_data->cell_clip = 0.0f;

    TF_LITE_ENSURE_OK(
        context,
        context->RequestScratchBufferInArena(
            context, 4 * op_data->hidden_size * sizeof(float),
            &op_data->scratch_buffer_index));

    return kTfLiteOk;
}

TfLiteStatus SLstmEvalFloat(TfLiteContext* context, TfLiteNode* node,
                            const OpDataSLstm* op_data) {
    MicroContext* micro_context = GetMicroContext(context);

    // Unpack tensors
    TfLiteTensor* input =
        micro_context->AllocateTempInputTensor(node, kSLstmInputTensor);
    TfLiteTensor* input_weights =
        micro_context->AllocateTempInputTensor(node, kSLstmInputWeightsTensor);
    TfLiteTensor* recurrent_weights =
        micro_context->AllocateTempInputTensor(node, kSLstmRecurrentWeightsTensor);
    TfLiteTensor* bias =
        micro_context->AllocateTempInputTensor(node, kSLstmBiasTensor);
    TfLiteTensor* hidden_state =
        micro_context->AllocateTempInputTensor(node, kSLstmHiddenStateTensor);
    TfLiteTensor* cell_state =
        micro_context->AllocateTempInputTensor(node, kSLstmCellStateTensor);
    TfLiteTensor* normalizer_state =
        micro_context->AllocateTempInputTensor(node, kSLstmNormalizerStateTensor);
    TfLiteTensor* stabilizer_state =
        micro_context->AllocateTempInputTensor(node, kSLstmStabilizerStateTensor);
    TfLiteTensor* output =
        micro_context->AllocateTempOutputTensor(node, kSLstmOutputTensor);

    float* scratch = static_cast<float*>(
        context->GetScratchBuffer(context, op_data->scratch_buffer_index));

    // Set up core params
    SlstmParams params;
    params.cell_clip = op_data->cell_clip;

    // Call portable core
    slstm_eval_f32(
        GetTensorData<float>(input),
        GetTensorData<float>(input_weights),
        GetTensorData<float>(recurrent_weights),
        GetTensorData<float>(bias),
        GetTensorData<float>(hidden_state),
        GetTensorData<float>(cell_state),
        GetTensorData<float>(normalizer_state),
        GetTensorData<float>(stabilizer_state),
        GetTensorData<float>(output),
        scratch,
        op_data->batch_size,
        op_data->time_steps,
        op_data->input_size,
        op_data->hidden_size,
        &params);

    // Deallocate temp tensors
    micro_context->DeallocateTempTfLiteTensor(input);
    micro_context->DeallocateTempTfLiteTensor(input_weights);
    micro_context->DeallocateTempTfLiteTensor(recurrent_weights);
    micro_context->DeallocateTempTfLiteTensor(bias);
    micro_context->DeallocateTempTfLiteTensor(hidden_state);
    micro_context->DeallocateTempTfLiteTensor(cell_state);
    micro_context->DeallocateTempTfLiteTensor(normalizer_state);
    micro_context->DeallocateTempTfLiteTensor(stabilizer_state);
    micro_context->DeallocateTempTfLiteTensor(output);

    return kTfLiteOk;
}

TfLiteStatus SLstmEval(TfLiteContext* context, TfLiteNode* node) {
    TFLITE_DCHECK(node->user_data != nullptr);
    const OpDataSLstm* op_data =
        static_cast<const OpDataSLstm*>(node->user_data);

    MicroContext* micro_context = GetMicroContext(context);
    TfLiteTensor* input =
        micro_context->AllocateTempInputTensor(node, kSLstmInputTensor);
    TfLiteType input_type = input->type;
    micro_context->DeallocateTempTfLiteTensor(input);

    switch (input_type) {
        case kTfLiteFloat32:
            return SLstmEvalFloat(context, node, op_data);
        default:
            MicroPrintf("Type %s (%d) not supported for sLSTM.",
                        TfLiteTypeGetName(input_type), input_type);
            return kTfLiteError;
    }
}

}  // namespace

TFLMRegistration Register_SLSTM() {
    return tflite::micro::RegisterOp(SLstmInit, SLstmPrepare, SLstmEval);
}

}  // namespace tflite
