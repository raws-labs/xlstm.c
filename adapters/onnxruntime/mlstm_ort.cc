/* Copyright 2026 RAWS Labs
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
 * mLSTM ONNX Runtime custom op - unpacks ORT tensors and calls core.
 * ===========================================================================*/

#include "mlstm_ort.h"
#include "mlstm.h"
#include "mlstm_s8.h"

#include <cstring>

void MLstmOrtKernel(
    const Ort::Custom::Tensor<float>& x,
    const Ort::Custom::Tensor<float>& W,
    const Ort::Custom::Tensor<float>& b,
    const Ort::Custom::Tensor<float>& y_init,
    const Ort::Custom::Tensor<float>& C_init,
    const Ort::Custom::Tensor<float>& n_init,
    const Ort::Custom::Tensor<float>& m_init,
    Ort::Custom::Tensor<float>& output,
    Ort::Custom::Tensor<float>& y_out,
    Ort::Custom::Tensor<float>& C_out,
    Ort::Custom::Tensor<float>& n_out,
    Ort::Custom::Tensor<float>& m_out)
{
    auto x_shape = x.Shape();
    auto y_shape = y_init.Shape();
    auto C_shape = C_init.Shape();
    auto m_shape = m_init.Shape();

    int batch_size  = static_cast<int>(x_shape[0]);
    int time_steps  = static_cast<int>(x_shape[1]);
    int input_size  = static_cast<int>(x_shape[2]);
    int hidden_size = static_cast<int>(y_shape[1]);

    // Allocate outputs
    float* out_data = output.Allocate({x_shape[0], x_shape[1], y_shape[1]});
    float* y_data   = y_out.Allocate(y_shape);
    float* C_data   = C_out.Allocate(C_shape);
    float* n_data   = n_out.Allocate(y_shape);
    float* m_data   = m_out.Allocate(m_shape);

    // Copy initial states into mutable outputs
    std::memcpy(y_data, y_init.Data(), batch_size * hidden_size * sizeof(float));
    std::memcpy(C_data, C_init.Data(), batch_size * hidden_size * hidden_size * sizeof(float));
    std::memcpy(n_data, n_init.Data(), batch_size * hidden_size * sizeof(float));
    std::memcpy(m_data, m_init.Data(), batch_size * 1 * sizeof(float));

    // Scratch buffer for gate pre-activations
    std::vector<float> scratch(4 * hidden_size + 2);

    MlstmParams params = {0.0f};

    mlstm_eval_f32(
        x.Data(), W.Data(), b.Data(),
        y_data, C_data, n_data, m_data,
        out_data, scratch.data(),
        batch_size, time_steps, input_size, hidden_size,
        &params);
}

void MLstmOrtKernelS8(
    const Ort::Custom::Tensor<int8_t>& x,
    const Ort::Custom::Tensor<float>& x_scale,
    const Ort::Custom::Tensor<int8_t>& x_zero_point,
    const Ort::Custom::Tensor<int8_t>& W,
    const Ort::Custom::Tensor<float>& W_scale,
    const Ort::Custom::Tensor<int32_t>& b,
    const Ort::Custom::Tensor<int8_t>& y_init,
    const Ort::Custom::Tensor<float>& y_scale,
    const Ort::Custom::Tensor<int8_t>& y_zero_point,
    const Ort::Custom::Tensor<int16_t>& C_init,
    const Ort::Custom::Tensor<float>& C_scale,
    const Ort::Custom::Tensor<int16_t>& n_init,
    const Ort::Custom::Tensor<float>& n_scale,
    const Ort::Custom::Tensor<float>& m_init,
    Ort::Custom::Tensor<int8_t>& output,
    Ort::Custom::Tensor<int8_t>& y_out,
    Ort::Custom::Tensor<int16_t>& C_out,
    Ort::Custom::Tensor<int16_t>& n_out,
    Ort::Custom::Tensor<float>& m_out)
{
    auto x_shape = x.Shape();
    auto y_shape = y_init.Shape();
    auto C_shape = C_init.Shape();
    auto m_shape = m_init.Shape();

    int batch_size  = static_cast<int>(x_shape[0]);
    int time_steps  = static_cast<int>(x_shape[1]);
    int input_size  = static_cast<int>(x_shape[2]);
    int hidden_size = static_cast<int>(y_shape[1]);

    // Allocate outputs
    int8_t*  out_data = output.Allocate({x_shape[0], x_shape[1], y_shape[1]});
    int8_t*  y_data   = y_out.Allocate(y_shape);
    int16_t* C_data   = C_out.Allocate(C_shape);
    int16_t* n_data   = n_out.Allocate(y_shape);
    float*   m_data   = m_out.Allocate(m_shape);

    // Copy initial states into mutable outputs
    std::memcpy(y_data, y_init.Data(), batch_size * hidden_size * sizeof(int8_t));
    std::memcpy(C_data, C_init.Data(), batch_size * hidden_size * hidden_size * sizeof(int16_t));
    std::memcpy(n_data, n_init.Data(), batch_size * hidden_size * sizeof(int16_t));
    std::memcpy(m_data, m_init.Data(), batch_size * 1 * sizeof(float));

    // Scratch buffer for gate accumulators (int32 on the quantized path)
    std::vector<int32_t> scratch(4 * hidden_size + 2);

    // Every scale/zero-point is a scalar tensor input: read element 0 and
    // hand it to the kernel. No arithmetic on this side.
    MlstmS8Params params;
    params.cell_clip = 0.0f;
    params.W_scale = W_scale.Data()[0];
    params.x_quant = {x_scale.Data()[0], x_zero_point.Data()[0]};
    params.y_quant = {y_scale.Data()[0], y_zero_point.Data()[0]};
    params.C_quant = {C_scale.Data()[0], 0};  // INT16 states are symmetric
    params.n_quant = {n_scale.Data()[0], 0};

    mlstm_eval_s8(
        x.Data(), W.Data(), b.Data(),
        y_data, C_data, n_data, m_data,
        out_data, scratch.data(),
        batch_size, time_steps, input_size, hidden_size,
        &params);
}
