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
 * sLSTM ONNX Runtime custom op - unpacks ORT tensors and calls core.
 * ===========================================================================*/

#include "slstm_ort.h"
#include "slstm.h"
#include "slstm_s8.h"

#include <cstring>

void SLstmOrtKernel(
    const Ort::Custom::Tensor<float>& x,
    const Ort::Custom::Tensor<float>& W,
    const Ort::Custom::Tensor<float>& R,
    const Ort::Custom::Tensor<float>& b,
    const Ort::Custom::Tensor<float>& y_init,
    const Ort::Custom::Tensor<float>& c_init,
    const Ort::Custom::Tensor<float>& n_init,
    const Ort::Custom::Tensor<float>& m_init,
    Ort::Custom::Tensor<float>& output,
    Ort::Custom::Tensor<float>& y_out,
    Ort::Custom::Tensor<float>& c_out,
    Ort::Custom::Tensor<float>& n_out,
    Ort::Custom::Tensor<float>& m_out)
{
    auto x_shape = x.Shape();
    auto y_shape = y_init.Shape();

    int batch_size  = static_cast<int>(x_shape[0]);
    int time_steps  = static_cast<int>(x_shape[1]);
    int input_size  = static_cast<int>(x_shape[2]);
    int hidden_size = static_cast<int>(y_shape[1]);

    // Allocate outputs
    float* out_data = output.Allocate({x_shape[0], x_shape[1], y_shape[1]});
    float* y_data   = y_out.Allocate(y_shape);
    float* c_data   = c_out.Allocate(y_shape);
    float* n_data   = n_out.Allocate(y_shape);
    float* m_data   = m_out.Allocate(y_shape);

    // Copy initial states into mutable outputs
    std::memcpy(y_data, y_init.Data(), batch_size * hidden_size * sizeof(float));
    std::memcpy(c_data, c_init.Data(), batch_size * hidden_size * sizeof(float));
    std::memcpy(n_data, n_init.Data(), batch_size * hidden_size * sizeof(float));
    std::memcpy(m_data, m_init.Data(), batch_size * hidden_size * sizeof(float));

    // Scratch buffer for gate pre-activations
    std::vector<float> scratch(4 * hidden_size);

    SlstmParams params = {0.0f};

    slstm_eval_f32(
        x.Data(), W.Data(), R.Data(), b.Data(),
        y_data, c_data, n_data, m_data,
        out_data, scratch.data(),
        batch_size, time_steps, input_size, hidden_size,
        &params);
}

void SLstmOrtKernelS8(
    const Ort::Custom::Tensor<int8_t>& x,
    const Ort::Custom::Tensor<float>& x_scale,
    const Ort::Custom::Tensor<int8_t>& x_zero_point,
    const Ort::Custom::Tensor<int8_t>& W,
    const Ort::Custom::Tensor<float>& W_scale,
    const Ort::Custom::Tensor<int8_t>& R,
    const Ort::Custom::Tensor<float>& R_scale,
    const Ort::Custom::Tensor<int32_t>& b,
    const Ort::Custom::Tensor<int8_t>& y_init,
    const Ort::Custom::Tensor<float>& y_scale,
    const Ort::Custom::Tensor<int8_t>& y_zero_point,
    const Ort::Custom::Tensor<int16_t>& c_init,
    const Ort::Custom::Tensor<float>& c_scale,
    const Ort::Custom::Tensor<int16_t>& n_init,
    const Ort::Custom::Tensor<float>& n_scale,
    const Ort::Custom::Tensor<float>& m_init,
    Ort::Custom::Tensor<int8_t>& output,
    Ort::Custom::Tensor<int8_t>& y_out,
    Ort::Custom::Tensor<int16_t>& c_out,
    Ort::Custom::Tensor<int16_t>& n_out,
    Ort::Custom::Tensor<float>& m_out)
{
    auto x_shape = x.Shape();
    auto y_shape = y_init.Shape();

    int batch_size  = static_cast<int>(x_shape[0]);
    int time_steps  = static_cast<int>(x_shape[1]);
    int input_size  = static_cast<int>(x_shape[2]);
    int hidden_size = static_cast<int>(y_shape[1]);

    // Allocate outputs
    int8_t*  out_data = output.Allocate({x_shape[0], x_shape[1], y_shape[1]});
    int8_t*  y_data   = y_out.Allocate(y_shape);
    int16_t* c_data   = c_out.Allocate(y_shape);
    int16_t* n_data   = n_out.Allocate(y_shape);
    float*   m_data   = m_out.Allocate(y_shape);

    // Copy initial states into mutable outputs
    std::memcpy(y_data, y_init.Data(), batch_size * hidden_size * sizeof(int8_t));
    std::memcpy(c_data, c_init.Data(), batch_size * hidden_size * sizeof(int16_t));
    std::memcpy(n_data, n_init.Data(), batch_size * hidden_size * sizeof(int16_t));
    std::memcpy(m_data, m_init.Data(), batch_size * hidden_size * sizeof(float));

    // Scratch buffer for gate accumulators (int32 on the quantized path)
    std::vector<int32_t> scratch(4 * hidden_size);

    // Every scale/zero-point is a scalar tensor input: read element 0 and
    // hand it to the kernel. No arithmetic on this side.
    SlstmS8Params params;
    params.cell_clip = 0.0f;
    params.W_scale = W_scale.Data()[0];
    params.R_scale = R_scale.Data()[0];
    params.x_quant = {x_scale.Data()[0], x_zero_point.Data()[0]};
    params.y_quant = {y_scale.Data()[0], y_zero_point.Data()[0]};
    params.c_quant = {c_scale.Data()[0], 0};  // INT16 states are symmetric
    params.n_quant = {n_scale.Data()[0], 0};

    slstm_eval_s8(
        x.Data(), W.Data(), R.Data(), b.Data(),
        y_data, c_data, n_data, m_data,
        out_data, scratch.data(),
        batch_size, time_steps, input_size, hidden_size,
        &params);
}
