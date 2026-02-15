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
 * mLSTM ONNX Runtime custom op — unpacks ORT tensors and calls core.
 * ===========================================================================*/

#include "mlstm_ort.h"
#include "mlstm.h"

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
