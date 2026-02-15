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
 * mLSTM ESP-DL module — unpacks ESP-DL tensors and calls core.
 * ===========================================================================*/

#include "mlstm_espdl.hpp"

#include <cstring>

namespace dl {
namespace module {

MLSTM::MLSTM(const char* name, int hidden_size, int input_size,
              module_inplace_t inplace, quant_type_t quant_type)
    : Module(name, inplace, quant_type),
      m_hidden_size(hidden_size),
      m_input_size(input_size),
      m_y(nullptr), m_C(nullptr), m_n(nullptr), m_m(nullptr),
      m_scratch(nullptr), m_initialized(false)
{
}

MLSTM::~MLSTM() {
    free_states();
}

void MLSTM::init_states() {
    if (m_initialized) return;
    int H = m_hidden_size;
    m_y       = new float[H]();
    m_C       = new float[H * H]();
    m_n       = new float[H]();
    m_m       = new float[1]();
    m_scratch = new float[4 * H + 2]();
    m_initialized = true;
}

void MLSTM::free_states() {
    delete[] m_y;
    delete[] m_C;
    delete[] m_n;
    delete[] m_m;
    delete[] m_scratch;
    m_y = m_C = m_n = m_m = m_scratch = nullptr;
    m_initialized = false;
}

std::vector<std::vector<int>> MLSTM::get_output_shape(
    std::vector<std::vector<int>>& input_shapes)
{
    auto& x_shape = input_shapes[0];
    std::vector<int> out_shape = x_shape;
    out_shape.back() = m_hidden_size;
    return {out_shape};
}

void MLSTM::forward(std::vector<dl::TensorBase*>& tensors, runtime_mode_t mode) {
    // tensors: [0]=X[B,T,I], [1]=W[4H+2,I], [2]=b[4H+2], [3]=output[B,T,H]
    TensorBase* input_x = tensors[0];
    TensorBase* input_W = tensors[1];
    TensorBase* input_b = tensors[2];
    TensorBase* output  = tensors[3];

    auto x_shape = input_x->get_shape();
    int batch_size  = (x_shape.size() == 3) ? x_shape[0] : 1;
    int time_steps  = (x_shape.size() == 3) ? x_shape[1] : x_shape[0];
    int input_size  = x_shape.back();

    init_states();

    MlstmParams params = {0.0f};

    mlstm_eval_f32(
        input_x->get_element_ptr<float>(),
        input_W->get_element_ptr<float>(),
        input_b->get_element_ptr<float>(),
        m_y, m_C, m_n, m_m,
        output->get_element_ptr<float>(),
        m_scratch,
        batch_size, time_steps, input_size, m_hidden_size,
        &params);
}

}  // namespace module
}  // namespace dl
