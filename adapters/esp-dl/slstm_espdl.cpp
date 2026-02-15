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
 * sLSTM ESP-DL module — unpacks ESP-DL tensors and calls core.
 * ===========================================================================*/

#include "slstm_espdl.hpp"

#include <cstring>

namespace dl {
namespace module {

SLSTM::SLSTM(const char* name, int hidden_size, int input_size,
              module_inplace_t inplace, quant_type_t quant_type)
    : Module(name, inplace, quant_type),
      m_hidden_size(hidden_size),
      m_input_size(input_size),
      m_y(nullptr), m_c(nullptr), m_n(nullptr), m_m(nullptr),
      m_scratch(nullptr), m_initialized(false)
{
}

SLSTM::~SLSTM() {
    free_states();
}

void SLSTM::init_states() {
    if (m_initialized) return;
    int H = m_hidden_size;
    m_y       = new float[H]();
    m_c       = new float[H]();
    m_n       = new float[H]();
    m_m       = new float[H]();
    m_scratch = new float[4 * H]();
    m_initialized = true;
}

void SLSTM::free_states() {
    delete[] m_y;
    delete[] m_c;
    delete[] m_n;
    delete[] m_m;
    delete[] m_scratch;
    m_y = m_c = m_n = m_m = m_scratch = nullptr;
    m_initialized = false;
}

std::vector<std::vector<int>> SLSTM::get_output_shape(
    std::vector<std::vector<int>>& input_shapes)
{
    // Input X shape: [B, T, I] or [T, I]
    auto& x_shape = input_shapes[0];
    std::vector<int> out_shape = x_shape;
    out_shape.back() = m_hidden_size;
    return {out_shape};
}

void SLSTM::forward(std::vector<dl::TensorBase*>& tensors, runtime_mode_t mode) {
    // tensors: [0]=X[B,T,I], [1]=W[4H,I], [2]=R[4H,H], [3]=b[4H], [4]=output[B,T,H]
    TensorBase* input_x = tensors[0];
    TensorBase* input_W = tensors[1];
    TensorBase* input_R = tensors[2];
    TensorBase* input_b = tensors[3];
    TensorBase* output  = tensors[4];

    auto x_shape = input_x->get_shape();
    int batch_size  = (x_shape.size() == 3) ? x_shape[0] : 1;
    int time_steps  = (x_shape.size() == 3) ? x_shape[1] : x_shape[0];
    int input_size  = x_shape.back();

    init_states();

    SlstmParams params = {0.0f};

    slstm_eval_f32(
        input_x->get_element_ptr<float>(),
        input_W->get_element_ptr<float>(),
        input_R->get_element_ptr<float>(),
        input_b->get_element_ptr<float>(),
        m_y, m_c, m_n, m_m,
        output->get_element_ptr<float>(),
        m_scratch,
        batch_size, time_steps, input_size, m_hidden_size,
        &params);
}

}  // namespace module
}  // namespace dl
