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
 * sLSTM ESP-DL module - unpacks ESP-DL tensors and calls core.
 * ===========================================================================*/

#include "slstm_espdl.hpp"

#include <cstring>

namespace dl {
namespace module {

SLSTM::SLSTM(const char* name, int hidden_size, int input_size,
              module_inplace_t inplace, quant_type_t quant_type,
              int c_exponent, int n_exponent)
    : Module(name, inplace, quant_type),
      m_hidden_size(hidden_size),
      m_input_size(input_size),
      m_y(nullptr), m_c(nullptr), m_n(nullptr), m_scratch(nullptr),
      m_y_q(nullptr), m_c_q(nullptr), m_n_q(nullptr), m_scratch_q(nullptr),
      m_m(nullptr),
      m_c_exponent(c_exponent), m_n_exponent(n_exponent),
      m_initialized(false)
{
}

SLSTM::~SLSTM() {
    free_states();
}

void SLSTM::init_states() {
    if (m_initialized) return;
    int H = m_hidden_size;
    if (quantized()) {
        m_y_q       = new int8_t[H]();
        m_c_q       = new int16_t[H]();
        m_n_q       = new int16_t[H]();
        m_scratch_q = new int32_t[4 * H]();
    } else {
        m_y       = new float[H]();
        m_c       = new float[H]();
        m_n       = new float[H]();
        m_scratch = new float[4 * H]();
    }
    m_m = new float[H]();
    m_initialized = true;
}

void SLSTM::free_states() {
    delete[] m_y;
    delete[] m_c;
    delete[] m_n;
    delete[] m_scratch;
    delete[] m_y_q;
    delete[] m_c_q;
    delete[] m_n_q;
    delete[] m_scratch_q;
    delete[] m_m;
    m_y = m_c = m_n = m_scratch = nullptr;
    m_y_q = nullptr;
    m_c_q = m_n_q = nullptr;
    m_scratch_q = nullptr;
    m_m = nullptr;
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
    (void)mode;
    init_states();
    if (quantized()) {
        forward_s8(tensors);
    } else {
        forward_f32(tensors);
    }
}

void SLSTM::forward_f32(std::vector<dl::TensorBase*>& tensors) {
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

void SLSTM::forward_s8(std::vector<dl::TensorBase*>& tensors) {
    // Same tensor order as the f32 path; X/W/R/output are int8, b is int32.
    TensorBase* input_x = tensors[0];
    TensorBase* input_W = tensors[1];
    TensorBase* input_R = tensors[2];
    TensorBase* input_b = tensors[3];
    TensorBase* output  = tensors[4];

    auto x_shape = input_x->get_shape();
    int batch_size  = (x_shape.size() == 3) ? x_shape[0] : 1;
    int time_steps  = (x_shape.size() == 3) ? x_shape[1] : x_shape[0];
    int input_size  = x_shape.back();

    // ESP-DL stores a scale as its base-2 exponent; DL_SCALE turns it back
    // into a float. Quantization is symmetric here, so every zero-point is
    // 0 - this is unpacking the framework's representation, not arithmetic.
    SlstmS8Params params;
    params.cell_clip = 0.0f;
    params.W_scale = DL_SCALE(input_W->exponent);
    params.R_scale = DL_SCALE(input_R->exponent);
    params.x_quant.scale      = DL_SCALE(input_x->exponent);
    params.x_quant.zero_point = 0;
    params.y_quant.scale      = DL_SCALE(output->exponent);
    params.y_quant.zero_point = 0;
    params.c_quant.scale      = DL_SCALE(m_c_exponent);
    params.c_quant.zero_point = 0;
    params.n_quant.scale      = DL_SCALE(m_n_exponent);
    params.n_quant.zero_point = 0;

    slstm_eval_s8(
        input_x->get_element_ptr<int8_t>(),
        input_W->get_element_ptr<int8_t>(),
        input_R->get_element_ptr<int8_t>(),
        input_b->get_element_ptr<int32_t>(),
        m_y_q, m_c_q, m_n_q, m_m,
        output->get_element_ptr<int8_t>(),
        m_scratch_q,
        batch_size, time_steps, input_size, m_hidden_size,
        &params);
}

}  // namespace module
}  // namespace dl
