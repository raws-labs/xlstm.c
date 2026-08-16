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
 * sLSTM ONNX Runtime custom op - lite API.
 *
 * Inputs:  X[B,T,I], W[4H,I], R[4H,H], b[4H],
 *          y_init[B,H], c_init[B,H], n_init[B,H], m_init[B,H]
 * Outputs: output[B,T,H], y[B,H], c[B,H], n[B,H], m[B,H]
 *
 * SLSTM_S8 is the quantized variant. Scale and zero-point arrive as scalar
 * tensor inputs next to the tensor they belong to, the way ONNX's own
 * QLinearConv / QLinearMatMul carry them - not as attributes, so a
 * calibration step can feed them without rebuilding the graph. Weights are
 * symmetric, so W and R take a scale with no zero-point; b is INT32 at
 * scale x_scale * W_scale and needs neither; c/n are INT16 and symmetric;
 * m stays float32 because the log-space stabilizer is not quantized.
 * output shares y's scale and zero-point, which is what the kernel assumes.
 * ===========================================================================*/

#ifndef SLSTM_ORT_H_
#define SLSTM_ORT_H_

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT
#include "onnxruntime_lite_custom_op.h"

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
    Ort::Custom::Tensor<float>& m_out);

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
    Ort::Custom::Tensor<float>& m_out);

#endif /* SLSTM_ORT_H_ */
