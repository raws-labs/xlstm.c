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
 * mLSTM ONNX Runtime custom op — lite API.
 *
 * Inputs:  X[B,T,I], W[4H+2,I], b[4H+2],
 *          y_init[B,H], C_init[B,H*H], n_init[B,H], m_init[B,1]
 * Outputs: output[B,T,H], y[B,H], C[B,H*H], n[B,H], m[B,1]
 * ===========================================================================*/

#ifndef MLSTM_ORT_H_
#define MLSTM_ORT_H_

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT
#include "onnxruntime_lite_custom_op.h"

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
    Ort::Custom::Tensor<float>& m_out);

#endif /* MLSTM_ORT_H_ */
