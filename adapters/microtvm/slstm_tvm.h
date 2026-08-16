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
 * sLSTM microTVM adapter - DLTensor-based packed function.
 *
 * Args (DLTensor*):
 *   [0] X[B,T,I]  [1] W[4H,I]  [2] R[4H,H]  [3] b[4H]
 *   [4] y[B,H]    [5] c[B,H]   [6] n[B,H]    [7] m[B,H]
 *   [8] output[B,T,H]
 *
 * States y/c/n/m are updated in-place.
 *
 * The same entry point serves both precisions: it dispatches on X's
 * DLDataType. kDLFloat runs the f32 kernel on the 9 args above. kDLInt
 * runs the quantized kernel, with X/W/R/y/output int8, b int32, c/n int16
 * and m float32, and eight more args carrying the quantization:
 *
 *   [9]  x_scale (float)   [10] x_zero_point (int)
 *   [11] W_scale (float)   [12] R_scale (float)
 *   [13] y_scale (float)   [14] y_zero_point (int)
 *   [15] c_scale (float)   [16] n_scale (float)
 *
 * Weights and the INT16 states are symmetric, so they carry a scale with
 * no zero-point. m stays float32 - the log-space stabilizer is not
 * quantized.
 * ===========================================================================*/

#ifndef SLSTM_TVM_H_
#define SLSTM_TVM_H_

#include <dlpack/dlpack.h>
#include <tvm/runtime/c_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

int32_t xlstm_tvm_slstm_eval(
    TVMValue* args, int* type_codes, int num_args,
    TVMValue* out_ret_value, int* out_ret_tcode,
    void* resource_handle);

#ifdef __cplusplus
}
#endif

#endif /* SLSTM_TVM_H_ */
