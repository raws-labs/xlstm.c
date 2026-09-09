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
 * mLSTM microTVM adapter - DLTensor-based packed function.
 *
 * Args (DLTensor*):
 *   [0] X[B,T,I]    [1] W[R,I]     [2] b[R]
 *   [3] y[B,DV]     [4] C[B,DQ*DV] [5] n[B,DQ]  [6] m[B,1]
 *   [7] output[B,T,DV]
 *
 * DQ is the query/key width and DV the value width, read off y's and n's
 * shapes; they are equal for a square cell. R = 2*DQ + 2*DV + 2.
 *
 * States y/C/n/m are updated in-place.
 *
 * Dispatches on X's DLDataType, exactly as slstm_tvm.h describes. The
 * kDLInt path takes seven extra args (mLSTM has no recurrent weight):
 *
 *   [8]  x_scale (float)   [9]  x_zero_point (int)
 *   [10] W_scale (float)
 *   [11] y_scale (float)   [12] y_zero_point (int)
 *   [13] C_scale (float)   [14] n_scale (float)
 * ===========================================================================*/

#ifndef MLSTM_TVM_H_
#define MLSTM_TVM_H_

#include <dlpack/dlpack.h>
#include <tvm/runtime/c_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

int32_t xlstm_tvm_mlstm_eval(
    TVMValue* args, int* type_codes, int num_args,
    TVMValue* out_ret_value, int* out_ret_tcode,
    void* resource_handle);

#ifdef __cplusplus
}
#endif

#endif /* MLSTM_TVM_H_ */
