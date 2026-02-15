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
 * mLSTM microTVM adapter — DLTensor-based packed function.
 *
 * Args (DLTensor*):
 *   [0] X[B,T,I]    [1] W[4H+2,I]  [2] b[4H+2]
 *   [3] y[B,H]      [4] C[B,H*H]   [5] n[B,H]   [6] m[B,1]
 *   [7] output[B,T,H]
 *
 * States y/C/n/m are updated in-place.
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
