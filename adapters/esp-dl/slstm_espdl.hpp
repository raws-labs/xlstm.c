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
 * sLSTM ESP-DL module — wraps core as an ESP-DL Module subclass.
 *
 * Input tensors (via forward() vector):
 *   [0] X[B,T,I]  [1] W[4H,I]  [2] R[4H,H]  [3] b[4H]  [4] output[B,T,H]
 *
 * States (y, c, n, m) are owned by the module and persist across calls.
 * ===========================================================================*/

#ifndef SLSTM_ESPDL_HPP_
#define SLSTM_ESPDL_HPP_

#include "dl_module_base.hpp"

extern "C" {
#include "slstm.h"
}

namespace dl {
namespace module {

class SLSTM : public Module {
public:
    int m_hidden_size;
    int m_input_size;

    SLSTM(const char* name,
           int hidden_size,
           int input_size,
           module_inplace_t inplace = MODULE_NON_INPLACE,
           quant_type_t quant_type = QUANT_TYPE_NONE);

    ~SLSTM();

    std::vector<std::vector<int>> get_output_shape(
        std::vector<std::vector<int>>& input_shapes) override;

    void forward(std::vector<dl::TensorBase*>& tensors,
                 runtime_mode_t mode = RUNTIME_MODE_AUTO) override;

private:
    float* m_y;
    float* m_c;
    float* m_n;
    float* m_m;
    float* m_scratch;
    bool m_initialized;

    void init_states();
    void free_states();
};

}  // namespace module
}  // namespace dl

#endif  // SLSTM_ESPDL_HPP_
