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
 * mLSTM ESP-DL module - wraps core as an ESP-DL Module subclass.
 *
 * Input tensors (via forward() vector):
 *   [0] X[B,T,I]  [1] W[2DQ+2DV+2,I]  [2] b[2DQ+2DV+2]  [3] output[B,T,DV]
 *
 * States (y, C, n, m) are owned by the module and persist across calls.
 *
 * QUANTIZATION
 *
 * forward() branches on quant_type - see slstm_espdl.hpp for the full
 * description. mLSTM has no recurrent weight, and its cell state is the
 * H x H matrix C, whose exponent is supplied to the constructor for the
 * same reason c's is there.
 * ===========================================================================*/

#ifndef MLSTM_ESPDL_HPP_
#define MLSTM_ESPDL_HPP_

#include "dl_module_base.hpp"

extern "C" {
#include "mlstm.h"
#include "mlstm_s8.h"
}

namespace dl {
namespace module {

class MLSTM : public Module {
public:
    // The value width sizes y, the output and C's columns; the query/key
    // width sizes the normalizer and C's rows. Equal for a square cell.
    int m_v_size;
    int m_qk_size;
    int m_input_size;

    MLSTM(const char* name,
           int v_size,
           int input_size,
           module_inplace_t inplace = MODULE_NON_INPLACE,
           quant_type_t quant_type = QUANT_TYPE_NONE,
           int C_exponent = 0,
           int n_exponent = 0,
           int qk_size = 0);   // 0 keeps the cell square

    ~MLSTM();

    std::vector<std::vector<int>> get_output_shape(
        std::vector<std::vector<int>>& input_shapes) override;

    void forward(std::vector<dl::TensorBase*>& tensors,
                 runtime_mode_t mode = RUNTIME_MODE_AUTO) override;

private:
    bool quantized() const { return quant_type == QUANT_TYPE_SYMM_8BIT; }

    void forward_f32(std::vector<dl::TensorBase*>& tensors);
    void forward_s8(std::vector<dl::TensorBase*>& tensors);

    /* f32 state */
    float* m_y;
    float* m_C;       /* [H*H] matrix cell state */
    float* m_n;
    float* m_scratch;
    /* INT8 state - y is INT8, C/n INT16, gate accumulators INT32 */
    int8_t* m_y_q;
    int16_t* m_C_q;
    int16_t* m_n_q;
    int32_t* m_scratch_q;
    /* m is a [1] scalar stabilizer, float32 on both paths */
    float* m_m;

    int m_C_exponent;
    int m_n_exponent;
    bool m_initialized;

    void init_states();
    void free_states();
};

}  // namespace module
}  // namespace dl

#endif  // MLSTM_ESPDL_HPP_
