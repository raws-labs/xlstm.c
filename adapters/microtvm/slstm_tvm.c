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
 * sLSTM microTVM adapter — unpacks DLTensors and calls core.
 * ===========================================================================*/

#include "slstm_tvm.h"
#include "slstm.h"

#include <string.h>

/* Helper: get float* from DLTensor with byte_offset */
static inline float* dl_float_ptr(DLTensor* t) {
    return (float*)((char*)t->data + t->byte_offset);
}

int32_t xlstm_tvm_slstm_eval(
    TVMValue* args, int* type_codes, int num_args,
    TVMValue* out_ret_value, int* out_ret_tcode,
    void* resource_handle)
{
    (void)type_codes;
    (void)num_args;
    (void)out_ret_value;
    (void)out_ret_tcode;
    (void)resource_handle;

    /* Unpack DLTensor pointers */
    DLTensor* x      = (DLTensor*)args[0].v_handle;
    DLTensor* W      = (DLTensor*)args[1].v_handle;
    DLTensor* R      = (DLTensor*)args[2].v_handle;
    DLTensor* b      = (DLTensor*)args[3].v_handle;
    DLTensor* y      = (DLTensor*)args[4].v_handle;
    DLTensor* c      = (DLTensor*)args[5].v_handle;
    DLTensor* n      = (DLTensor*)args[6].v_handle;
    DLTensor* m      = (DLTensor*)args[7].v_handle;
    DLTensor* output = (DLTensor*)args[8].v_handle;

    int batch_size  = (int)x->shape[0];
    int time_steps  = (int)x->shape[1];
    int input_size  = (int)x->shape[2];
    int hidden_size = (int)y->shape[1];

    /* Scratch buffer on stack (fine for small H on MCU) */
    float scratch[4 * hidden_size];

    SlstmParams params = {0.0f};

    slstm_eval_f32(
        dl_float_ptr(x),
        dl_float_ptr(W),
        dl_float_ptr(R),
        dl_float_ptr(b),
        dl_float_ptr(y),
        dl_float_ptr(c),
        dl_float_ptr(n),
        dl_float_ptr(m),
        dl_float_ptr(output),
        scratch,
        batch_size, time_steps, input_size, hidden_size,
        &params);

    return 0;
}
