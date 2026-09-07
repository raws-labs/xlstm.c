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
 * mLSTM microTVM adapter - unpacks DLTensors and calls core.
 * ===========================================================================*/

#include "mlstm_tvm.h"
#include "mlstm.h"
#include "mlstm_s8.h"

#include <string.h>

/* Helper: get float* from DLTensor with byte_offset */
static inline float* dl_float_ptr(DLTensor* t) {
    return (float*)((char*)t->data + t->byte_offset);
}

/* Same, for the quantized path's integer tensors. byte_offset is applied
 * before the cast in every case: it is a byte count, not an element count. */
static inline void* dl_ptr(DLTensor* t) {
    return (void*)((char*)t->data + t->byte_offset);
}

/* Scalar quantization args ride in the TVMValue union, not in a DLTensor.
 * The caller decides whether a Python float arrives as kTVMArgFloat or an
 * int as kTVMArgInt, so read the union member the type code names rather
 * than assuming one. */
static float dl_arg_float(const TVMValue* args, const int* type_codes, int i) {
    return (type_codes[i] == kTVMArgInt) ? (float)args[i].v_int64
                                         : (float)args[i].v_float64;
}

static int32_t dl_arg_int(const TVMValue* args, const int* type_codes, int i) {
    return (type_codes[i] == kTVMArgFloat) ? (int32_t)args[i].v_float64
                                           : (int32_t)args[i].v_int64;
}

static int32_t mlstm_eval_s8_packed(TVMValue* args, int* type_codes);

int32_t xlstm_tvm_mlstm_eval(
    TVMValue* args, int* type_codes, int num_args,
    TVMValue* out_ret_value, int* out_ret_tcode,
    void* resource_handle)
{
    (void)num_args;
    (void)out_ret_value;
    (void)out_ret_tcode;
    (void)resource_handle;

    /* Unpack DLTensor pointers */
    DLTensor* x      = (DLTensor*)args[0].v_handle;

    /* Quantized graphs hand this same packed function int8 tensors plus
     * the quantization args - see mlstm_tvm.h for the full arg list. */
    if (x->dtype.code == kDLInt) {
        return mlstm_eval_s8_packed(args, type_codes);
    }

    DLTensor* W      = (DLTensor*)args[1].v_handle;
    DLTensor* b      = (DLTensor*)args[2].v_handle;
    DLTensor* y      = (DLTensor*)args[3].v_handle;
    DLTensor* C      = (DLTensor*)args[4].v_handle;
    DLTensor* n      = (DLTensor*)args[5].v_handle;
    DLTensor* m      = (DLTensor*)args[6].v_handle;
    DLTensor* output = (DLTensor*)args[7].v_handle;

    int batch_size  = (int)x->shape[0];
    int time_steps  = (int)x->shape[1];
    int input_size  = (int)x->shape[2];
    int hidden_size = (int)y->shape[1];

    /* Scratch buffer on stack */
    float scratch[4 * hidden_size + 2];

    MlstmParams params = {0.0f};

    mlstm_eval_f32(
        dl_float_ptr(x),
        dl_float_ptr(W),
        dl_float_ptr(b),
        dl_float_ptr(y),
        dl_float_ptr(C),
        dl_float_ptr(n),
        dl_float_ptr(m),
        dl_float_ptr(output),
        scratch,
        batch_size, time_steps, input_size, hidden_size, hidden_size,
        &params);

    return 0;
}

static int32_t mlstm_eval_s8_packed(TVMValue* args, int* type_codes) {
    DLTensor* x      = (DLTensor*)args[0].v_handle;
    DLTensor* W      = (DLTensor*)args[1].v_handle;
    DLTensor* b      = (DLTensor*)args[2].v_handle;
    DLTensor* y      = (DLTensor*)args[3].v_handle;
    DLTensor* C      = (DLTensor*)args[4].v_handle;
    DLTensor* n      = (DLTensor*)args[5].v_handle;
    DLTensor* m      = (DLTensor*)args[6].v_handle;
    DLTensor* output = (DLTensor*)args[7].v_handle;

    int batch_size  = (int)x->shape[0];
    int time_steps  = (int)x->shape[1];
    int input_size  = (int)x->shape[2];
    int hidden_size = (int)y->shape[1];

    /* Gate accumulators are int32 on this path, not float */
    int32_t scratch[4 * hidden_size + 2];

    MlstmS8Params params;
    params.cell_clip = 0.0f;
    params.W_scale = dl_arg_float(args, type_codes, 10);
    params.x_quant.scale      = dl_arg_float(args, type_codes, 8);
    params.x_quant.zero_point = dl_arg_int(args, type_codes, 9);
    params.y_quant.scale      = dl_arg_float(args, type_codes, 11);
    params.y_quant.zero_point = dl_arg_int(args, type_codes, 12);
    /* C and n are INT16 and symmetric - no zero-point term exists */
    params.C_quant.scale      = dl_arg_float(args, type_codes, 13);
    params.C_quant.zero_point = 0;
    params.n_quant.scale      = dl_arg_float(args, type_codes, 14);
    params.n_quant.zero_point = 0;

    mlstm_eval_s8(
        (const int8_t*)dl_ptr(x),
        (const int8_t*)dl_ptr(W),
        (const int32_t*)dl_ptr(b),
        (int8_t*)dl_ptr(y),
        (int16_t*)dl_ptr(C),
        (int16_t*)dl_ptr(n),
        dl_float_ptr(m),
        (int8_t*)dl_ptr(output),
        scratch,
        batch_size, time_steps, input_size, hidden_size,
        &params);

    return 0;
}
