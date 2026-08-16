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
 * Shared quantization types and helpers for xLSTM INT8 kernels.
 *
 * Quantization convention:
 *   real_value = scale * (quantized_value - zero_point)
 *
 * Symmetric (weights): zero_point = 0, scale = max_abs / 127
 * Asymmetric (activations): scale = (max - min) / 255, zero_point computed
 * ===========================================================================*/

#ifndef XLSTM_QUANT_H_
#define XLSTM_QUANT_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Per-tensor quantization parameters */
typedef struct {
    float scale;        /* real_value = scale * (quantized_value - zero_point) */
    int32_t zero_point; /* 0 for symmetric (weights), variable for asymmetric */
} XlstmQuantParam;

/* Compute symmetric quant params from float tensor (weights: zp=0) */
void xlstm_quant_symmetric(const float* data, int len, XlstmQuantParam* out);

/* Compute asymmetric quant params from float tensor (activations).
 * Range is expanded to include zero for proper zero-padding support. */
void xlstm_quant_asymmetric(const float* data, int len, XlstmQuantParam* out);

/* Compute symmetric quant params at INT16 granularity (scale =
 * max_abs * headroom / 32767, zero_point 0).
 *
 * Use this - not xlstm_quant_asymmetric - for the INT16 state tensors
 * (sLSTM c/n, mLSTM C/n). Those are read and written as strictly
 * symmetric: no zero-point term appears anywhere in slstm_s8.c or
 * mlstm_s8.c. Calibrating them with range/255 and a non-zero zero_point
 * is an INT8-shaped calibration applied to a 16-bit symmetric tensor and
 * throws away roughly 7 bits of the available range.
 *
 * headroom > 1 widens the scale so a state trajectory that peaks above
 * the calibration sample does not clip. Calibrating from a single
 * end-of-sequence snapshot can undershoot the true mid-sequence peak by
 * several times; 4.0 is the value the correctness gate is calibrated
 * against. Pass 1.0 for no headroom. */
void xlstm_quant_symmetric_s16(const float* data, int len, float headroom,
                                XlstmQuantParam* out);

/* Quantize/dequantize helpers */
void xlstm_quantize_f32_to_s8(const float* src, int8_t* dst, int len,
                               const XlstmQuantParam* qp);
void xlstm_dequantize_s8_to_f32(const int8_t* src, float* dst, int len,
                                 const XlstmQuantParam* qp);
void xlstm_quantize_f32_to_s16(const float* src, int16_t* dst, int len,
                                const XlstmQuantParam* qp);
void xlstm_dequantize_s16_to_f32(const int16_t* src, float* dst, int len,
                                  const XlstmQuantParam* qp);
void xlstm_quantize_f32_to_s32(const float* src, int32_t* dst, int len,
                                const XlstmQuantParam* qp);

#ifdef __cplusplus
}
#endif

#endif /* XLSTM_QUANT_H_ */
