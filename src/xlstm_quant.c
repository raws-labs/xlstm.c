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
 * Quantization/dequantization helpers — pure C99
 * ===========================================================================*/

#include "xlstm_quant.h"

#include <math.h>

void xlstm_quant_symmetric(const float* data, int len, XlstmQuantParam* out) {
    int i;
    float max_abs = 0.0f;

    for (i = 0; i < len; ++i) {
        float a = fabsf(data[i]);
        if (a > max_abs) max_abs = a;
    }

    out->scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
    out->zero_point = 0;
}

void xlstm_quant_asymmetric(const float* data, int len, XlstmQuantParam* out) {
    int i;
    float min_val, max_val, range;
    int32_t zp;

    min_val = data[0];
    max_val = data[0];
    for (i = 1; i < len; ++i) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }

    /* Ensure range includes zero (standard convention for activations) */
    if (min_val > 0.0f) min_val = 0.0f;
    if (max_val < 0.0f) max_val = 0.0f;

    range = max_val - min_val;
    if (range < 1e-10f) {
        out->scale = 1.0f / 255.0f;
        out->zero_point = 0;
        return;
    }

    out->scale = range / 255.0f;
    zp = (int32_t)roundf(-128.0f - min_val / out->scale);
    if (zp < -128) zp = -128;
    if (zp > 127) zp = 127;
    out->zero_point = zp;
}

void xlstm_quantize_f32_to_s8(const float* src, int8_t* dst, int len,
                               const XlstmQuantParam* qp) {
    int i;
    for (i = 0; i < len; ++i) {
        float v = roundf(src[i] / qp->scale) + (float)qp->zero_point;
        if (v < -128.0f) v = -128.0f;
        if (v > 127.0f) v = 127.0f;
        dst[i] = (int8_t)(int32_t)v;
    }
}

void xlstm_dequantize_s8_to_f32(const int8_t* src, float* dst, int len,
                                 const XlstmQuantParam* qp) {
    int i;
    for (i = 0; i < len; ++i) {
        dst[i] = qp->scale * ((float)src[i] - (float)qp->zero_point);
    }
}

void xlstm_quantize_f32_to_s16(const float* src, int16_t* dst, int len,
                                const XlstmQuantParam* qp) {
    int i;
    for (i = 0; i < len; ++i) {
        float v = roundf(src[i] / qp->scale) + (float)qp->zero_point;
        if (v < -32768.0f) v = -32768.0f;
        if (v > 32767.0f) v = 32767.0f;
        dst[i] = (int16_t)(int32_t)v;
    }
}

void xlstm_dequantize_s16_to_f32(const int16_t* src, float* dst, int len,
                                  const XlstmQuantParam* qp) {
    int i;
    for (i = 0; i < len; ++i) {
        dst[i] = qp->scale * ((float)src[i] - (float)qp->zero_point);
    }
}

void xlstm_quantize_f32_to_s32(const float* src, int32_t* dst, int len,
                                const XlstmQuantParam* qp) {
    int i;
    for (i = 0; i < len; ++i) {
        float v = roundf(src[i] / qp->scale) + (float)qp->zero_point;
        if (v < -2147483648.0f) v = -2147483648.0f;
        if (v > 2147483647.0f) v = 2147483647.0f;
        dst[i] = (int32_t)v;
    }
}
