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
 * Shared utilities for xLSTM kernels (sLSTM, mLSTM) — pure inline C99.
 * ===========================================================================*/

#ifndef XLSTM_UTIL_H_
#define XLSTM_UTIL_H_

#include <math.h>

static inline float sigmoid_f32(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float log_sigmoid_f32(float x) {
    /* log(sigmoid(x)) = -softplus(-x)
     * Split for numerical stability. */
    if (x >= 0.0f) {
        return -logf(1.0f + expf(-x));
    } else {
        return x - logf(1.0f + expf(x));
    }
}

#endif /* XLSTM_UTIL_H_ */
