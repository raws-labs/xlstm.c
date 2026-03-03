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
 * xlstm.h — single-include umbrella header for the xlstm.c library.
 *
 * Pulls in all public kernel APIs (f32 + INT8) and exposes library version
 * and active SIMD backend for introspection.
 *
 * Reference: https://arxiv.org/abs/2405.04517
 * ===========================================================================*/

#ifndef XLSTM_H_
#define XLSTM_H_

#define XLSTM_VERSION_MAJOR 0
#define XLSTM_VERSION_MINOR 2
#define XLSTM_VERSION_PATCH 0
#define XLSTM_VERSION "0.2.0"

/* Kernel APIs */
#include "slstm.h"
#include "mlstm.h"
#include "slstm_q8.h"
#include "mlstm_q8.h"

/* SIMD backend introspection */
#include "xlstm_simd.h"

#endif /* XLSTM_H_ */
