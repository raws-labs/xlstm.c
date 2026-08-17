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
 * ESP32-S3 backend. Only compiles under the ESP-IDF toolchain.
 *
 * ONE of the four contract functions is accelerated: xlstm_matvec_f32 calls
 * ESP-DSP's dot product, and only when the caller's buffers satisfy the
 * alignment guard below. No Xtensa PIE instruction is issued anywhere in
 * this file.
 *
 * xlstm_matvec_s8, xlstm_rank1_update_f32 and xlstm_vecmat_f32 - and the
 * unaccelerated half of xlstm_matvec_f32 - defer to the scalar bodies in
 * xlstm_simd_scalar.h, the same text xlstm_simd_ref.c compiles rather than
 * a copy of it. Copies here would be a second reference free to drift from
 * the one every backend is defined against.
 *
 * `make test-esp` runs the golden vectors against this backend on an
 * emulated ESP32-S3 and reports how many f32 matvecs reached the
 * accelerated path; read that target's comment before quoting a green run.
 * ===========================================================================*/

#include "xlstm_simd.h"
#include "dsps_dotprod.h"

#include "xlstm_simd_scalar.h"

#include <stdint.h>

/* dsps_dotprod_f32 is a macro that resolves to dsps_dotprod_f32_aes3 on the
 * S3 (_ae32 on the ESP32, _ansi elsewhere). The S3 body loads 128 bits at a
 * time (EE.LDF.128.IP src, 16) and consumes four floats per iteration, so it
 * wants both operands 16-byte aligned and the length a multiple of 4.
 *
 * Calling it without those is NOT undefined and does NOT silently return
 * wrong numbers - a claim this comment used to make, and it is wrong.
 * _aes3 opens by testing exactly those three conditions (`len & 3`, then
 * `(src1 | src2) & 15`) and branches to the _ae32 body when any fails, so
 * the arithmetic is right either way. That prologue is byte-identical in
 * esp-dsp v1.2.0, v1.5.2 and v1.8.2, so it is a property of the library and
 * not of one release.
 *
 * The guard below therefore buys speed rather than correctness - and one
 * other thing. The _ae32 body it steers around reads one float past the end
 * of src2: its inner macro loads x2[i+1] at the bottom of every iteration,
 * including the last. The overread is real, and unreachable from here by
 * construction, because `fast` is precisely the condition under which _aes3
 * never enters that body.
 *
 * xlstm_matvec_f32 walks rows as M + i*cols, so row alignment depends on
 * both the base pointer and cols; the guard is hoisted accordingly.
 *
 * Note also that despite the header's doc comment claiming
 * "*dest += src1[i]*src2[i]", every implementation ASSIGNS (*dest = acc;
 * `ssi f0, a4, 0` in the assembly). The uninitialized `dot` below is
 * therefore correct - do not "fix" it by zeroing, and do not assume the
 * accumulate semantics the comment advertises.
 */
#define XLSTM_ESP_DSP_ALIGN 16

static inline int xlstm_esp_aligned(const void* p)
{
    return (((uintptr_t)p) & (XLSTM_ESP_DSP_ALIGN - 1)) == 0;
}

/* Whether the accelerated path was taken is a property of the buffers the
 * caller happened to pass, not of this file - so a build can link this
 * backend, reproduce every golden vector, and never once execute the
 * accelerated instruction. Nothing else here can observe that, which makes
 * it exactly the kind of silent loss of coverage a gate has to be able to
 * fail on. These two counters are how it does: test/esp/main/esp_gate.cc
 * calls this function with buffers it aligns itself and asserts the split
 * moved the way it must.
 *
 * Off unless XLSTM_ESP_FASTPATH_COUNTERS is defined (test/esp sets it, the
 * ordinary build does not), so a shipping kernel carries neither the
 * counters nor the increment. */
#ifdef XLSTM_ESP_FASTPATH_COUNTERS
unsigned long xlstm_esp_matvec_f32_fast = 0;
unsigned long xlstm_esp_matvec_f32_scalar = 0;
#define XLSTM_ESP_COUNT(fast) \
    ((void)((fast) ? ++xlstm_esp_matvec_f32_fast : ++xlstm_esp_matvec_f32_scalar))
#else
#define XLSTM_ESP_COUNT(fast) ((void)0)
#endif

void xlstm_matvec_f32(const float* M, const float* v,
                      float* out, int rows, int cols)
{
    /* Hoisted: if cols is a multiple of 4 then every row shares the base
     * pointer's alignment, so this is decided once rather than per row. */
    const int fast = (cols % 4) == 0 && xlstm_esp_aligned(M) && xlstm_esp_aligned(v);
    int i;

    XLSTM_ESP_COUNT(fast);

    /* The guard is loop-invariant, so the unaccelerated case is the whole
     * call and hands off to the shared body rather than to a per-row copy
     * of it.
     *
     * Deliberately NOT dsps_dotprod_f32_ansi. That computes the dot into a
     * fresh accumulator and leaves the caller to do out[i] += dot, whereas
     * the scalar body seeds one running accumulator from out[i]. The
     * mathematics is identical but the float grouping is not, and on a
     * cancellation-sensitive case it moves the result: mLSTM SweepM17 y[0]
     * came out 4.20e-05 from the golden value against a ~3.98e-05 bound,
     * failing the gate on the esp backend's first end-to-end run.
     *
     * The fallback is plain C either way, so there is nothing to gain from
     * ESP-DSP here and a real divergence to lose. */
    if (!fast) {
        xlstm_scalar_matvec_f32(M, v, out, rows, cols);
        return;
    }

    for (i = 0; i < rows; ++i) {
        /* The chip-generic macro, not _ae32. Earlier code hardcoded the
         * ESP32 variant, which compiles on the S3 (its guard is generic
         * Xtensa capability flags, not a chip check) and leaves the
         * S3-optimized body unused. */
        float dot;
        dsps_dotprod_f32(M + i * cols, v, &dot, cols);
        out[i] += dot;
    }
}

void xlstm_matvec_s8(const int8_t* M, const int8_t* v,
                     int32_t* out, int rows, int cols, int32_t v_zp)
{
    /* TODO: PIE EE.VMULAS.S8.ACCX for 16-way int8 MAC.
     * Scalar fallback until PIE intrinsics are available. */
    xlstm_scalar_matvec_s8(M, v, out, rows, cols, v_zp);
}

void xlstm_rank1_update_f32(float* C, float f_gate, float i_gate,
                            const float* k, const float* v, int H)
{
    xlstm_scalar_rank1_update_f32(C, f_gate, i_gate, k, v, H);
}

void xlstm_vecmat_f32(const float* q, const float* M,
                      float* out, int rows, int cols)
{
    xlstm_scalar_vecmat_f32(q, M, out, rows, cols);
}

const char* xlstm_simd_backend(void)
{
    return "esp";
}
