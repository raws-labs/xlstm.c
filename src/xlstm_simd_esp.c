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
 * ONE of the four contract functions is accelerated: xlstm_matvec_f32 takes
 * four rows at a time and reads each row four columns at a time with
 * EE.LDF.128.IP, the S3's 128-bit load. Which path a call takes is decided
 * by its rows and cols alone - not by where the caller's buffers happened to
 * land, which is all the alignment guard this replaces ever measured.
 *
 * xlstm_matvec_s8, xlstm_rank1_update_f32 and xlstm_vecmat_f32 - and the
 * rows the blocked body cannot cover - defer to the scalar bodies in
 * xlstm_simd_scalar.h, the same text xlstm_simd_ref.c compiles rather than
 * a copy of it. Copies here would be a second reference free to drift from
 * the one every backend is defined against.
 *
 * `make test-esp` runs the golden vectors against this backend on an
 * emulated ESP32-S3 and reports how many f32 matvecs reached the blocked
 * path; read that target's comment before quoting a green run.
 * ===========================================================================*/

#include "xlstm_simd.h"

#include "xlstm_simd_scalar.h"

#include <stddef.h>
#include <stdint.h>

/* The PIE - the S3's vector extension - has no feature macro of its own the
 * way ACLE gives ARM one, and the assembler simply rejects the mnemonic
 * below on a core without it. __XTENSA__ is what can be tested here, and it
 * turns "wrong backend selected" into one line instead of an assembler error
 * from the middle of a loop. */
#ifndef __XTENSA__
#error "XLSTM_SIMD=esp is the ESP32-S3 backend: build it with the ESP-IDF" \
       " xtensa-esp32s3 toolchain, or use XLSTM_SIMD=ref."
#endif

/* WHY THIS IS NOT A VECTOR DOT PRODUCT
 *
 * The obvious accelerated matvec is ESP-DSP's dsps_dotprod_f32, and this
 * backend used to call it. That function sums into four partial accumulators
 * and adds them at the end, which reassociates the addition - and the f32
 * goldens have no room for it. Merely moving out[i] from the seed of the
 * accumulator to a final add, a far smaller change than reassociating, put
 * mLSTM SweepM17 y[0] 4.20e-05 from its golden value against a ~3.98e-05
 * bound. The call site survived only because its alignment guard was almost
 * never satisfied, so no case with a tight bound ever reached it; widening
 * that guard while keeping the arithmetic would have failed the gate.
 *
 * Nothing here has to reassociate anything, because the S3 has no f32 vector
 * ALU. dsps_dotprod_f32_aes3's speed comes from EE.LDF.128.IP, a 128-bit
 * load that fills four FP registers, feeding four ordinary scalar madd.s.
 * The width is in the LOAD. So the body below keeps one accumulator per row,
 * seeded from out[i] and summed in ascending j exactly as
 * xlstm_scalar_matvec_f32 does, and takes the width where it actually is.
 * Every f32 result this backend produces is bit-identical to ref, which is
 * also why the golden data does not move.
 *
 * ALIGNMENT, WHICH THE KERNEL OWNS AND THE CALLER DOES NOT
 *
 * EE.LDF.128.IP needs its address 16-byte aligned. M and v belong to the
 * caller, so a guard demanding they arrive aligned measures the link and not
 * this kernel. What the kernel can do is reach alignment itself: at most
 * three columns of scalar prefix bring a row to a 16-byte boundary, and
 * every group after it is aligned by construction.
 *
 * One prefix has to serve all four rows of a block, so those rows must agree
 * modulo 16 bytes. Rows r and r + step do exactly when step*cols is a
 * multiple of 4 floats, i.e. step = 4 / gcd(cols, 4): 1 for a cols divisible
 * by four, 2 for an even one, 4 for an odd one. A block therefore takes four
 * rows step apart, and step blocks cover 4*step consecutive rows. Spelling
 * it this way is what keeps odd sizes on the vector body - H = 17 keeps 64
 * of its 68 (sLSTM) or 70 (mLSTM) rows there - rather than abandoning it
 * where much of the work is.
 *
 * Two shapes still leave the blocked body, and both are decided by rows and
 * cols alone:
 *
 *   - cols < 7. A prefix of up to 3 plus one whole group of 4 needs 7
 *     columns to be certain of fitting, and certain is the point: at cols 4
 *     to 6 whether a group fits would depend on the alignment the caller
 *     happened to supply, which is the property being removed.
 *   - rows < 4*step, fewer than one block. Both callers pass 4H or 4H + 2
 *     rows, so this is H < 4 against an odd cols.
 *
 * v is read scalar, one float per column per block rather than one load per
 * column per row, and deliberately not through EE.LDF.128.IP as well: v's
 * own alignment is independent of M's, so vectorizing it would buy 3
 * instructions in every 24 at the price of making the instruction sequence
 * depend on the caller's buffers again.
 *
 * Four rows per block, because that is where the FP register file runs out:
 * a block holds four accumulators, the four floats of the group in flight
 * and the four v values it multiplies them by, which is 12 of the S3's 16 FP
 * registers. Eight rows would need 16 with nothing left to load into.
 */

/* One 128-bit load: four consecutive floats into four FP registers, cursor
 * advanced 16 bytes. Operand order is fu, fs, fr, fq with fq the LOWEST
 * address, so the template lists the caller's ascending x0..x3 backwards -
 * measured on the part's emulator, not inferred from the dot product's use
 * of it, which pairs its lanes symmetrically and so cannot tell the order.
 *
 * The "m" input is not referenced by the template. It is there so that the
 * compiler knows those 16 bytes are read and cannot move the load across a
 * store to them; the alternative, a "memory" clobber, would also spill the
 * accumulators this loop exists to keep in registers. */
#define XLSTM_ESP_LDF128(x0, x1, x2, x3, p)                       \
    __asm__("ee.ldf.128.ip %3, %2, %1, %0, %4, 16"                \
            : "=f"(x0), "=f"(x1), "=f"(x2), "=f"(x3), "+r"(p)     \
            : "m"(*(const float(*)[4])(p)))

/* Whether the widened path was taken is now a property of rows and cols, but
 * a build can still link this backend, reproduce every golden vector, and
 * execute the 128-bit load rarely or never - the shapes decide, and nothing
 * else here can observe which shapes arrived. That makes it exactly the kind
 * of silent loss of coverage a gate has to be able to fail on. These two
 * counters are how it does: test/esp/main/esp_gate.cc calls this function
 * across shapes and alignments and asserts the split is the one the rules
 * above predict.
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

/* Four rows, rstep floats apart so that they share an alignment, and their
 * four outputs ostep apart. Each accumulator starts at its out[] element and
 * runs ascending j through prefix, groups and tail without regrouping, which
 * is what makes the result bit-identical to the scalar body. */
static void xlstm_esp_block4(const float* row, size_t rstep, const float* v,
                             float* out, int ostep, int cols)
{
    const float* p0 = row;
    const float* p1 = row + rstep;
    const float* p2 = p1 + rstep;
    const float* p3 = p2 + rstep;
    /* Floats to skip before p0 - and therefore p1, p2 and p3 - is 16-byte
     * aligned. A float* is already 4-byte aligned, so this is 0 to 3. */
    const int pre = (int)(((16u - ((uintptr_t)p0 & 15u)) & 15u) >> 2);
    const float* q0 = p0 + pre;
    const float* q1 = p1 + pre;
    const float* q2 = p2 + pre;
    const float* q3 = p3 + pre;
    float a0 = out[0];
    float a1 = out[ostep];
    float a2 = out[2 * ostep];
    float a3 = out[3 * ostep];
    int j;

    for (j = 0; j < pre; ++j) {
        float x = v[j];
        a0 += p0[j] * x;
        a1 += p1[j] * x;
        a2 += p2[j] * x;
        a3 += p3[j] * x;
    }

    /* q0..q3 are aligned here and stay aligned: every pass consumes exactly
     * one 16-byte group from each. */
    for (; j + 3 < cols; j += 4) {
        float x0 = v[j];
        float x1 = v[j + 1];
        float x2 = v[j + 2];
        float x3 = v[j + 3];
        float b0, b1, b2, b3;

        XLSTM_ESP_LDF128(b0, b1, b2, b3, q0);
        a0 += b0 * x0;
        a0 += b1 * x1;
        a0 += b2 * x2;
        a0 += b3 * x3;

        XLSTM_ESP_LDF128(b0, b1, b2, b3, q1);
        a1 += b0 * x0;
        a1 += b1 * x1;
        a1 += b2 * x2;
        a1 += b3 * x3;

        XLSTM_ESP_LDF128(b0, b1, b2, b3, q2);
        a2 += b0 * x0;
        a2 += b1 * x1;
        a2 += b2 * x2;
        a2 += b3 * x3;

        XLSTM_ESP_LDF128(b0, b1, b2, b3, q3);
        a3 += b0 * x0;
        a3 += b1 * x1;
        a3 += b2 * x2;
        a3 += b3 * x3;
    }

    for (; j < cols; ++j) {
        float x = v[j];
        a0 += p0[j] * x;
        a1 += p1[j] * x;
        a2 += p2[j] * x;
        a3 += p3[j] * x;
    }

    out[0] = a0;
    out[ostep] = a1;
    out[2 * ostep] = a2;
    out[3 * ostep] = a3;
}

void xlstm_matvec_f32(const float* M, const float* v,
                      float* out, int rows, int cols)
{
    const int step = (cols % 4 == 0) ? 1 : ((cols % 2 == 0) ? 2 : 4);
    const int tile = 4 * step;
    const int fast = cols >= 7 && rows >= tile;
    int done = 0;

    XLSTM_ESP_COUNT(fast);

    if (fast) {
        int base;
        for (base = 0; base + tile <= rows; base += tile) {
            int q;
            /* The step blocks starting at base + 0 .. base + step - 1 cover
             * rows base .. base + tile - 1 between them, each block taking
             * every step'th row of that range. */
            for (q = 0; q < step; ++q) {
                int r = base + q;
                xlstm_esp_block4(M + (size_t)r * (size_t)cols,
                                 (size_t)step * (size_t)cols,
                                 v, out + r, step, cols);
            }
        }
        done = base;
    }

    /* Whatever is left is fewer rows than a block. It runs the shared scalar
     * body rather than a narrower vector one: at 4H or 4H + 2 rows this is
     * at most 6 rows of a call, and a second body here would be a second
     * definition of the accumulation order. */
    if (done < rows) {
        xlstm_scalar_matvec_f32(M + (size_t)done * (size_t)cols, v,
                                out + done, rows - done, cols);
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
