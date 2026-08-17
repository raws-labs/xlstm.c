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
 * TWO of the four contract functions are accelerated:
 *
 *   xlstm_matvec_f32  four rows at a time, four columns per EE.LDF.128.IP -
 *                     the S3 has no f32 vector ALU, so the width is in the
 *                     load and the arithmetic stays scalar.
 *   xlstm_matvec_s8   16 columns per EE.VMULAS.S8.ACCX, the PIE's 16-lane
 *                     int8 multiply-accumulate into a 40-bit accumulator.
 *
 * Which path a call takes is decided by its shape and, for INT8, its zero
 * point - never by where the caller's buffers happened to land, which is all
 * the alignment guard the f32 body once carried ever measured.
 *
 * xlstm_rank1_update_f32 and xlstm_vecmat_f32 - and the f32 rows the blocked
 * body cannot cover - defer to the scalar bodies in xlstm_simd_scalar.h, the
 * same text xlstm_simd_ref.c compiles rather than a copy of it. Copies here
 * would be a second reference free to drift from the one every backend is
 * defined against.
 *
 * `make test-esp` runs the golden vectors against this backend on an
 * emulated ESP32-S3 and reports how much of each matvec reached the vector
 * body; read that target's comment before quoting a green run.
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

/* Whether the widened path was taken is now a property of the call's shape,
 * but a build can still link this backend, reproduce every golden vector,
 * and execute the wide instructions rarely or never - the shapes decide, and
 * nothing else here can observe which shapes arrived. That makes it exactly
 * the kind of silent loss of coverage a gate has to be able to fail on.
 * These counters are how it does: test/esp/main/esp_gate.cc calls both
 * matvecs across shapes and alignments and asserts the split is the one the
 * rules above predict.
 *
 * Off unless XLSTM_ESP_FASTPATH_COUNTERS is defined (test/esp sets it, the
 * ordinary build does not), so a shipping kernel carries neither the
 * counters nor the increment. */
#ifdef XLSTM_ESP_FASTPATH_COUNTERS
unsigned long xlstm_esp_matvec_f32_fast = 0;
unsigned long xlstm_esp_matvec_f32_scalar = 0;
unsigned long xlstm_esp_matvec_s8_fast = 0;
unsigned long xlstm_esp_matvec_s8_scalar = 0;
#define XLSTM_ESP_COUNT(fast) \
    ((void)((fast) ? ++xlstm_esp_matvec_f32_fast : ++xlstm_esp_matvec_f32_scalar))
#define XLSTM_ESP_COUNT_S8(fast) \
    ((void)((fast) ? ++xlstm_esp_matvec_s8_fast : ++xlstm_esp_matvec_s8_scalar))
#else
#define XLSTM_ESP_COUNT(fast) ((void)0)
#define XLSTM_ESP_COUNT_S8(fast) ((void)0)
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

/* =========================== INT8 =========================================
 *
 * WHY THE ZERO POINT BECOMES A CONSTANT VECTOR
 *
 * EE.VMULAS.S8.ACCX multiplies two 16-lane int8 vectors and accumulates all
 * 16 products into ACCX, a 40-bit accumulator. It has no zero-point operand,
 * and v[j] - v_zp does not fit an int8 lane - it spans [-255, 255] - so the
 * subtraction cannot be done to the vector before the multiply. What the
 * kernel uses instead is
 *
 *   sum_j M[j] * (v[j] - v_zp)  =  sum_j M[j]*v[j]  +  sum_j M[j]*c,  c = -v_zp
 *
 * run as two MACs over the same M window: one against the v window, one
 * against a vector holding c in every lane. Both land in the same ACCX,
 * which is what makes this affordable - the part has exactly one such
 * accumulator, so a formulation needing a separate row sum would need a
 * second pass over M.
 *
 * This is a regrouping of a sum of integers, not an approximation. Every
 * product is exact, ACCX is exact (see below), and integer addition is
 * associative, so the result equals the scalar body's term for term - which
 * is the whole requirement, integers admitting no tolerance.
 *
 * c has to be an int8 to sit in a lane, and c = 128 is reachable: a tensor
 * with no negative values calibrates to v_zp = -128. So c is carried in two
 * lanes, c1 + c2, and the second MAC becomes two whenever c2 is non-zero.
 * Both instances are the same source under a compile-time flag; the common
 * case pays nothing for the rare one.
 *
 * OVERFLOW, IN ACCX AND OUT OF IT
 *
 * ACCX is 40 bits, and no lane exceeds 128 in magnitude, so each of the at
 * most three MAC streams contributes under 128 * 128 = 16384 per column and
 * |ACCX| stays below 49152 * cols throughout the accumulation - inside 40
 * bits (2^39 = 5.5e11) until about 11 million columns.
 *
 * RUR.ACCX_0 returns the low 32 of those 40, and that is the answer rather
 * than a truncation of it. ACCX holds the sum exactly, so its low 32 bits
 * are that sum's two's-complement int32 value. Where the sum fits int32 -
 * |M[j] * (v[j] - v_zp)| <= 128 * 255 = 32640, so below about 65000 columns,
 * which is the same headroom argument that bounds the scalar body's own
 * accumulator - that IS the value. Past it that accumulator has already
 * wrapped, and the low 32 bits reproduce the identical wrap. The two agree
 * bit for bit either way.
 *
 * ALIGNMENT, WHICH THE KERNEL OWNS AND THE CALLER DOES NOT
 *
 * A 128-bit load needs a 16-byte aligned address and both operands belong to
 * the caller. The f32 body reaches alignment with a scalar prefix and blocks
 * rows so one prefix serves several; that works there because rows agree
 * modulo a 4-float group every 4 / gcd(cols, 4) rows. A 16-byte group makes
 * both halves of that fail: rows agree only every 16 / gcd(cols, 16) rows -
 * 16 of them for an odd cols, so 256 consecutive rows, more than the 4H a
 * caller ever passes at H = 17 - and a prefix would cost up to 15 columns of
 * scalar per row, which at cols = 17 leaves nothing behind it.
 *
 * So this kernel does not seek alignment at all. Each 16-column group is
 * assembled from the two aligned blocks that contain it:
 * EE.LD.128.USAR.IP loads the aligned block containing an address and sets
 * SAR_BYTE to that address's offset within it, and EE.SRC.Q takes the window
 * across the pair. Alignment therefore never enters the dispatch, and every
 * row of every shape runs on the vector body at every alignment - which is
 * the property the gate asserts.
 *
 * THE LAST GROUP IS PARTIAL, AND STILL VECTOR
 *
 * cols is not a multiple of 16 in general. Running the leftover t columns
 * scalar would abandon the vector body exactly where the smaller hidden
 * sizes live - at H = 8 it would be the entire row - so the last group runs
 * on the vector unit too, against two 16-byte constants built once per call:
 * v's last t values zero-padded, and c masked to its first t lanes. Both
 * extra lanes multiply by zero, so whatever M's window holds above column
 * cols - 1 contributes nothing at all. That is what puts every gated hidden
 * size on the vector body, H = 8 and H = 17 included.
 *
 * A group is 16 lanes whether or not the row has 16 columns left, so the
 * smallest sizes pay for lanes they do not use. Counted off the emitted
 * image: a whole group is 10 instructions where the scalar body spends 14
 * per column, and a row carries about 38 fixed instructions on top of that.
 * So this is a win from four columns up - 3x at H = 8, 6x at H = 16, 13x at
 * H = 64 - about a wash at cols = 2, and roughly twice the instructions at
 * cols = 1. Those two stay on the vector body deliberately: the alternative
 * is a width threshold chosen from an instruction count, and instructions
 * are not cycles. Nothing available here can measure the difference on this
 * part, and a threshold guessed from the wrong quantity is worse than the
 * uniform rule it would replace - which is also the rule the gate asserts.
 *
 * NOTHING IS READ THAT CANNOT BE READ
 *
 * Every 128-bit load reads the 16-byte aligned block containing its address.
 * All those addresses lie inside their own buffer except the last group's
 * upper block, which is loaded only when the row's final byte falls in it -
 * the same statement. So every block loaded holds at least one byte of the
 * caller's data, and an aligned 16-byte block is entirely readable whenever
 * one of its bytes is: this part's memory regions are 4 KiB aligned or
 * coarser, so such a block cannot straddle the end of one.
 *
 * This is the difference from esp-dsp's dsps_dp_s8_aes3, which is otherwise
 * the obvious thing to call: it reads 16 bytes past the end of BOTH operands
 * with no alignment guarantee, which is a fault rather than a wrong number,
 * and it engages SIMD only when the length is a multiple of 16 - abandoning
 * four of the six hidden sizes this library gates, H = 17 among them.
 */

/* -v_zp is carried in two int8 lanes, so |v_zp| <= 254 folds exactly (two
 * lanes reach 254 before one of them leaves int8). Every caller passes a
 * zero point in [-128, 127] - xlstm_quant_asymmetric clamps to it - so this
 * bound exists to keep the kernel exact for any int32 the signature permits,
 * not for a range anything actually uses. */
#define XLSTM_ESP_ZP_MAX 254

#define XLSTM_ESP_A16 __attribute__((aligned(16)))

/* One 16-column group of one row: the two aligned blocks holding M's window,
 * the two holding v's, both windows, and the two MACs. Every cursor advances
 * one block.
 *
 * EE.SRC.Q's destination is a third register, never one of its sources. The
 * in-place spelling (qu == qs0) that esp-dsp's memcpy uses does not produce
 * the window under the QEMU this backend is gated on - measured there, not
 * inferred - and a kernel whose result depends on which machine executes it
 * is not a kernel. q4 and q5 are those third registers; q6 carries c1 for
 * the whole call and q7 carries c2.
 *
 * A "memory" clobber rather than the f32 macro's "m" inputs: the accumulator
 * here is ACCX, which is not a compiler register, so there is nothing for a
 * clobber to spill - the reason the f32 body avoids one does not apply. */
#define XLSTM_ESP_S8_GROUP(ml, mh, vl, vh)                        \
    __asm__ volatile("ee.ld.128.usar.ip q0, %0, 16\n\t"           \
                     "ee.ld.128.usar.ip q1, %1, 16\n\t"           \
                     "ee.src.q q4, q0, q1\n\t"                    \
                     "ee.ld.128.usar.ip q2, %2, 16\n\t"           \
                     "ee.ld.128.usar.ip q3, %3, 16\n\t"           \
                     "ee.src.q q5, q2, q3\n\t"                    \
                     "ee.vmulas.s8.accx q4, q5\n\t"               \
                     "ee.vmulas.s8.accx q4, q6"                   \
                     : "+r"(ml), "+r"(mh), "+r"(vl), "+r"(vh)     \
                     : : "memory")

/* The partial last group. v and the zero point come from the two padded
 * constants instead of from windows, so no cursor advances. */
#define XLSTM_ESP_S8_TAIL(ml, mh, vt, kt)                         \
    __asm__ volatile("ee.ld.128.usar.ip q0, %0, 0\n\t"            \
                     "ee.ld.128.usar.ip q1, %1, 0\n\t"            \
                     "ee.src.q q4, q0, q1\n\t"                    \
                     "ee.vld.128.ip q2, %2, 0\n\t"                \
                     "ee.vld.128.ip q3, %3, 0\n\t"                \
                     "ee.vmulas.s8.accx q4, q2\n\t"               \
                     "ee.vmulas.s8.accx q4, q3"                   \
                     : "+r"(ml), "+r"(mh), "+r"(vt), "+r"(kt)     \
                     : : "memory")

/* The second half of a split zero point, against the M window the block
 * above left in q4. Separate statements rather than a second copy of each
 * block: MACs into ACCX commute, so where this one sits does not matter. */
#define XLSTM_ESP_S8_GROUP_C2() __asm__ volatile("ee.vmulas.s8.accx q4, q7")
#define XLSTM_ESP_S8_TAIL_C2(k2)                                  \
    __asm__ volatile("ee.vld.128.ip q5, %0, 0\n\t"                \
                     "ee.vmulas.s8.accx q4, q5"                   \
                     : "+r"(k2) : : "memory")

#define XLSTM_ESP_LOADQ(qreg, p)                                  \
    __asm__ volatile("ee.vld.128.ip " qreg ", %0, 0"              \
                     : "+r"(p) : : "memory")

#define XLSTM_ESP_ACCX_ZERO(z)                                    \
    __asm__ volatile("wur.accx_0 %0\n\t wur.accx_1 %0" : : "r"(z))

/* The two nops are not decoration: esp-dsp's own generated kernels never put
 * RUR.ACCX_0 within two instructions of a MAC. QEMU does not model whatever
 * that spacing is for, so it cannot be tested here, and one instruction per
 * row is not worth the question. */
#define XLSTM_ESP_ACCX_READ(x)                                    \
    __asm__ volatile("nop\n\t nop\n\t rur.accx_0 %0" : "=r"(x))

/* c1 and c2 are compile-time-fixed lanes of -v_zp; split says whether the
 * second one is live. always_inline so that stays a constant however the
 * inliner's size heuristic feels about two loops - a surviving runtime
 * `split` would put a branch on every group of the common case. */
static inline __attribute__((always_inline)) void
xlstm_esp_matvec_s8_body(const int8_t* M, const int8_t* v, int32_t* out,
                         int rows, int cols, int c1, int c2, int split)
{
    /* nfull whole groups and then one partial group of t in [1, 16], rather
     * than floor(cols/16) whole groups and a scalar remainder. Spelling it
     * this way is what keeps the last group - the only group at all when
     * cols <= 16 - on the vector body. */
    const int nfull = (cols - 1) >> 4;
    const int t = cols - (nfull << 4);
    const size_t stride = (size_t)cols;
    /* v's alignment is fixed for the whole call; M's is not, since each row
     * starts cols bytes after the last. Both are read the same way. */
    const int voff = ((uintptr_t)v & 15u) ? 16 : 0;
    XLSTM_ESP_A16 int8_t kmain[16];   /* c1 in every lane */
    XLSTM_ESP_A16 int8_t kmain2[16];  /* c2 in every lane */
    XLSTM_ESP_A16 int8_t ktail[16];   /* c1 below column t, 0 above */
    XLSTM_ESP_A16 int8_t ktail2[16];  /* c2 below column t, 0 above */
    XLSTM_ESP_A16 int8_t vtail[16];   /* v's last t values, zero-padded */
    const int8_t* kp;
    int i, g, k;

    /* Nothing to write, so nothing to read either: the constants below touch
     * v, which the scalar body would not have read for an empty row range. */
    if (rows <= 0) {
        return;
    }

    for (k = 0; k < 16; ++k) {
        kmain[k] = (int8_t)c1;
        kmain2[k] = (int8_t)c2;
        ktail[k] = (k < t) ? (int8_t)c1 : (int8_t)0;
        ktail2[k] = (k < t) ? (int8_t)c2 : (int8_t)0;
        vtail[k] = (k < t) ? v[(nfull << 4) + k] : (int8_t)0;
    }
    kp = kmain;
    XLSTM_ESP_LOADQ("q6", kp);
    if (split) {
        kp = kmain2;
        XLSTM_ESP_LOADQ("q7", kp);
    }

    for (i = 0; i < rows; ++i) {
        const int8_t* p = M + (size_t)i * stride;
        const int sm = (int)((uintptr_t)p & 15u);
        /* An operand already on a block boundary needs no second block, and
         * an offset of 0 makes its pair of loads read the same one twice.
         * That is deliberate rather than a wasted load in disguise: with
         * SAR_BYTE 0 the window IS that block, so the alternative is a second
         * instruction sequence for a case this one already answers - and it
         * would be reading the block AFTER the operand, which for an aligned
         * row is the one place a load here could leave the caller's buffer. */
        const int8_t* ml = p;
        const int8_t* mh = p + (sm ? 16 : 0);
        const int8_t* vl = v;
        const int8_t* vh = v + voff;
        const int8_t* tv = vtail;
        const int8_t* tk = ktail;
        const int32_t zero = 0;
        int32_t acc;

        XLSTM_ESP_ACCX_ZERO(zero);
        for (g = 0; g < nfull; ++g) {
            XLSTM_ESP_S8_GROUP(ml, mh, vl, vh);
            if (split) {
                XLSTM_ESP_S8_GROUP_C2();
            }
        }

        /* ml is p + 16*nfull now, the partial group's first column. Its
         * upper block is loaded only when the row's last byte lies in it,
         * which is exactly when sm + t exceeds a block; the address is then
         * up to 15 bytes past the row, so it is formed as an integer rather
         * than by walking a pointer off the end of the object. */
        mh = (const int8_t*)((uintptr_t)ml + ((sm + t > 16) ? 16u : 0u));
        XLSTM_ESP_S8_TAIL(ml, mh, tv, tk);
        if (split) {
            tk = ktail2;
            XLSTM_ESP_S8_TAIL_C2(tk);
        }

        XLSTM_ESP_ACCX_READ(acc);
        out[i] = acc;
    }
}

void xlstm_matvec_s8(const int8_t* M, const int8_t* v,
                     int32_t* out, int rows, int cols, int32_t v_zp)
{
    /* Two things leave the vector body, and neither is alignment: a zero
     * point two int8 lanes cannot carry, and a cols of 0 or less, for which
     * a row is not a range of addresses at all. */
    const int fast = cols > 0 &&
                     v_zp >= -XLSTM_ESP_ZP_MAX && v_zp <= XLSTM_ESP_ZP_MAX;
    /* Guarded so that the negation itself cannot overflow on an INT32_MIN
     * zero point that is heading for the scalar body anyway. */
    const int32_t c = fast ? -v_zp : 0;
    const int32_t c1 = (c < -128) ? -128 : ((c > 127) ? 127 : c);
    const int32_t c2 = c - c1;

    XLSTM_ESP_COUNT_S8(fast);

    if (!fast) {
        xlstm_scalar_matvec_s8(M, v, out, rows, cols, v_zp);
    } else if (c2 != 0) {
        xlstm_esp_matvec_s8_body(M, v, out, rows, cols, (int)c1, (int)c2, 1);
    } else {
        xlstm_esp_matvec_s8_body(M, v, out, rows, cols, (int)c1, 0, 0);
    }
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
