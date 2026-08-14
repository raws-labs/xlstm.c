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
 * ESP32-S3 board layer for the hardware-in-the-loop (HIL) test firmware.
 *
 * Implements test/hil_platform.h (the board shim test/hil_runner.cc talks
 * to) and app_main - the ESP-IDF equivalent of test/hil_platform_host.c's
 * main(). See test/hil_platform.h for what the two shim functions are for
 * and why the surface is this narrow.
 *
 * Boot-output race (see test/hil/README.md for the rig-side story): the rig
 * flashes, opens serial, then optionally sends a trigger byte - output
 * printed before it finishes opening the port can be missed entirely.
 * app_main here (1) waits briefly for that trigger byte with a timeout
 * fallback, so a plain `srig serial` session with no --send still works,
 * (2) runs the fused suite, (3) prints the sentinel, then (4) loops forever
 * re-printing a short summary and the sentinel every few seconds - so a
 * missed or late-attached listener still observes the result. app_main
 * never returns and this file never calls esp_restart().
 * ===========================================================================*/

#include "hil_platform.h"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "sdkconfig.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/uart.h"
#include "driver/uart_vfs.h"

#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_idf_version.h"
#if CONFIG_SPIRAM
#include "esp_psram.h"
#endif

/* Defined in test/hil_runner.cc as extern "C" - same cross-language call
 * test/hil_platform_host.c makes from its own plain-C main(). */
extern int xlstm_hil_run(void);

#define HIL_UART_NUM        UART_NUM_0
#define HIL_UART_RX_BUF     256
#define HIL_BOOT_WAIT_MS    5000
#define HIL_REPEAT_DELAY_MS 5000

static bool s_uart_driver_ready = false;

/* ---------------------------------------------------------------------
 * UART0 bring-up
 *
 * CONFIG_ESP_CONSOLE_UART_DEFAULT=y already routes stdout/stdin to UART0
 * at early boot, but in the default (non-interrupt) mode: printf() writes
 * by polling the TX FIFO directly and getchar()-style reads are
 * non-blocking best-effort. That is fine for output alone, but we need a
 * genuinely blocking, timeout-bounded read for the trigger byte, which
 * needs the real interrupt-driven UART driver.
 *
 * The two are not automatically compatible: installing the interrupt
 * driver without also switching the VFS console over to it leaves two
 * uncoordinated code paths (raw polling writes vs. the driver's ISR)
 * touching the same UART0 peripheral, which is exactly the "output stops
 * entirely" failure mode. The fix, verified against IDF v5.4's own
 * sources (components/console/esp_console_repl_chip.c and
 * examples/common_components/protocol_examples_common/stdin_out.c both do
 * precisely this, in this order, for exactly this reason - console
 * already owns UART0 in default polling mode and they need blocking
 * reads too):
 *
 *   1. uart_driver_install() - installs the interrupt-driven driver.
 *   2. uart_vfs_dev_use_driver() - tells the VFS console layer to route
 *      ALL further UART0 I/O (both printf and read) through that driver
 *      instead of the raw polling path, so nothing is left uncoordinated.
 *
 * uart_is_driver_installed() guards against double-install, matching
 * stdin_out.c's own guard (harmless here since nothing installs it before
 * us, but cheap and keeps this safe if that ever changes upstream).
 * --------------------------------------------------------------------- */
static void hil_uart_bringup(void) {
    if (uart_is_driver_installed(HIL_UART_NUM)) {
        s_uart_driver_ready = true;
        return;
    }

    setvbuf(stdin, NULL, _IONBF, 0);

    esp_err_t err = uart_driver_install(HIL_UART_NUM, HIL_UART_RX_BUF, 0, 0, NULL, 0);
    if (err != ESP_OK) {
        /* Still on the polling console here - this is visible even though
         * the interrupt driver failed to install. */
        printf("hil_esp32s3: uart_driver_install failed (%d) - trigger-byte "
               "wait disabled, falling back to a plain timeout\n", (int)err);
        return;
    }

    uart_vfs_dev_use_driver(HIL_UART_NUM);
    s_uart_driver_ready = true;
}

/* Wait up to HIL_BOOT_WAIT_MS for one byte on UART0, then proceed
 * regardless. Uses the driver directly (uart_read_bytes has a native tick
 * timeout) rather than stdio, so this does not depend on how the VFS
 * console layer's own read semantics behave. If the driver never came up,
 * fall back to a plain delay so the boot-race mitigation still holds. */
static void hil_wait_for_trigger_byte(void) {
    if (!s_uart_driver_ready) {
        vTaskDelay(pdMS_TO_TICKS(HIL_BOOT_WAIT_MS));
        return;
    }

    uint8_t rx_byte = 0;
    int n = uart_read_bytes(HIL_UART_NUM, &rx_byte, 1, pdMS_TO_TICKS(HIL_BOOT_WAIT_MS));
    if (n > 0) {
        hil_platform_println("HIL_TRIGGER: byte received, proceeding");
    } else {
        hil_platform_println("HIL_TRIGGER: timeout waiting for trigger byte, proceeding anyway");
    }
}

/* ---------------------------------------------------------------------
 * test/hil_platform.h implementation
 * --------------------------------------------------------------------- */

void hil_platform_println(const char *line) {
    fputs(line, stdout);
    fputc('\n', stdout);
    fflush(stdout);
    /* Belt-and-suspenders on top of the fflush: block until UART0's TX
     * queue has actually drained onto the wire, not just been handed to
     * the driver's ring buffer. Only meaningful once the interrupt driver
     * owns UART0; on the polling-console fallback, fflush() alone is
     * already synchronous (each character is written by busy-polling the
     * TX FIFO), so skipping this there is not a gap. */
    if (s_uart_driver_ready) {
        uart_wait_tx_done(HIL_UART_NUM, pdMS_TO_TICKS(1000));
    }
}

static const char *hil_chip_model_name(esp_chip_model_t model) {
    switch (model) {
        case CHIP_ESP32:   return "esp32";
        case CHIP_ESP32S2: return "esp32s2";
        case CHIP_ESP32S3: return "esp32s3";
        case CHIP_ESP32C3: return "esp32c3";
        default:           return "unknown";
    }
}

const char *hil_platform_provenance_fields(void) {
    static char buf[320];

    esp_chip_info_t info;
    esp_chip_info(&info);
    /* esp_chip_info_t.revision is MXX-encoded: wafer major*100 + minor. */
    unsigned rev_major = info.revision / 100u;
    unsigned rev_minor = info.revision % 100u;

    /* Runtime-detected, not the sdkconfig.defaults guess (CONFIG_ESPTOOLPY_
     * FLASHSIZE_4MB) - the whole point of reporting this is to let a real
     * run correct that guess from evidence. NULL requests the default/boot
     * flash chip. */
    uint32_t flash_bytes = 0;
    esp_flash_get_size(NULL, &flash_bytes);

#if CONFIG_SPIRAM
    bool psram_present = esp_psram_is_initialized();
#else
    /* esp_psram.c is not even compiled into this build when CONFIG_SPIRAM
     * is off (this project's sdkconfig.defaults deliberately leaves PSRAM
     * disabled), so esp_psram_is_initialized() is not available to call -
     * we already know the answer without asking. */
    bool psram_present = false;
#endif

    snprintf(buf, sizeof(buf),
        "\"platform\":\"esp32s3\",\"chip_model\":\"%s\","
        "\"chip_revision\":\"v%u.%u\",\"chip_cores\":%u,"
        "\"cpu_mhz\":%d,\"flash_size_bytes\":%lu,\"psram\":%s,"
        "\"idf_version\":\"%s\"",
        hil_chip_model_name(info.model), rev_major, rev_minor,
        (unsigned)info.cores,
        (int)CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ,
        (unsigned long)flash_bytes,
        psram_present ? "true" : "false",
        esp_get_idf_version());

    return buf;
}

/* ---------------------------------------------------------------------
 * Entry point
 * --------------------------------------------------------------------- */

void app_main(void) {
    hil_uart_bringup();

    /* Step 1: give the rig time to finish opening the port and (optionally)
     * send its trigger byte, before anything the rig needs to see is
     * printed. */
    hil_wait_for_trigger_byte();

    /* Step 2: run the fused suite. All of its output (provenance banner,
     * per-suite progress, the suites' own pass/fail detail via stdio, and
     * the "##srig-exit:N##" sentinel) goes out over the now-driver-backed
     * UART0 console. */
    int rc = xlstm_hil_run();

    /* Step 3: the sentinel. xlstm_hil_run() already printed it as its own
     * last action; print it again explicitly here so this function's own
     * step list (as specified for this board layer) is visibly complete
     * even if xlstm_hil_run()'s internals ever change. A duplicate line on
     * the wire is harmless - if anything it hedges against the rig
     * catching the stream mid-line. */
    char summary[96];
    snprintf(summary, sizeof(summary), "HIL_DONE: rc=%d", rc);
    hil_platform_println(summary);
    hil_platform_println(rc == 0 ? "##srig-exit:0##" : "##srig-exit:1##");

    /* Step 4: never restart, never return - loop forever so a rig that
     * missed the trigger window or attached late still observes the
     * result. */
    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(HIL_REPEAT_DELAY_MS));
        hil_platform_println(summary);
        hil_platform_println(rc == 0 ? "##srig-exit:0##" : "##srig-exit:1##");
    }
}
