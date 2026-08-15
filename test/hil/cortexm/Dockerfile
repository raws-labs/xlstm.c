# Cortex-M QEMU HIL harness: builds three firmware ELFs (M4/an386,
# M7/an500, M33/an505); CMD boots each headless under QEMU and exits
# non-zero on any hang, crash, or missing ##srig-exit:N## sentinel.
#   docker build -f test/hil/cortexm/Dockerfile -t xlstm-hil-cortexm .
#   docker run --rm xlstm-hil-cortexm
FROM debian:bookworm-slim
SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc-arm-none-eabi libnewlib-arm-none-eabi libstdc++-arm-none-eabi-newlib \
      qemu-system-arm make \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY include/ include/
COPY src/ src/
COPY test/ test/
WORKDIR /workspace/test/hil/cortexm
RUN make all size
# Firmware spins forever after its sentinel, so poll for it and kill qemu
# ourselves. HIL_CORTEXM_CORES restricts to a subset (e.g. "m4");
# HIL_QEMU_TIMEOUT overrides the per-core boot timeout in seconds.
CMD set -uo pipefail; rc=0; \
    for core in ${HIL_CORTEXM_CORES:-m4 m7 m33}; do \
      case "$core" in m4) mach=mps2-an386 ;; m7) mach=mps2-an500 ;; m33) mach=mps2-an505 ;; \
        *) echo "$core: unknown core"; rc=1; continue ;; esac; \
      log="build/qemu_$core.log"; : >"$log"; \
      qemu-system-arm -M "$mach" -kernel "build/hil_$core.elf" -nographic -semihosting-config enable=on,target=native >"$log" 2>&1 & \
      pid=$!; t=0; \
      while [ "$t" -lt "${HIL_QEMU_TIMEOUT:-60}" ] && ! grep -qE '##srig-exit:[0-9]+##' "$log" && kill -0 "$pid" 2>/dev/null; do \
        sleep 1; t=$((t + 1)); \
      done; \
      kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; \
      echo "=== $core ($mach) ==="; cat "$log"; echo; \
      grep -qE '##srig-exit:0##' "$log" || { echo "$core: FAIL"; rc=1; }; \
    done; \
    exit $rc
