# The CI toolchain image: every cross compiler and emulator the gates need,
# baked in, so a job installs nothing.
#
# The reason is measured. On a GitHub runner, apt fetching 57 MB from Ubuntu's
# mirror ran at 217 kB/s and outlived a ten-minute job, while 190 MB of release
# tarballs from GitHub's CDN arrived in 13 s. Acquisition was about 90% of every
# cross job; the tests themselves are 3 s each.
#
# 24.04 is not incidental. test/perf_baseline.txt's instruction counts were
# recorded with the gcc and valgrind this release ships, and `make perf` refuses
# to compare across a toolchain change. The assertion near the end of this file
# fails the image build rather than shipping an image that silently turns that
# gate red - or, worse, one that measures the compiler instead of the code.
#
# Everything is pruned to the targets the gates actually build for, because the
# image is now on the critical path of every job: a pull that costs more than
# the apt it replaced would be no gain at all. Unpruned this is 5.6 GB, almost
# all of it multilib for parts this repository never targets.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# apt is the only thing here still talking to an Ubuntu mirror, so the guards
# live where a throttled mirror still bites. Two failures seen: a mirror that
# accepts a connection and then goes silent, which blocks apt indefinitely
# because it has no default fetch timeout; and one that answers at 217 kB/s.
# The first needs a bound, the second needs a different mirror.
#
# The prune runs in this layer, not a later one: bytes deleted in a later layer
# are still bytes that get pulled.
RUN set -eu; \
    opts="-o Acquire::Retries=3 \
          -o Acquire::http::Timeout=20 \
          -o Acquire::https::Timeout=20"; \
    pkgs="build-essential make git python3 ca-certificates curl xz-utils \
          valgrind \
          gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
          gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf qemu-user \
          gcc-arm-none-eabi binutils-arm-none-eabi libnewlib-arm-none-eabi \
          libstdc++-arm-none-eabi-newlib qemu-system-arm \
          libpixman-1-0 libsdl2-2.0-0 libslirp0"; \
    ok=0; \
    for m in http://archive.ubuntu.com/ubuntu \
             http://azure.archive.ubuntu.com/ubuntu \
             http://us.archive.ubuntu.com/ubuntu ; do \
      sed -i -E "s#https?://[a-z0-9.]*archive\.ubuntu\.com/ubuntu#$m#g" \
        /etc/apt/sources.list.d/ubuntu.sources; \
      if apt-get $opts update \
         && apt-get $opts install -y --no-install-recommends $pkgs; then \
        echo "installed via $m"; ok=1; break; \
      fi; \
      echo "mirror $m did not serve the packages, trying the next"; \
    done; \
    [ "$ok" = 1 ]; \
    rm -rf /var/lib/apt/lists/* /usr/share/doc /usr/share/man /usr/share/info; \
    \
    # gcc-arm-none-eabi ships newlib and libgcc for two dozen Arm variants, \
    # 2.4 GB of it. The helium gate builds one: whatever gcc itself selects \
    # for the flags in the Makefile's test-helium target. Ask the driver \
    # rather than hard-coding the directory name, so a toolchain that maps \
    # cortex-m55 somewhere else prunes correctly instead of silently wrongly. \
    keep=$(arm-none-eabi-gcc -mcpu=cortex-m55 -mthumb -mfloat-abi=hard \
           -print-multi-directory); \
    gccdir=$(dirname "$(arm-none-eabi-gcc -print-libgcc-file-name)"); \
    libdir=$(dirname "$(arm-none-eabi-gcc -print-file-name=libc.a)"); \
    for d in $(arm-none-eabi-gcc -print-multi-lib | cut -d';' -f1); do \
      case "$d" in .|"$keep") continue ;; esac; \
      rm -rf "$gccdir/$d" "$libdir/$d"; \
    done; \
    find "$gccdir" "$libdir" -mindepth 1 -type d -empty -delete; \
    \
    # qemu-user installs an emulator for every architecture it supports. Two \
    # of them are used here; qemu-system-arm is the third binary, from the \
    # other package. lto-dump is 110 MB of gcc tooling nothing here invokes. \
    find /usr/bin -maxdepth 1 -name 'qemu-*' \
      ! -name qemu-arm ! -name qemu-aarch64 ! -name qemu-system-arm -delete; \
    find /usr/bin -maxdepth 1 -name '*lto-dump*' -delete

# The two Espressif release tarballs, pinned by URL, unpacked at build time so
# the esp gate fetches nothing at all. Both versions are deliberate: the
# toolchain is the one ESP-IDF v5.4 ships, so the gate compiles the kernels
# with the compiler a user of that IDF gets, and QEMU has to be Espressif's
# fork because upstream models no esp32s3 machine. libpixman, libsdl2 and
# libslirp above are qemu-system-xtensa's DT_NEEDED entries - SDL2 is linked
# rather than dlopened, so the binary will not start without it even under
# -display none.
ARG XTENSA_TOOLCHAIN=https://github.com/espressif/crosstool-NG/releases/download/esp-14.2.0_20241119/xtensa-esp-elf-14.2.0_20241119-x86_64-linux-gnu.tar.xz
ARG ESP_QEMU=https://github.com/espressif/qemu/releases/download/esp-develop-9.2.2-20260417/qemu-xtensa-softmmu-esp_develop_9.2.2_20260417-x86_64-linux-gnu.tar.xz
RUN set -eu; \
    fetch="curl -fsSL --retry 5 --retry-delay 5 --retry-all-errors \
           --connect-timeout 30 --max-time 900"; \
    $fetch "$XTENSA_TOOLCHAIN" | tar -xJ -C /opt; \
    $fetch "$ESP_QEMU" | tar -xJ -C /opt; \
    # The toolchain carries newlib and picolibc for esp32, esp32s2 and \
    # esp32s3. The gate builds esp32s3; the other two are 560 MB. \
    for core in esp32 esp32s2; do \
      rm -rf "/opt/xtensa-esp-elf/xtensa-esp-elf/lib/$core" \
             "/opt/xtensa-esp-elf/picolibc/xtensa-esp-elf/lib/$core"; \
    done; \
    rm -rf /opt/xtensa-esp-elf/share/doc /opt/xtensa-esp-elf/share/man \
           /opt/xtensa-esp-elf/share/info
ENV PATH=/opt/xtensa-esp-elf/bin:/opt/qemu/bin:$PATH

# A container job checks the workspace out as root over a directory the runner
# created as another user, which git reads as dubious ownership and refuses to
# touch. check-internal-refs is `git ls-files`, so without this the hygiene
# gate cannot see the tree at all.
RUN git config --system --add safe.directory '*'

# Fail here rather than in the perf job. `make perf` compares gcc --version
# against the line recorded in the baseline and refuses to run on a mismatch,
# so an image whose gcc has moved does not measure anything - it just goes red
# in a way that reads like a code regression. valgrind is checked too: the
# baseline records it, and a different version can count differently.
COPY test/perf_baseline.txt /tmp/perf_baseline.txt
RUN set -eu; \
    want_cc=$(sed -n 's/^# toolchain: //p' /tmp/perf_baseline.txt); \
    want_vg=$(sed -n 's/^# valgrind: *//p' /tmp/perf_baseline.txt); \
    got_cc=$(gcc --version | head -1); \
    got_vg=$(valgrind --version); \
    [ "$want_cc" = "$got_cc" ] || { \
      echo "gcc differs from the perf baseline:"; \
      echo "  baseline: $want_cc"; echo "  image:    $got_cc"; exit 1; }; \
    [ "$want_vg" = "$got_vg" ] || { \
      echo "valgrind differs from the perf baseline:"; \
      echo "  baseline: $want_vg"; echo "  image:    $got_vg"; exit 1; }; \
    echo "perf toolchain matches the baseline: $got_cc / $got_vg"; \
    rm -f /tmp/perf_baseline.txt

# Every binary the jobs invoke, plus every archive the two bare-metal links
# pull in. The second half is what keeps the pruning above honest: a multilib
# directory removed in error shows up here as a library the driver can no
# longer resolve, and fails the image build rather than a kernel gate.
RUN set -eu; \
    for t in gcc g++ make git python3 valgrind \
             aarch64-linux-gnu-gcc aarch64-linux-gnu-g++ qemu-aarch64 \
             arm-linux-gnueabihf-gcc arm-linux-gnueabihf-g++ qemu-arm \
             arm-none-eabi-gcc arm-none-eabi-g++ qemu-system-arm \
             xtensa-esp32s3-elf-gcc xtensa-esp32s3-elf-g++ \
             qemu-system-xtensa ; do \
      command -v "$t" >/dev/null || { echo "missing: $t"; exit 1; }; \
    done; \
    qemu-system-arm -M mps3-an547 -display none -version >/dev/null; \
    qemu-system-xtensa -M esp32s3 -display none -version >/dev/null; \
    for cc_flags in \
      "arm-none-eabi-gcc -mcpu=cortex-m55 -mthumb -mfloat-abi=hard" \
      "xtensa-esp32s3-elf-gcc" ; do \
      for lib in libc.a libm.a libgcc.a libstdc++.a libsupc++.a; do \
        p=$($cc_flags -print-file-name=$lib); \
        [ -f "$p" ] || { echo "$cc_flags cannot resolve $lib"; exit 1; }; \
      done; \
    done; \
    p=$(arm-none-eabi-gcc -mcpu=cortex-m55 -mthumb -mfloat-abi=hard \
        -print-file-name=librdimon.a); \
    [ -f "$p" ] || { echo "cannot resolve librdimon.a"; exit 1; }; \
    echo "toolchain image OK"
