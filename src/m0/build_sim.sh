#!/bin/sh
# Version 1.4. Fetches and builds the tools used for the Cortex-M0 costing of Section 6.6,
# inside run/m0/. Needs git, curl, a C compiler and libssl-dev (the simulator links -lssl -lcrypto).
set -e
# Instruction-set simulator of Nikov et al. [21], at the tag their paper cites (reference [12] there)
git clone -q https://github.com/PicoPET/thumbulator-stm32f0x.git thumb
(cd thumb && git checkout -q teamplay-D4.5 && test "$(git rev-parse HEAD)" = c55ea1b1926b792d7aabb7e3495b55a7b69ef134 && make board COPS="-std=c99 -O3 -Wall -pedantic -fcommon")
# Copy with program output enabled, used only to read the predictions back (counters come from thumb/)
cp -r thumb thumbp
sed -i 's/#define DISABLE_PROGRAM_PRINTING 1/#define DISABLE_PROGRAM_PRINTING 0/' thumbp/sim_support.h
(cd thumbp && make clean >/dev/null && make board COPS="-std=c99 -O3 -Wall -pedantic -fcommon")
# Arm GNU toolchain, xPack 14.2.1-1.1 (Linux x64)
curl -sSL -o gcc.tgz https://github.com/xpack-dev-tools/arm-none-eabi-gcc-xpack/releases/download/v14.2.1-1.1/xpack-arm-none-eabi-gcc-14.2.1-1.1-linux-x64.tar.gz
echo "ed8c7d207a85d00da22b90cf80ab3b0b2c7600509afadf6b7149644e9d4790a6  gcc.tgz" | sha256sum -c -
tar xzf gcc.tgz && rm gcc.tgz
echo "built: thumb/sim_main, thumbp/sim_main, xpack-arm-none-eabi-gcc-14.2.1-1.1/"
