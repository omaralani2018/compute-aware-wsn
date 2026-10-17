#!/bin/sh
# Builds the ARM host-check binaries with the firmware's flags (from the RIOT verbose build) and runs them under qemu-arm.
C="-Os -std=c11 -fwrapv -fshort-enums -ffunction-sections -fdata-sections -specs=nano.specs -nostartfiles -Wl,--gc-sections -Wall"
arm-none-eabi-gcc -mcpu=cortex-m3 -mthumb -mfloat-abi=soft $C -o h_m3_float harness.c -lm -lc -lgcc
arm-none-eabi-gcc -mcpu=cortex-m3 -mthumb -mfloat-abi=soft -DNETX_INT=1 $C -o h_m3_int harness.c -lm -lc -lgcc
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 $C -o h_nrf harness.c -lm -lc -lgcc
for b in m3_float m3_int nrf; do qemu-arm ./h_$b > out_$b.txt; done
python3 check.py
