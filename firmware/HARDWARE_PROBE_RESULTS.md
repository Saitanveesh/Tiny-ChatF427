# Hardware probe results

Target board: Pixhawk 2.4.8-class STM32F427 hardware.

Recorded hardware probe values from the project:

```text
Flash size register : 2048 KB
DBGMCU_IDCODE       : 0x20016419
Device ID           : 0x419
Revision ID         : 0x2001
Cortex CPUID        : 0x410FC241
FLASH CLASS         : 2 MB
```

A recorded `TinyChat_HWProbe` build summary was:

```text
Text (B)            : 410128
Data (B)            : 1912
BSS (B)             : 58384
Total Flash Used    : 412040
Free Flash          : 1668720
```

These figures established the physical 2 MB Flash target and a realistic firmware baseline before attempting the larger Transformer design.
