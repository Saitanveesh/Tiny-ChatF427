# STM32F427 / Pixhawk firmware work

This directory documents the embedded side of TinyChat-F427.

The hardware target is a Pixhawk 2.4.8-class board built around an STM32F427 Cortex-M4. The project intentionally avoids SD-card model streaming and external PSRAM for the final memory-bound target claim.

## Completed hardware work

Earlier bring-up experiments established that the main STM32F427 compute path was usable even though the particular flight-controller board had an IOMCU-related fault for normal flight use.

Completed checks included:

- USB/UART application bring-up
- integer execution
- floating-point execution
- 32 KiB SRAM buffer test
- stable firmware heartbeat
- device and Flash identification
- a tiny character-RNN inference and benchmark path on the MCU

See [`HARDWARE_PROBE_RESULTS.md`](HARDWARE_PROBE_RESULTS.md) and [`V1_TINYLM_RESULTS.md`](V1_TINYLM_RESULTS.md).

## Important status boundary

The archived project bundle supplied for this public repository did not contain the final v2 Transformer MCU firmware source. Therefore the repository does **not** claim that the 1.206M-parameter Transformer has already run end-to-end on the STM32F427.

The PC Transformer model, tokenizer, training pipeline, and evaluation are reproducible here; the final integer/quantized STM32 Transformer kernel remains an open deployment milestone.
