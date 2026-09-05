# Hardware target

## Target class

The target platform is the STM32F427-class Cortex-M4 used on Pixhawk 2.4.8 hardware.

The STM32F427 family datasheet specifies:

- Arm Cortex-M4 CPU with FPU
- up to 180 MHz
- up to 2 MB internal Flash
- up to 256+4 KB SRAM depending on part/package configuration

The development board probe used during this project reported a 2 MB Flash-class device.

## Why this target is difficult

The FP32 checkpoint requires about 4.83 MB for 1,206,864 32-bit weights, so it cannot be stored directly in 2 MB internal Flash. Quantized storage is therefore required for a fully on-chip implementation.

Approximate raw weight payloads are:

| Weight format | Payload |
|---|---:|
| FP32 | 4,827,456 bytes |
| INT8 | 1,206,864 bytes |
| INT4 | 603,432 bytes before scales/metadata |

With MQA, a 128-token, six-layer int8 KV cache has a theoretical payload of 55,296 bytes. Runtime code, activations, stack, serial buffers, tokenizer state and allocator overhead must fit in the remaining SRAM.

## Scope rule

This project intentionally does not use an SD card to stream model weights and does not assume external PSRAM or an NPU for the target claim.

## Current deployment status

Earlier bring-up work demonstrated custom code execution and serial output on the STM32F427/Pixhawk path, including a tiny RNN and hardware diagnostics. The **final 1.206M-parameter Transformer firmware is not contained in the uploaded archive used to publish this repository and has not been represented here as completed**.
