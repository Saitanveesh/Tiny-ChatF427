# Project status

## Completed

- hardware compute-path qualification on the STM32F427/Pixhawk target class
- from-scratch byte-level BPE tokenizer
- exact 1,206,864-parameter Transformer reference implementation
- corpus construction and packing pipeline
- CPU training qualification
- 30M-token baseline training
- multiple assistant-specialization stages
- controlled factual, conversational, arithmetic and refusal evaluations
- free-form PC interactive challenge test
- frozen baseline and final FP32 checkpoints

## Not completed in this repository

- production-quality integer Transformer kernels for STM32F427
- W4A8/INT8 end-to-end quantized Transformer export
- measured peak SRAM for final Transformer firmware
- measured per-token STM32F427 latency for the final Transformer
- final on-device byte-BPE tokenizer implementation
- end-to-end final Transformer deployment on the Pixhawk board

These are deliberately listed as open items to keep the repository reproducible and to prevent the PC reference results from being mistaken for MCU deployment results.
