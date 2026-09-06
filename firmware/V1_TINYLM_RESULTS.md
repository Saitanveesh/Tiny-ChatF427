# Earlier TinyLM-F427 MCU baseline

Before the 1.206M-parameter Transformer, the project used a much smaller character RNN to validate the complete MCU inference path.

Recorded baseline:

- vanilla character RNN
- vocabulary: 96
- hidden size: 14
- parameters: 2,994
- FP32 generation: PASS on STM32F427

Recorded FP32 STM32 timing:

```text
hidden update  : 35.864 us
output layer   : 101.046 us
full character : 136.921 us
throughput     : 7303.48 char/s
```

An INT8-storage / FP32-compute experiment reduced model storage by about 74.83%, but it was slower on the MCU because each weight was dequantized during computation. This result motivated the later conclusion that a practical multi-million-parameter deployment needs true integer kernels rather than storage-only quantization.
