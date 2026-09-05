# Architecture

TinyChat-F427 uses a compact decoder-only Transformer chosen around a severe embedded-memory target rather than for architectural novelty.

## Locked configuration

| Item | Value |
|---|---:|
| Vocabulary | 1,024 |
| Model width | 144 |
| Layers | 6 |
| Query heads | 4 |
| KV heads | 1 |
| Head dimension | 36 |
| FFN width | 288 |
| Context | 128 tokens |
| Attention | causal MQA |
| Normalization | Pre-RMSNorm |
| Position | RoPE |
| FFN | SwiGLU |
| Biases | none |
| Embedding / output | tied |
| Parameters | **1,206,864** |

## Exact parameter accounting

The tied token embedding contributes `1024 x 144 = 147,456` parameters.

Each block contains:

- Q projection: `144 x 144 = 20,736`
- K projection: `144 x 36 = 5,184`
- V projection: `144 x 36 = 5,184`
- output projection: `144 x 144 = 20,736`
- SwiGLU projections: `3 x (144 x 288) = 124,416`
- two RMSNorm vectors: `2 x 144 = 288`

Total per block: **176,544**.

Six blocks contribute **1,059,264** parameters. Adding the token embedding and final RMSNorm gives:

`1,059,264 + 147,456 + 144 = 1,206,864`.

## Why MQA

A single KV head reduces cache growth relative to standard multi-head attention. For a 128-token context, six layers, one KV head, head dimension 36, and int8 KV storage, the theoretical cache payload is:

`2 x 36 x 6 x 128 = 55,296 bytes`.

This is a design-budget calculation, not a measured final-firmware SRAM figure.
