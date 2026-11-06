# Development roadmap

TinyChat-F427 is an experimental project. These are **targets, not promises or shipped capabilities**.

## Now — make the PC reference reproducible

- [x] Publish importable Transformer reference implementation.
- [x] Publish frozen tokenizer and architecture documentation.
- [x] Publish selected held-out data and historical result summaries.
- [x] Add lightweight model-contract tests and clearer first-run documentation.
- [ ] Make the final FP32 checkpoint available with a checked checksum and redistribution review.
- [ ] Restore or reimplement missing historical baseline and Stage-C–F training scripts.
- [ ] Add a reproducible evaluation entrypoint for published metrics.
- [ ] Document a PC inference baseline: hardware, RAM use, tokens/second, output examples.

**Exit criterion:** a fresh clone can obtain approved artifacts, run inference, and reproduce a defined evaluation without guessing which files are needed.

## Next — quantify the embedded feasibility

- [ ] Export and verify quantized weights against PC logits and held-out outputs.
- [ ] Budget model weights, scales, firmware, tokenizer tables, KV cache, activations and stack.
- [ ] Implement and test embedded attention, normalization and matrix kernels.
- [ ] Document SRAM peak and Flash map on the selected exact STM32F427 part.
- [ ] Run repeatable host-vs-target numerical consistency tests.

**Exit criterion:** a checked build and a repeatable memory/accuracy report, with every memory figure identified as measured or estimated.

## Later — on-device Transformer demonstration

- [ ] Implement a fully local MCU tokenizer and autoregressive decode loop.
- [ ] Demonstrate end-to-end prompt to generated tokens on the physical board.
- [ ] Record per-token latency, throughput, energy if instrumented, and failure cases.
- [ ] Compare quantized-model behavior against the frozen PC reference.
- [ ] Release reproducible build instructions and test evidence.

**Exit criterion:** independent contributors can rebuild and verify a working on-device demo without a cloud service, external PSRAM or SD-based model streaming.

## Research questions worth pursuing

1. How much quality is lost under INT8 versus lower-bit weight quantization at this scale?
2. What dominates peak MCU SRAM once the KV cache, scratch buffers and tokenizer are all accounted for?
3. Where do tiny language models fail outside narrow controlled evaluations?
4. What are the measurable latency–memory trade-offs of a single-KV-head attention design on Cortex-M4?

Open an [issue](https://github.com/Saitanveesh/Tiny-ChatF427/issues) for one small, testable contribution rather than attempting the entire roadmap in one PR.
