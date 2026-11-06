# Project status

This file distinguishes **implemented source**, **historical results** and **future deployment claims**.

| Component | Status | Notes |
|---|---|---|
| Reference PyTorch Transformer | Implemented | Importable under `tinychat/` |
| Frozen byte-level BPE tokenizer | Included | `tokenizer/tokenizer.json` |
| Historical corpus preparation | Partially included | Source downloads depend on upstream datasets |
| Historical 30M-token / Stage-F training | Recorded | Full original trainer scripts are **not** included |
| Evaluation metrics | Recorded | Some fixtures and logs included; complete execution scripts absent |
| Interactive PC chat | Client present | Final checkpoint binary must be provided separately |
| STM32F427 hardware validation | Previously performed | See `firmware/` for scope and logs |
| Final Transformer quantization and MCU inference | Not demonstrated | Requires code, testing and measurements |

## Do not conflate

- A **recorded PC model score** with general open-ended assistant accuracy.
- A **theoretical INT8 memory budget** with measured MCU Flash/SRAM usage.
- **Earlier tiny RNN/TinyLM hardware experiments** with deployed execution of the 1.206M-parameter Transformer.
- A **checkpoint hash/manifest** with a downloadable checkpoint.
- Historical documentation of training stages with a fully present public training pipeline.

See [reproducibility](REPRODUCIBILITY.md) and the [roadmap](ROADMAP.md) for the next verifiable milestones.
