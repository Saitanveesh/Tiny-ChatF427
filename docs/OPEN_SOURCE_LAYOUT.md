# Open-source layout

The repository is intentionally organized by function rather than by the chronological order in which experiments were created.

- `tinychat/` — importable model definition
- `training/` — corpus, tokenizer, packing, architecture verification and training scripts
- `tokenizer/` — frozen tokenizer artifacts
- `evaluation/` — evaluation utilities
- `data/eval/` — compact held-out evaluation data
- `inference/` — interactive PC reference inference
- `results/` — recorded metrics and logs
- `checkpoints/` — checkpoint manifest and artifact instructions
- `firmware/` — STM32F427 hardware bring-up evidence and deployment boundary
- `docs/` — architecture, training, evaluation, target-hardware and reproducibility notes

Large generated corpora and intermediate binaries are not placed in the source tree. This keeps the repository reviewable while retaining the scripts needed to regenerate them.
