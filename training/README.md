# Training pipeline

This directory contains the **subset of training-related scripts published in this source snapshot**, including corpus preparation, tokenizer creation, token packing and a CPU architecture/training benchmark.

## Files currently available

| Script | Purpose |
|---|---|
| `build_corpus.py` | Build local text corpora from upstream datasets |
| `repair_control_tokens.py` | Normalize/repair control token spelling |
| `verify_corpus.py` | Check prepared corpus |
| `train_tokenizer.py` | Train 1,024-token byte-level BPE |
| `audit_tokens.py` | Inspect token composition |
| `pack_tokens.py` | Pack token streams |
| `benchmark_architecture.py` | Validate architecture **and run CPU training benchmarks** |
| `tinychat_model.py` | Historical training-side model definition |

## Important missing steps

The original experiment also used scripts named `tune_cpu.py`, `qualify_training.py`, `train_full.py`, and `build/train_stage_*.py` for later specialization. **Those scripts are not present in the current public repository**. Documentation of their historical results does not make this a runnable end-to-end training pipeline.

If you plan to contribute a replacement or restored trainer, start with an issue and specify the data/metric compatibility target.

Work from the repository root so relative paths resolve. See [training history](../docs/TRAINING.md) and [reproducibility notes](../docs/REPRODUCIBILITY.md).
