# Training pipeline

This directory contains the original Python scripts used during the TinyChat-F427 v2 experiment. They are preserved as executable experiment code rather than rewritten pseudocode.

## Recommended order

1. `build_corpus.py` — build the raw corpus from public datasets.
2. `repair_control_tokens.py` — restore the conversational control-token spelling used by the project.
3. `verify_corpus.py` — check corpus integrity.
4. `train_tokenizer.py` — train the 1,024-token byte-level BPE tokenizer from scratch.
5. `audit_tokens.py` — measure source/token composition.
6. `pack_tokens.py` — create packed uint16 training streams.
7. `benchmark_architecture.py` — verify the exact 1,206,864-parameter architecture.
8. `tune_cpu.py` — benchmark CPU thread/batch settings.
9. `qualify_training.py` — short learning-rate and stability qualification.
10. `train_full.py` — run the 30,003,200-token baseline training.
11. `build_stage_c.py`, `build_stage_c_v2.py`, `train_stage_c_v2.py` — early conversational specialization experiments.
12. `build_stage_d.py`, `build_stage_d_v2.py`, `train_stage_d_v2.py` — controlled factual/refusal specialization.
13. `build_stage_e.py`, `train_stage_e.py` — conversation repair with factual guardrails.
14. `train_stage_f.py` — final conversation-generalization repair.

`tinychat_model.py` is kept in this directory because the original scripts import it directly. The same architecture is also exposed as `tinychat/model.py` for package-style imports.

Run scripts from the repository root so relative paths such as `data/`, `tokenizer/`, and `checkpoints/` resolve consistently.
