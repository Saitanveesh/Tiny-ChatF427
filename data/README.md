# Data

This repository intentionally does not redistribute the large raw corpora, SQLite deduplication database, packed token streams, or generated NumPy training tensors.

The original experiment used public-source data from WikiText-103, OpenAssistant OASST1, Databricks Dolly 15k, and SQuAD. The corpus and packed training data can be regenerated with the scripts in `training/`.

This directory keeps only compact evaluation artifacts required to inspect the reported held-out results:

- `eval/stage_d_v2/eval_paraphrase.json`
- `eval/stage_d_v2/eval_arithmetic_holdout.json`
- `eval/stage_d_v2/eval_unknown_holdout.json`
- `eval/stage_e/conversation_eval.json`

Large generated files are excluded to keep the repository cloneable and to avoid redistributing upstream dataset content.
