# Data

Large generated training corpora, SQLite deduplication databases, packed `.bin` streams and intermediate `.npy` tensors are intentionally excluded from Git.

They are reproducible using the scripts under `training/` and the upstream public datasets referenced by `training/build_corpus.py`.

Compact held-out JSON evaluation sets are retained under `data/eval/` so the experiment design remains inspectable.
