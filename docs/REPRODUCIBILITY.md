# Reproducibility and artifact availability

TinyChat-F427 publishes the model specification, a tokenizer, part of the corpus preparation path and historical measurements. **It is not presently an end-to-end reproducible Stage-F training release.** This distinction matters when comparing its reported results with independently repeated experiments.

## Available in the public snapshot

- Importable PyTorch reference model with exact 1,206,864-parameter accounting
- Byte-level BPE tokenizer artifacts and tokenizer training script
- Corpus construction, repair, validation and packing scripts
- CPU model/training benchmark script
- Selected compact controlled evaluation fixtures
- Historical logs, summaries, hardware documentation and checkpoint SHA-256 manifest
- PC chat client code that loads a separately provided checkpoint

## Not available in the public snapshot

- Large raw corpora, packed training arrays, SQLite deduplication data and generated NumPy tensors
- Frozen baseline and Stage-F `.pt` checkpoint binaries
- Original `train_full.py`, `qualify_training.py` and historical Stage-C–F training scripts mentioned in older notes
- Many of the original evaluation execution scripts referenced in historical notes
- End-to-end firmware for the final Transformer

Therefore **a fresh clone can inspect the architecture and tokenizer, but cannot reproduce the published Stage-F training/evaluation scores or trained chat end-to-end without additional files**. The corpus-preparation subset can be rerun subject to upstream dataset availability and terms.

## Artifact integrity

The expected filenames and SHA-256 hashes of selected original model checkpoints are in [the manifest](../checkpoints/checkpoint_manifest.json). A checksum verifies the identity of a supplied file; it does **not** make that file available or establish its redistribution rights.

## Interpreting historical metrics

The archived run reports 30,003,200 training tokens, final baseline validation loss 2.7485, and Stage-F controlled factual/held-out conversation results. These are **recorded experimental outputs**, not independently verified benchmark reproductions. Consult [Evaluation](EVALUATION.md) for the deliberately documented free-form failure cases.

## What would make this reproducible?

Publish reviewed checkpoints, restore the missing training/evaluation scripts, lock dependencies and dataset revisions, and provide a single documented command that regenerates an identified evaluation report. See [the roadmap](ROADMAP.md).
