# Reproducibility notes

The repository separates source-controlled experiment code from large generated artifacts.

## Reproducible components

- exact Transformer architecture and parameter count
- tokenizer training code and frozen tokenizer vocabulary/merge files
- corpus construction scripts
- token packing pipeline
- training qualification and full-training scripts
- Stage-C through Stage-F specialization scripts
- held-out evaluation sets
- training histories and logs
- PC interactive inference code

## Large artifacts intentionally omitted

The following are generated locally and are not committed:

- raw corpus JSONL files
- deduplication SQLite database
- packed `.bin` token streams
- generated `.npy` tensors
- most intermediate checkpoints
- final `.pt` checkpoint binary in this repository snapshot

The source archive recorded a 30,003,200-token full run. The final Stage-F candidate reported 84.1% controlled factual accuracy, 75.0% held-out conversation semantic/exact accuracy, and 100% synthetic unknown/refusal behavior. These bounded metrics must be interpreted together with the documented free-form failures.
