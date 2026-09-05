# Training history

The project was trained from random initialization. No pretrained model weights or pretrained tokenizer were used.

## Base corpus

`training/build_corpus.py` constructs a local corpus from WikiText-103, OpenAssistant OASST1, Databricks Dolly 15k, and SQuAD. The builder performs normalization and exact deduplication with SQLite. The tokenizer is then trained from scratch as a 1,024-token byte-level BPE.

The packed training stream contained 31,979,337 tokens after document boundary tokens. The full baseline optimization consumed 30,003,200 effective tokens.

## Training qualification

Four peak learning rates were evaluated for 400 steps on real data. The recorded final validation losses were:

| Peak LR | Final validation loss |
|---:|---:|
| 3e-4 | 4.3383 |
| 6e-4 | 3.8998 |
| 1e-3 | 3.7290 |
| 1.5e-3 | **3.7068** |

The 1.5e-3 candidate was selected for the main pretraining run.

## 30M-token baseline

The full run used two phases:

- Phase A: 24,002,560 tokens
- Phase B: 6,000,640 tokens

Final baseline validation loss: **2.7485**.

The baseline checkpoint is retained as `checkpoints/tinychat_f427_baseline_30M.pt`.

## Specialization history

The project intentionally preserves unsuccessful as well as successful stages.

- **Stage C** attempted short conversational specialization from OASST1, Dolly and SQuAD. It produced fluent but often irrelevant answers. A major design problem was using SQuAD questions without their supporting passages.
- **Stage D** switched to a small auditable factual/definition corpus, assistant-only loss, explicit refusal examples, and true holdouts for arithmetic and synthetic unknown entities.
- **Stage E** repaired common conversational intents while rehearsing facts and refusal behavior.
- **Stage F** added more conversational paraphrases and retained factual/refusal guardrails.

The final Stage-F checkpoint is `checkpoints/tinychat_f427_final_trained_fp32.pt`.

## CPU configuration used for the recorded run

- 4 PyTorch threads
- batch size 32
- context 128
- 4,096 tokens per optimization step during the main run

See `results/` for the recorded logs and JSON histories.
