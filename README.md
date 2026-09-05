# TinyChat-F427

TinyChat-F427 is an experimental, from-scratch autoregressive Transformer designed around the memory limits of an STM32F427-class Cortex-M4 system.

The project asks a narrow engineering question: **how much short-form language capability can be retained when the model is designed for a microcontroller with roughly 2 MB of internal Flash and 256 KB of SRAM, without relying on a cloud model?**

This repository contains the model definition, corpus/tokenizer pipeline, training stages, evaluation scripts, PC-side interactive inference, compact evaluation data, checkpoints, and experiment logs used during development.

> **Status:** the 1.206M-parameter Transformer is fully trained and evaluated on PC. The repository does **not** claim that this final Transformer checkpoint has already been deployed end-to-end on the STM32F427. The earlier MCU bring-up and tiny-RNN experiments established the hardware path, but the final Transformer firmware source was not present in the archived project bundle used to publish this repository.

![Pixhawk / STM32F427 test hardware](docs/images/board_1.jpg)

## What was built

- Decoder-only Transformer trained from random initialization
- **1,206,864 trainable parameters**
- **1,024-token byte-level BPE** tokenizer trained from scratch
- 6 Transformer blocks
- model width 144
- 4 query heads, 1 KV head (multi-query attention)
- head dimension 36
- SwiGLU FFN, width 288
- Pre-RMSNorm
- RoPE positional encoding
- tied token embedding / LM output weights
- 128-token context window
- no linear biases
- assistant-only specialization stages for short answers and refusal behavior

The architecture is intentionally small. It is not intended to compete with general-purpose LLMs.

## Key results

The final Stage-F candidate preserved useful behavior on controlled held-out tests:

| Test | Result |
|---|---:|
| Factual QA (controlled, non-math) | **84.1%** |
| Held-out conversation semantic accuracy | **75.0%** |
| Held-out conversation exact accuracy | **75.0%** |
| Synthetic unknown/refusal | **100.0%** |
| Stage-D unseen arithmetic | **14.3%** |

These numbers must be read together with the free-form test. In unrestricted interaction the model produced severe intent and knowledge errors, for example answering a speed-of-light question with a statement about the heart. The project therefore treats the gap between **bounded benchmark performance** and **open-ended conversational reliability** as an important result, not something to hide.

See [docs/EVALUATION.md](docs/EVALUATION.md).

## Quick start

### 1. Environment

Python 3.10 was used for the recorded run.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### 2. Run the frozen PC model

```bash
python inference/chat_tinychat.py
```

The interactive script uses greedy decoding, a 48-token generation ceiling, a two-sentence post-limit, and a small deterministic arithmetic router.

### 3. Verify the architecture

```bash
python training/benchmark_architecture.py
```

Expected parameter count:

```text
1,206,864
```

## Reproducing the training path

The large generated corpora and packed binary streams are intentionally **not committed**. They can be reconstructed from the public source datasets by the included scripts.

A typical run is:

```bash
python training/build_corpus.py
python training/repair_control_tokens.py
python training/verify_corpus.py
python training/train_tokenizer.py
python training/audit_tokens.py
python training/pack_tokens.py
python training/qualify_training.py
python training/train_full.py
```

The later specialization stages are preserved as separate scripts because they are part of the experimental history rather than a single polished training recipe. See [docs/TRAINING.md](docs/TRAINING.md).

## Training data

The corpus builder references:

- WikiText-103 (`Salesforce/wikitext`, `wikitext-103-raw-v1`)
- OpenAssistant OASST1 (`OpenAssistant/oasst1`)
- Databricks Dolly 15k (`databricks/databricks-dolly-15k`)
- SQuAD (`rajpurkar/squad`)

This repository does not redistribute the full raw corpora. Users are responsible for the licenses and terms of the upstream datasets.

The final pretraining run processed **30,003,200 tokens**. The recorded final validation loss of the 30M baseline was **2.7485**.

## Frozen artifacts

| Artifact | SHA-256 |
|---|---|
| `tokenizer/tokenizer.json` | `e0ab232a332b5f1f9b55094e21c97e69350b2f8045e6d49de935a61a2dd52c85` |
| `checkpoints/tinychat_f427_baseline_30M.pt` | `089b72b339c5edc56877214b2a028b97af5e94867c2cb2c5caec608cc31b7a5c` |
| `checkpoints/tinychat_f427_final_trained_fp32.pt` | `3e34e93bb58680df6e0269659ed90ad553740122a74963dfea83706aad3e4768` |

## Hardware target

The target class is STM32F427 / Pixhawk 2.4.8 hardware. The STM32F427 family provides an Arm Cortex-M4 with FPU, up to 180 MHz, up to 2 MB Flash, and up to 256+4 KB SRAM depending on part configuration.

The project intentionally avoids using SD-card model streaming or external PSRAM as part of the target claim.

See [docs/HARDWARE_TARGET.md](docs/HARDWARE_TARGET.md).

## Repository structure

```text
tinychat/       model implementation
training/       corpus, tokenizer, pretraining and specialization scripts
evaluation/     controlled evaluation utilities
inference/      interactive PC inference
tokenizer/      frozen 1,024-token BPE
checkpoints/    baseline and final FP32 checkpoints
data/eval/      compact held-out evaluation sets
results/        recorded metrics, histories and logs
docs/           architecture, training, evaluation and hardware notes
firmware/       hardware-deployment status note
```

## Limitations

TinyChat-F427 is a research prototype with a deliberately tiny model. It can reproduce a bounded set of learned facts and conversational intents, but it is **not a reliable general-knowledge assistant**. Controlled accuracy must not be interpreted as open-ended QA accuracy. Arithmetic generalization in the neural model is poor. The final STM32F427 Transformer deployment remains to be completed and measured.

## License

Source code is released under the MIT License. Upstream datasets retain their own licenses and terms. Checkpoint use should also respect the licenses of the data used to train it.
