# TinyChat-F427

**A from-scratch 1.206M-parameter autoregressive Transformer built around the limits of an STM32F427-class Cortex-M4 system.**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Parameters](https://img.shields.io/badge/Parameters-1.206M-informational)
![Context](https://img.shields.io/badge/Context-128%20tokens-informational)
![Target](https://img.shields.io/badge/Target-STM32F427%20Cortex--M4-orange)
![License](https://img.shields.io/badge/License-MIT-green)

TinyChat-F427 explores a simple but difficult question:

> **How much useful short-form language behavior can be retained when an autoregressive model is designed for a microcontroller-class platform with roughly 2 MB of internal Flash and 256 KB of SRAM, without cloud inference, external PSRAM, or SD-card model streaming?**

The project includes the model architecture, tokenizer, corpus pipeline, full training path, specialization stages, evaluation scripts, PC-side inference, checkpoints, logs, and STM32/Pixhawk hardware bring-up work.

<p align="center">
  <img src="docs/images/pixhawk_stm32f427.jpg" alt="Pixhawk STM32F427 target hardware" width="520">
</p>
<p align="center"><em>Pixhawk-class STM32F427 hardware used as the embedded target during hardware bring-up.</em></p>

> **Current status:** the final 1.206M-parameter Transformer is trained and evaluated on PC. The earlier TinyLM firmware and hardware diagnostics established the STM32F427 execution path. The final Transformer checkpoint has not yet been demonstrated end-to-end on the MCU, so this repository does not claim completed Transformer deployment on hardware.

---

## Why this project exists

Most language-model work assumes hardware with far more memory than a conventional Cortex-M4 microcontroller. TinyChat-F427 deliberately starts from the opposite direction: the hardware limit is fixed first, then the model is designed around it.

This makes the project useful for studying the boundary between:

- a model that **fits a severe embedded budget**, and
- a model that is actually **reliable enough to behave like a useful language interface**.

The project therefore treats failures as results too. A model can score well on a narrow held-out benchmark while still failing badly on unrestricted questions.

---

## Model at a glance

| Property | TinyChat-F427 |
|---|---:|
| Architecture | Decoder-only Transformer |
| Trainable parameters | **1,206,864** |
| Initialization | Random / from scratch |
| Vocabulary | **1,024-token byte-level BPE** |
| Context length | **128 tokens** |
| Transformer blocks | **6** |
| Model width | **144** |
| Query heads | **4** |
| KV heads | **1** |
| Attention type | Multi-Query Attention |
| Head dimension | **36** |
| FFN | SwiGLU |
| FFN width | **288** |
| Normalization | Pre-RMSNorm |
| Position encoding | RoPE |
| Embedding / LM head | Tied |
| Linear biases | None |

### Parameter accounting

```text
Token embedding      147,456
6 Transformer blocks 1,059,264
Final RMSNorm             144
--------------------------------
Total                1,206,864
```

The architecture is intentionally compact. MQA, tied embeddings, the small vocabulary, and the 128-token context were chosen to reduce storage and runtime memory pressure rather than to introduce a new Transformer primitive.

---

## Training path

TinyChat-F427 was trained from random initialization. No pretrained language-model weights were used.

The main corpus pipeline uses:

- WikiText-103
- OpenAssistant OASST1
- Databricks Dolly 15k
- SQuAD

A custom byte-level BPE tokenizer was trained from scratch with five control tokens:

```text
<BOS> <EOS> <USER> <ASSISTANT> <EOT>
```

The final pretraining run processed:

```text
30,003,200 tokens
```

The recorded final validation loss of the 30M baseline was:

```text
2.7485
```

Later stages introduced assistant-only loss, short-answer behavior, factual rehearsal, conversation repair, and explicit refusal training.

See [docs/TRAINING.md](docs/TRAINING.md) for the stage-by-stage path.

---

## Controlled evaluation

The final Stage-F checkpoint produced the following results on the controlled evaluation sets:

| Test | Result |
|---|---:|
| Factual QA, controlled non-math | **84.1%** |
| Held-out conversation semantic accuracy | **75.0%** |
| Held-out conversation exact accuracy | **75.0%** |
| Synthetic unknown/refusal | **100.0%** |
| Stage-D unseen arithmetic | **14.3%** |

These values describe the specific held-out evaluation sets included with the project. They are **not general-knowledge accuracy numbers**.

### The important failure case

A separate unrestricted terminal test exposed severe semantic failures. Examples included:

```text
USER: what is speed of light
AI:   The heart pumps blood through the body.

USER: are you gpt
AI:   Gravity.

USER: define speed
AI:   Goodbye.
```

That gap is central to the project: **bounded benchmark performance does not automatically translate into robust open-ended conversation at this model scale.**

See [docs/EVALUATION.md](docs/EVALUATION.md).

---

## Hardware target

TinyChat-F427 targets the STM32F427 / Pixhawk 2.4.8 class of hardware.

Relevant device-class constraints are:

```text
CPU      Arm Cortex-M4 + FPU
Clock    up to 180 MHz
Flash    up to 2 MB
SRAM     up to 256+4 KB
NPU      none
PSRAM    not used for the target claim
SD model streaming  not used
Cloud inference     not used
```

The project intentionally uses a legacy, non-accelerated MCU target because the point is to explore how far autoregressive language generation can be pushed under a genuinely small on-chip memory envelope.

See [docs/HARDWARE_TARGET.md](docs/HARDWARE_TARGET.md).

---

## What already ran on STM32F427

Before the Transformer work, the hardware path was validated using smaller firmware experiments:

- custom UART / USB serial communication
- integer execution
- floating-point execution
- SRAM buffer tests
- stable heartbeat execution
- a **2,994-parameter character RNN**
- FP32 TinyLM inference on STM32F427
- INT8-storage TinyLM inference experiments

These experiments established that custom local neural inference and serial interaction were possible on the board. They should not be confused with completion of the final 1.206M-parameter Transformer port.

---

## Quick start

### 1. Create the environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

The recorded development environment used Python 3.10.

### 2. Run the interactive PC model

```bash
python inference/chat_tinychat.py
```

The interactive script uses greedy decoding, a 48-token generation ceiling, a two-sentence response limit, and a deterministic router for simple arithmetic.

### 3. Verify the architecture

```bash
python training/benchmark_architecture.py
```

Expected output:

```text
1,206,864 parameters
```

---

## Reproduce the pretraining pipeline

Large generated corpora and packed token streams are intentionally not committed. They can be reconstructed from the source datasets using the included scripts.

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

The later Stage-C through Stage-F scripts are preserved because they document the experimental progression from broad pretraining to bounded assistant behavior.

---

## Frozen artifacts

| Artifact | SHA-256 |
|---|---|
| `tokenizer/tokenizer.json` | `e0ab232a332b5f1f9b55094e21c97e69350b2f8045e6d49de935a61a2dd52c85` |
| `checkpoints/tinychat_f427_baseline_30M.pt` | `089b72b339c5edc56877214b2a028b97af5e94867c2cb2c5caec608cc31b7a5c` |
| `checkpoints/tinychat_f427_final_trained_fp32.pt` | `3e34e93bb58680df6e0269659ed90ad553740122a74963dfea83706aad3e4768` |

Hashes are provided so experiments can be tied to exact frozen artifacts.

---

## Repository layout

```text
tinychat/       model implementation
training/       corpus, tokenizer, pretraining and specialization scripts
evaluation/     controlled evaluation utilities
inference/      interactive PC inference
tokenizer/      frozen 1,024-token BPE
checkpoints/    frozen model checkpoints
data/eval/      compact held-out evaluation sets
results/        recorded metrics, histories and logs
docs/           architecture, training, evaluation and hardware notes
firmware/       embedded deployment notes and firmware work
```

---

## What TinyChat-F427 is — and is not

### It is

- a fully from-scratch tiny autoregressive Transformer experiment
- a reproducible training and evaluation pipeline
- a study of language-model behavior under a severe MCU-oriented memory budget
- an open record of both successful controlled behavior and failure modes

### It is not

- a replacement for a general-purpose LLM
- a reliable unrestricted knowledge assistant
- a claim of novel MQA, RoPE, RMSNorm, or SwiGLU primitives
- a claim that the final Transformer is already running end-to-end on STM32F427

---

## Next embedded milestone

The remaining systems milestone is a faithful quantized implementation of the frozen Transformer on the STM32F427 target, followed by direct measurement of:

- Flash usage
- peak SRAM usage
- KV-cache footprint
- per-token latency
- tokens per second
- behavioral retention after quantization

Until that measurement is complete, MCU-fit calculations should be treated as design targets rather than final deployment results.

---

## License

Source code is released under the MIT License.

The upstream datasets retain their own licenses and terms. Users reproducing the training pipeline are responsible for reviewing those terms, and checkpoint use should respect the licenses of the data used during training.
