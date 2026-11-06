# Getting started

This guide separates **working with the PC reference code** from **running the trained model** and from **STM32 deployment**. They are not interchangeable.

## Requirements

- Python 3.10 or newer.
- A computer with sufficient disk space for PyTorch and dependencies.
- Internet access for installing dependencies.
- **No microcontroller required** for the PC workflow.

Run the commands from the repository root. Python 3.10 is the recorded development version; other compatible versions are not independently qualified here.

## 1. Install

```bash
git clone https://github.com/Saitanveesh/Tiny-ChatF427.git
cd Tiny-ChatF427
python -m venv .venv
```

Activate your environment:

| Platform | Command |
|---|---|
| Linux / macOS | `source .venv/bin/activate` |
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `.venv\Scripts\activate.bat` |

Then:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

**PowerShell script execution blocked?** Use Command Prompt activation instead, or invoke `.venv\Scripts\python.exe` directly. You do not have to weaken your machine-wide PowerShell security policy.

## 2. Run the architecture check (no weights required)

```bash
python -c "from tinychat.model import TinyChatF427, count_parameters; print(count_parameters(TinyChatF427()))"
```

Expected output:

```text
1206864
```

This instantiates a randomly initialized architecture, not a trained assistant. For more detailed CPU training benchmarks, run `python training/benchmark_architecture.py`; it runs several forward/backward training steps and is **not** a lightweight smoke test.

## 3. Run PC chat (requires a separate artifact)

The tokenizer is included at `tokenizer/tokenizer.json`. The trained model weights are **not** included in the current public clone.

To run the historical Stage-F client:

1. Obtain `tinychat_f427_final_trained_fp32.pt` from the original training archive or project maintainer; no public direct-download URL is supplied.
2. Copy it to `checkpoints/tinychat_f427_final_trained_fp32.pt`.
3. Compare its SHA-256 with [the manifest](../checkpoints/checkpoint_manifest.json).
4. Run `python inference/chat_tinychat.py`.

For checksum verification:

```bash
# Linux / macOS
sha256sum checkpoints/tinychat_f427_final_trained_fp32.pt

# Windows PowerShell
Get-FileHash checkpoints/tinychat_f427_final_trained_fp32.pt -Algorithm SHA256
```

Expected SHA-256 for that particular checkpoint:

```text
3e34e93bb58680df6e0269659ed90ad553740122a74963dfea83706aad3e4768
```

The interactive script is CPU-only and uses greedy decoding. Simple binary arithmetic answers come from a Python arithmetic router, not the Transformer.

## 4. Explore the research

- [Model specification](ARCHITECTURE.md)
- [Training history](TRAINING.md)
- [Recorded evaluations and limitations](EVALUATION.md)
- [Hardware memory budget](HARDWARE_TARGET.md)
- [What's available for reproduction](REPRODUCIBILITY.md)

## Common issues

| Symptom | Explanation / remedy |
|---|---|
| `ModuleNotFoundError: tinychat` | Run `python -m pip install -e .` inside the activated environment. |
| Missing checkpoint when launching chat | Expected for a new clone. Supply and verify the final checkpoint first. |
| Large dependency download | PyTorch and dataset tooling are substantial; the model itself is only ~1.2M parameters. |
| `benchmark_architecture.py` takes time | It benchmarks training batches, not just parameter counting. Use the short `python -c` command above for a quick check. |
| Expecting an STM32 firmware binary | Final Transformer MCU firmware is an open milestone; see [firmware notes](../firmware/README.md). |

## Hardware safety

The STM32/Pixhawk target is a development and experimentation platform in this project. **Do not flash research firmware onto a flight controller used for active flight** without independent verification and a dedicated test setup.
