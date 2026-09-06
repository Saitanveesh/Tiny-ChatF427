# PC inference

`chat_tinychat.py` is the interactive reference client for the frozen TinyChat-F427 model.

## Required files

1. Install the repository dependencies.
2. Ensure `tokenizer/tokenizer.json` exists.
3. Place the final checkpoint at `checkpoints/tinychat_f427_final_trained_fp32.pt`.
4. Run:

```bash
python inference/chat_tinychat.py
```

The client uses greedy decoding, a 48-token generation ceiling, a two-sentence output limiter, and a deterministic router for simple integer arithmetic. The arithmetic router is deliberately separate because the neural model did not generalize arithmetic reliably.

If the checkpoint is absent, see `checkpoints/README.md` and `checkpoints/checkpoint_manifest.json` for the expected filename and SHA-256 value.
