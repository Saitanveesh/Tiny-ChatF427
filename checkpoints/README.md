# Checkpoints

The training archive contains the FP32 checkpoints listed in `checkpoint_manifest.json`.

Binary `.pt` files are not committed in this source-only repository snapshot. The final checkpoint is about 4.85 MB, so it can be added directly to GitHub or Git LFS by the repository owner if a ready-to-run clone is desired.

For PC inference, place the final model at:

```text
checkpoints/tinychat_f427_final_trained_fp32.pt
```

Then run:

```bash
python inference/chat_tinychat.py
```

Always verify the SHA-256 value against `checkpoint_manifest.json` before using a copied checkpoint.
