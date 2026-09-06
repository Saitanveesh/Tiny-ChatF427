# Tokenizer

TinyChat-F427 uses a tokenizer trained from scratch for this project.

- Type: byte-level BPE
- Vocabulary size: 1,024
- Special tokens: `<BOS>`, `<EOS>`, `<USER>`, `<ASSISTANT>`, `<EOT>`
- Frozen special-token IDs: 0, 1, 2, 3, 4 respectively

Files:

- `tokenizer.json` — complete Hugging Face Tokenizers serialization
- `tinychat-vocab.json` — vocabulary
- `tinychat-merges.txt` — learned BPE merges
- `tokenizer_manifest.json` — recorded hashes/configuration

Frozen tokenizer SHA-256:

`e0ab232a332b5f1f9b55094e21c97e69350b2f8045e6d49de935a61a2dd52c85`
