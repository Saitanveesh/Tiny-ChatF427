import os
import json
import hashlib
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

TOKENIZER_PATH = "tokenizer/tokenizer.json"
INPUT_FILES = {"train": "data/train_raw.jsonl", "valid": "data/valid_raw.jsonl"}
SOURCES = ["wikitext", "oasst1", "dolly", "squad"]
OUT_DIR = Path("data/bin")
OUT_DIR.mkdir(parents=True, exist_ok=True)
BATCH_DOCS = 256

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
VOCAB_SIZE = tokenizer.get_vocab_size(with_added_tokens=True)
BOS = tokenizer.token_to_id("<BOS>")
EOS = tokenizer.token_to_id("<EOS>")
USER = tokenizer.token_to_id("<USER>")
ASSISTANT = tokenizer.token_to_id("<ASSISTANT>")
EOT = tokenizer.token_to_id("<EOT>")
assert VOCAB_SIZE == 1024
assert [BOS, EOS, USER, ASSISTANT, EOT] == [0, 1, 2, 3, 4]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


manifest = {
    "project": "TinyChat-F427",
    "format": "little-endian uint16 token IDs",
    "vocab_size": VOCAB_SIZE,
    "special_tokens": {"<BOS>": BOS, "<EOS>": EOS, "<USER>": USER, "<ASSISTANT>": ASSISTANT, "<EOT>": EOT},
    "splits": {},
}

for split, input_path in INPUT_FILES.items():
    print(f"\nPACKING {split.upper()}")
    handles = {}
    split_stats = {}
    for source in SOURCES:
        output_path = OUT_DIR / f"{split}_{source}.bin"
        handles[source] = open(output_path, "wb")
        split_stats[source] = {"documents": 0, "tokens": 0, "bos_added": 0, "eos_added": 0}

    batch = []

    def flush_batch(records):
        if not records:
            return
        texts = [r["text"] for r in records]
        encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
        for record, encoding in zip(records, encodings):
            source = record["source"]
            if source not in handles:
                raise RuntimeError(f"Unknown source: {source}")
            ids = list(encoding.ids)
            if not ids:
                continue
            if ids[0] != BOS:
                ids.insert(0, BOS)
                split_stats[source]["bos_added"] += 1
            if ids[-1] != EOS:
                ids.append(EOS)
                split_stats[source]["eos_added"] += 1
            array = np.asarray(ids, dtype="<u2")
            if len(array) and int(array.max()) >= VOCAB_SIZE:
                raise RuntimeError("TOKEN ID OUT OF RANGE")
            handles[source].write(array.tobytes())
            split_stats[source]["documents"] += 1
            split_stats[source]["tokens"] += len(array)

    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=split):
            batch.append(json.loads(line))
            if len(batch) >= BATCH_DOCS:
                flush_batch(batch)
                batch = []
        flush_batch(batch)

    for handle in handles.values():
        handle.close()

    manifest["splits"][split] = {}
    total_tokens = 0
    total_docs = 0
    for source in SOURCES:
        path = OUT_DIR / f"{split}_{source}.bin"
        size = os.path.getsize(path)
        s = split_stats[source]
        expected_bytes = s["tokens"] * 2
        if size != expected_bytes:
            raise RuntimeError(f"SIZE FAILURE: {path} {size} != {expected_bytes}")
        digest = sha256_file(path)
        print(f"{source:10s} docs={s['documents']:>7,d} tokens={s['tokens']:>12,d} size={size/(1024*1024):7.2f} MiB")
        print(f"           BOS added={s['bos_added']:,} EOS added={s['eos_added']:,}")
        print(f"           SHA256={digest}")
        manifest["splits"][split][source] = {**s, "bytes": size, "sha256": digest, "file": str(path)}
        total_tokens += s["tokens"]
        total_docs += s["documents"]
    manifest["splits"][split]["_total"] = {"documents": total_docs, "tokens": total_tokens, "bytes": total_tokens * 2}

manifest["tokenizer_sha256"] = sha256_file(TOKENIZER_PATH)
manifest["training_plan"] = {
    "phase_A": {
        "name": "language_foundation",
        "target_tokens": 24_000_000,
        "sources": {"wikitext": 16_000_000, "oasst1": 2_000_000, "dolly": 3_000_000, "squad": 3_000_000},
    },
    "phase_B": {
        "name": "conversation_specialization",
        "target_tokens": 6_000_000,
        "sources": {"wikitext": 0, "oasst1": 2_500_000, "dolly": 2_500_000, "squad": 1_000_000},
    },
}

with open(OUT_DIR / "packed_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("\nPACKED CORPUS LOCK  : PASS")
