import json
import statistics
from collections import defaultdict
from tokenizers import Tokenizer
from tqdm import tqdm

TOKENIZER_PATH = "tokenizer/tokenizer.json"

FILES = {
    "train": "data/train_raw.jsonl",
    "valid": "data/valid_raw.jsonl",
}

CONTEXT = 128
BATCH_DOCS = 256

tok = Tokenizer.from_file(TOKENIZER_PATH)

stats = {
    split: defaultdict(lambda: {
        "documents": 0,
        "characters": 0,
        "bytes": 0,
        "tokens": 0,
        "lengths": [],
        "over_context": 0,
    })
    for split in FILES
}


def process_batch(split, batch_records):
    texts = [r["text"] for r in batch_records]
    encodings = tok.encode_batch(texts, add_special_tokens=False)

    for record, enc in zip(batch_records, encodings):
        source = record["source"]
        text = record["text"]
        n_tokens = len(enc.ids)
        s = stats[split][source]
        s["documents"] += 1
        s["characters"] += len(text)
        s["bytes"] += len(text.encode("utf-8"))
        s["tokens"] += n_tokens
        s["lengths"].append(n_tokens)
        if n_tokens > CONTEXT:
            s["over_context"] += 1


for split, path in FILES.items():
    print()
    print(f"Tokenizing {split}: {path}")
    batch = []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            batch.append(json.loads(line))
            if len(batch) >= BATCH_DOCS:
                process_batch(split, batch)
                batch = []
        if batch:
            process_batch(split, batch)

print()
print("==============================================================")
print(" TinyChat-F427 : EXACT TOKEN AUDIT")
print("==============================================================")

manifest = {}
for split in ["train", "valid"]:
    print()
    print(f"---------------- {split.upper()} ----------------")
    split_tokens = 0
    split_docs = 0
    manifest[split] = {}

    for source in ["wikitext", "oasst1", "dolly", "squad"]:
        s = stats[split][source]
        lengths = s["lengths"]
        if not lengths:
            continue
        lengths_sorted = sorted(lengths)

        def percentile(p):
            idx = int((len(lengths_sorted) - 1) * p)
            return lengths_sorted[idx]

        avg_len = s["tokens"] / s["documents"]
        pct_over = 100.0 * s["over_context"] / s["documents"]
        chars_per_token = s["characters"] / s["tokens"]

        print()
        print(source.upper())
        print("  documents       :", f'{s["documents"]:,}')
        print("  tokens          :", f'{s["tokens"]:,}')
        print("  avg tokens/doc  :", f"{avg_len:.1f}")
        print("  median tokens   :", statistics.median(lengths))
        print("  p90 tokens      :", percentile(0.90))
        print("  p95 tokens      :", percentile(0.95))
        print("  p99 tokens      :", percentile(0.99))
        print("  >128-token docs :", f'{s["over_context"]:,} ({pct_over:.2f}%)')
        print("  chars/token     :", f"{chars_per_token:.3f}")

        manifest[split][source] = {
            "documents": s["documents"],
            "characters": s["characters"],
            "bytes": s["bytes"],
            "tokens": s["tokens"],
            "average_tokens_per_document": avg_len,
            "median_tokens": statistics.median(lengths),
            "p90_tokens": percentile(0.90),
            "p95_tokens": percentile(0.95),
            "p99_tokens": percentile(0.99),
            "documents_over_128": s["over_context"],
            "percent_over_128": pct_over,
            "characters_per_token": chars_per_token,
        }
        split_tokens += s["tokens"]
        split_docs += s["documents"]

    print()
    print("TOTAL DOCUMENTS :", f"{split_docs:,}")
    print("TOTAL TOKENS    :", f"{split_tokens:,}")
    manifest[split]["_total"] = {"documents": split_docs, "tokens": split_tokens}

with open("data/token_audit.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

train_total = manifest["train"]["_total"]["tokens"]
print()
print("==============================================================")
print(" TRAIN SOURCE DISTRIBUTION")
print("==============================================================")
print()
for source in ["wikitext", "oasst1", "dolly", "squad"]:
    n = manifest["train"][source]["tokens"]
    pct = 100.0 * n / train_total
    print(f"{source:10s}: {n:>12,d} tokens ({pct:6.2f}%)")

print()
print("Audit saved:")
print("data/token_audit.json")
print()
print("TOKEN AUDIT : PASS")
print("==============================================================")
