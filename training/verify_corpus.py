import json
from collections import Counter

FILES = [
    "data/train_raw.jsonl",
    "data/valid_raw.jsonl",
]

GOOD = [
    "<BOS>",
    "<EOS>",
    "<USER>",
    "<ASSISTANT>",
    "<EOT>",
]

BAD = [
    "[BOS]",
    "[EOS]",
    "[USER]",
    "[ASSISTANT]",
    "[EOT]",
]

sources = Counter()
categories = Counter()
markers = Counter()
bad_markers = Counter()

documents = 0

for path in FILES:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)

            documents += 1
            sources[r["source"]] += 1
            categories[r["category"]] += 1

            lines = {
                x.strip()
                for x in r["text"].splitlines()
            }

            for token in GOOD:
                if token in lines:
                    markers[token] += 1

            for token in BAD:
                if token in lines:
                    bad_markers[token] += 1


print("==============================================")
print(" TinyChat-F427 : CORPUS INTEGRITY CHECK")
print("==============================================")
print()
print("Documents :", f"{documents:,}")
print()
print("Sources:")
for k, v in sources.items():
    print(f"  {k:10s}: {v:,}")

print()
print("Categories:")
for k, v in categories.items():
    print(f"  {k:12s}: {v:,}")

print()
print("Documents containing control tokens:")
for token in GOOD:
    print(f"  {token:12s}: {markers[token]:,}")

print()
print("BAD marker lines remaining:")
for token in BAD:
    print(f"  {token:12s}: {bad_markers[token]:,}")

assert documents == 191998, (
    f"Unexpected document count: {documents}"
)

assert markers["<USER>"] > 0
assert markers["<ASSISTANT>"] > 0
assert markers["<BOS>"] > 0
assert markers["<EOS>"] > 0
assert markers["<EOT>"] > 0

assert sum(bad_markers.values()) == 0, (
    f"Bad markers remain: {bad_markers}"
)

print()
print("CORPUS INTEGRITY : PASS")
print("==============================================")
