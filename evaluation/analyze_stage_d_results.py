import json
from collections import Counter

EVAL = "data/eval/stage_d_v2/eval_paraphrase.json"

data = json.load(open(EVAL, encoding="utf-8"))
counts = Counter(r["category"] for r in data)

print()
print("==============================================")
print(" STAGE-D PARAPHRASE EVALUATION COMPOSITION")
print("==============================================")

for category, n in sorted(counts.items()):
    print(f"{category:15s}: {n:4d}")

print()
print("Non-math examples :", sum(
    n for category, n in counts.items()
    if category != "math"
))
print("Math examples     :", counts["math"])
print("==============================================")
