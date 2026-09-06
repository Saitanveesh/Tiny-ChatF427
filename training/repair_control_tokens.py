import json
import os

FILES = [
    "data/train_raw.jsonl",
    "data/valid_raw.jsonl",
]

CONTROL_MAP = {
    "[BOS]": "<BOS>",
    "[EOS]": "<EOS>",
    "[USER]": "<USER>",
    "[ASSISTANT]": "<ASSISTANT>",
    "[EOT]": "<EOT>",
}

def repair_text(text, category):
    # WikiText/prose must never be touched.
    if category == "prose":
        return text, 0

    lines = text.split("\n")
    changed = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Only replace a COMPLETE marker line.
        # We do not replace marker-like strings inside normal text.
        if stripped in CONTROL_MAP:
            leading = line[:len(line) - len(line.lstrip())]
            trailing = line[len(line.rstrip()):]
            lines[i] = leading + CONTROL_MAP[stripped] + trailing
            changed += 1

    return "\n".join(lines), changed


grand_records = 0
grand_markers = 0

for path in FILES:
    tmp = path + ".repairing"

    records = 0
    markers = 0

    with open(path, "r", encoding="utf-8") as fin, \
         open(tmp, "w", encoding="utf-8") as fout:

        for line in fin:
            record = json.loads(line)

            text, count = repair_text(
                record["text"],
                record.get("category", "")
            )

            record["text"] = text

            fout.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )

            records += 1
            markers += count

    # Atomic replacement only after successful completion.
    os.replace(tmp, path)

    grand_records += records
    grand_markers += markers

    print(
        f"{path}: "
        f"{records:,} records, "
        f"{markers:,} control markers restored"
    )

print()
print("TOTAL RECORDS :", f"{grand_records:,}")
print("TOTAL MARKERS :", f"{grand_markers:,}")
print("CORPUS REPAIR : PASS")
