import os
import re
import json
import sqlite3
import hashlib
import unicodedata
from collections import defaultdict

from datasets import load_dataset
from tqdm import tqdm


# ============================================================
# TinyChat-F427
# Curated corpus builder
#
# Output:
#   data/train_raw.jsonl
#   data/valid_raw.jsonl
#   data/source_stats.json
#
# No pretrained model.
# No pretrained tokenizer.
# ============================================================


DATA_DIR = "data"

TRAIN_FILE = os.path.join(
    DATA_DIR,
    "train_raw.jsonl"
)

VALID_FILE = os.path.join(
    DATA_DIR,
    "valid_raw.jsonl"
)

DB_FILE = os.path.join(
    DATA_DIR,
    "dedup.sqlite3"
)

STATS_FILE = os.path.join(
    DATA_DIR,
    "source_stats.json"
)


# ------------------------------------------------------------
# RAW CHARACTER TARGETS
#
# We deliberately build more raw text than the final
# ~30M BPE-token training target.
#
# After our own 1024-token tokenizer is trained, we will
# measure the exact token count and cut the final stream.
# ------------------------------------------------------------

TARGETS = {

    "wikitext": 55_000_000,

    "oasst1": 25_000_000,

    "dolly": 10_000_000,

    "squad": 15_000_000,
}


SPECIAL_TOKENS = [
    "<BOS>",
    "<EOS>",
    "<USER>",
    "<ASSISTANT>",
    "<EOT>",
]


os.makedirs(
    DATA_DIR,
    exist_ok=True
)


# ============================================================
# CLEAN OLD OUTPUT
# ============================================================

for path in [
    TRAIN_FILE,
    VALID_FILE,
    DB_FILE,
    STATS_FILE,
]:

    if os.path.exists(path):
        os.remove(path)


# ============================================================
# SQLITE EXACT DEDUP
#
# Keeps memory use low even if corpus gets large.
# ============================================================

db = sqlite3.connect(DB_FILE)

db.execute(
    """
    CREATE TABLE seen (
        hash TEXT PRIMARY KEY
    )
    """
)

db.commit()


# ============================================================
# STATS
# ============================================================

stats = defaultdict(
    lambda: {
        "documents": 0,
        "characters": 0,
        "duplicates": 0,
        "rejected": 0,
        "train_documents": 0,
        "valid_documents": 0,
    }
)


# ============================================================
# NORMALIZATION
# ============================================================

_control_re = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
)

_spaces_re = re.compile(
    r"[ \t]+"
)

_many_newlines_re = re.compile(
    r"\n{3,}"
)


def normalize_text(text):

    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = text.replace(
        "\r\n",
        "\n"
    ).replace(
        "\r",
        "\n"
    )

    text = _control_re.sub(
        "",
        text
    )

    text = _spaces_re.sub(
        " ",
        text
    )

    text = _many_newlines_re.sub(
        "\n\n",
        text
    )

    text = text.strip()


    # Prevent accidental collision with our control tokens.
    for token in SPECIAL_TOKENS:

        replacement = (
            "["
            +
            token[1:-1]
            +
            "]"
        )

        text = text.replace(
            token,
            replacement
        )


    return text


# ============================================================
# BASIC QUALITY FILTER
# ============================================================

def acceptable(text):

    if len(text) < 30:
        return False

    # Avoid pathological huge records.
    if len(text) > 10_000:
        return False

    printable = sum(
        1
        for c in text
        if c.isprintable() or c == "\n"
    )

    if printable / max(1, len(text)) < 0.98:
        return False

    # Reject text dominated by punctuation/noise.
    alpha = sum(
        c.isalpha()
        for c in text
    )

    if alpha < 10:
        return False

    return True


# ============================================================
# DETERMINISTIC DOCUMENT SPLIT
#
# About 2% validation.
# ============================================================

def is_validation(hash_hex):

    value = int(
        hash_hex[:8],
        16
    )

    return (
        value % 100
    ) < 2


# ============================================================
# ADD DOCUMENT
# ============================================================

train_fp = open(
    TRAIN_FILE,
    "w",
    encoding="utf-8"
)

valid_fp = open(
    VALID_FILE,
    "w",
    encoding="utf-8"
)


def add_document(
    source,
    category,
    text,
):

    text = normalize_text(
        text
    )

    if not acceptable(text):

        stats[source]["rejected"] += 1

        return False


    # Normalize whitespace for dedup hashing.
    dedup_form = " ".join(
        text.lower().split()
    )

    digest = hashlib.sha256(
        dedup_form.encode(
            "utf-8"
        )
    ).hexdigest()


    try:

        db.execute(
            "INSERT INTO seen(hash) VALUES (?)",
            (digest,)
        )

    except sqlite3.IntegrityError:

        stats[source]["duplicates"] += 1

        return False


    record = {
        "source": source,
        "category": category,
        "text": text,
    }


    encoded = json.dumps(
        record,
        ensure_ascii=False
    )


    if is_validation(digest):

        valid_fp.write(
            encoded + "\n"
        )

        stats[source][
            "valid_documents"
        ] += 1

    else:

        train_fp.write(
            encoded + "\n"
        )

        stats[source][
            "train_documents"
        ] += 1


    stats[source]["documents"] += 1

    stats[source]["characters"] += len(
        text
    )


    return True


# ============================================================
# WIKITEXT-103 RAW
#
# General prose.
# ============================================================

print()
print("==============================================")
print("SOURCE 1/4 : WikiText-103")
print("==============================================")


wiki = load_dataset(
    "Salesforce/wikitext",
    "wikitext-103-raw-v1",
    split="train",
    streaming=True,
)


# Shuffle a bounded streaming window so we do not simply
# take the beginning of WikiText.
wiki = wiki.shuffle(
    seed=427,
    buffer_size=20_000,
)


pbar = tqdm(
    total=TARGETS["wikitext"],
    unit="char"
)


for row in wiki:

    text = normalize_text(
        row.get(
            "text",
            ""
        )
    )


    # WikiText headings frequently look like:
    #
    # = Heading =
    #
    # We want prose, not heading-heavy language.
    if (
        text.startswith("=")
        and
        text.endswith("=")
    ):
        continue


    if add_document(
        "wikitext",
        "prose",
        text,
    ):

        pbar.update(
            len(text)
        )


    if (
        stats["wikitext"]["characters"]
        >=
        TARGETS["wikitext"]
    ):
        break


pbar.close()


# ============================================================
# OPENASSISTANT OASST1
#
# Build high-quality English conversation paths.
# Dataset is small enough to reconstruct trees in memory.
# ============================================================

print()
print("==============================================")
print("SOURCE 2/4 : OpenAssistant OASST1")
print("==============================================")


oasst = load_dataset(
    "OpenAssistant/oasst1",
    split="train",
)


messages = {}


children = defaultdict(
    list
)


for row in tqdm(
    oasst,
    desc="Indexing OASST1"
):

    if row.get("deleted", False):
        continue

    if row.get("lang") != "en":
        continue

    text = normalize_text(
        row.get(
            "text",
            ""
        )
    )

    if not acceptable(text):
        continue


    message_id = row[
        "message_id"
    ]

    parent_id = row.get(
        "parent_id"
    )


    messages[message_id] = {
        "message_id": message_id,
        "parent_id": parent_id,
        "role": row.get("role"),
        "text": text,
        "rank": row.get("rank"),
    }


for mid, row in messages.items():

    parent_id = row[
        "parent_id"
    ]

    if parent_id in messages:

        children[
            parent_id
        ].append(mid)


roots = [
    mid
    for mid, row in messages.items()
    if row["parent_id"] not in messages
]


def child_sort_key(mid):

    rank = messages[mid].get(
        "rank"
    )

    if rank is None:
        rank = 9999

    return (
        rank,
        mid
    )


def best_path(root_id):

    path = []

    current = root_id


    while current in messages:

        path.append(
            messages[current]
        )

        options = children.get(
            current,
            []
        )

        if not options:
            break

        current = sorted(
            options,
            key=child_sort_key
        )[0]


    return path


for root in tqdm(
    roots,
    desc="Writing OASST1"
):

    path = best_path(
        root
    )

    # Require at least USER + ASSISTANT.
    if len(path) < 2:
        continue


    formatted = [
        "<BOS>"
    ]


    valid_turns = 0


    for message in path:

        role = message[
            "role"
        ]

        text = message[
            "text"
        ]


        if role == "prompter":

            formatted.append(
                "<USER>"
            )

        elif role == "assistant":

            formatted.append(
                "<ASSISTANT>"
            )

        else:
            continue


        formatted.append(
            text
        )

        formatted.append(
            "<EOT>"
        )

        valid_turns += 1


    formatted.append(
        "<EOS>"
    )


    if valid_turns >= 2:

        dialogue = "\n".join(
            formatted
        )

        add_document(
            "oasst1",
            "dialogue",
            dialogue,
        )


    if (
        stats["oasst1"]["characters"]
        >=
        TARGETS["oasst1"]
    ):
        break


# ============================================================
# DATABRICKS DOLLY-15K
#
# Instruction / assistant responses.
# ============================================================

print()
print("==============================================")
print("SOURCE 3/4 : Dolly-15K")
print("==============================================")


dolly = load_dataset(
    "databricks/databricks-dolly-15k",
    split="train",
)


for row in tqdm(
    dolly,
    desc="Writing Dolly"
):

    instruction = normalize_text(
        row.get(
            "instruction",
            ""
        )
    )

    context = normalize_text(
        row.get(
            "context",
            ""
        )
    )

    response = normalize_text(
        row.get(
            "response",
            ""
        )
    )


    if not instruction or not response:
        continue


    user_text = instruction


    # Keep useful context, but cap it because TinyChat has
    # only a 128-token deployment context.
    if context:

        if len(context) > 1200:
            context = context[
                :1200
            ]

        user_text += (
            "\n\nContext:\n"
            +
            context
        )


    formatted = (
        "<BOS>\n"
        "<USER>\n"
        + user_text
        + "\n<EOT>\n"
        "<ASSISTANT>\n"
        + response
        + "\n<EOT>\n"
        "<EOS>"
    )


    add_document(
        "dolly",
        "instruction",
        formatted,
    )


    if (
        stats["dolly"]["characters"]
        >=
        TARGETS["dolly"]
    ):
        break


# ============================================================
# SQuAD
#
# Concise factual Q&A.
# We intentionally do not feed entire Wikipedia context
# passages because deployment context is only 128 tokens.
# ============================================================

print()
print("==============================================")
print("SOURCE 4/4 : SQuAD")
print("==============================================")


squad = load_dataset(
    "rajpurkar/squad",
    split="train",
    streaming=True,
)


squad = squad.shuffle(
    seed=428,
    buffer_size=10_000,
)


for row in tqdm(
    squad,
    desc="Writing SQuAD"
):

    question = normalize_text(
        row.get(
            "question",
            ""
        )
    )


    answers = row.get(
        "answers",
        {}
    )


    answer_list = answers.get(
        "text",
        []
    )


    if (
        not question
        or
        not answer_list
    ):
        continue


    answer = normalize_text(
        answer_list[0]
    )


    if not answer:
        continue


    formatted = (
        "<BOS>\n"
        "<USER>\n"
        + question
        + "\n<EOT>\n"
        "<ASSISTANT>\n"
        + answer
        + "\n<EOT>\n"
        "<EOS>"
    )


    add_document(
        "squad",
        "qa",
        formatted,
    )


    if (
        stats["squad"]["characters"]
        >=
        TARGETS["squad"]
    ):
        break


# ============================================================
# CLOSE + COMMIT
# ============================================================

db.commit()
db.close()

train_fp.close()
valid_fp.close()


# Convert defaultdict to normal dict.
final_stats = {
    source: dict(values)
    for source, values in stats.items()
}


with open(
    STATS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final_stats,
        f,
        indent=2
    )


# ============================================================
# REPORT
# ============================================================

print()
print("==============================================")
print(" TinyChat-F427 : RAW CORPUS COMPLETE")
print("==============================================")


total_docs = 0
total_chars = 0


for source in [
    "wikitext",
    "oasst1",
    "dolly",
    "squad",
]:

    s = stats[
        source
    ]

    total_docs += s[
        "documents"
    ]

    total_chars += s[
        "characters"
    ]


    print()
    print(source.upper())

    print(
        " documents  :",
        s["documents"]
    )

    print(
        " characters :",
        f'{s["characters"]:,}'
    )

    print(
        " train docs :",
        s["train_documents"]
    )

    print(
        " valid docs :",
        s["valid_documents"]
    )

    print(
        " duplicates :",
        s["duplicates"]
    )

    print(
        " rejected   :",
        s["rejected"]
    )


print()
print("----------------------------------------------")

print(
    "TOTAL DOCUMENTS :",
    f"{total_docs:,}"
)

print(
    "TOTAL CHARACTERS:",
    f"{total_chars:,}"
)

print()
print(
    "Train file :",
    TRAIN_FILE
)

print(
    "Valid file :",
    VALID_FILE
)

print(
    "Stats file :",
    STATS_FILE
)

print()
print("Next: train our own 1024-token byte BPE tokenizer.")
print("==============================================")
