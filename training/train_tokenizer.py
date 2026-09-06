import os
import json
import hashlib
from tokenizers import Tokenizer, AddedToken
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


# ============================================================
# TinyChat-F427
# 1024-token byte-level BPE tokenizer
#
# IMPORTANT:
# - trained ONLY on training split
# - no pretrained vocabulary
# - all 256 byte values available
# - exactly 1024 total tokens
# ============================================================

TRAIN_FILE = "data/train_raw.jsonl"

OUT_DIR = "tokenizer"
TOKENIZER_JSON = os.path.join(
    OUT_DIR,
    "tokenizer.json"
)

VOCAB_SIZE = 1024

SPECIAL_TOKENS = [
    "<BOS>",
    "<EOS>",
    "<USER>",
    "<ASSISTANT>",
    "<EOT>",
]

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


# ============================================================
# BUILD TOKENIZER
# ============================================================

tokenizer = Tokenizer(
    BPE(
        dropout=None,
        unk_token=None,
        fuse_unk=False,
        byte_fallback=False,
    )
)


# ByteLevel converts arbitrary UTF-8 input into a reversible
# representation backed by all 256 byte values.
tokenizer.pre_tokenizer = ByteLevel(
    add_prefix_space=False,
    use_regex=True,
)

tokenizer.decoder = ByteLevelDecoder()


special_added_tokens = [
    AddedToken(
        token,
        single_word=False,
        lstrip=False,
        rstrip=False,
        normalized=False,
        special=True,
    )
    for token in SPECIAL_TOKENS
]


trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    show_progress=True,
    special_tokens=special_added_tokens,

    # Critical:
    # Forces the initial alphabet to contain all 256 possible
    # byte symbols before BPE merges are learned.
    initial_alphabet=ByteLevel.alphabet(),
)


# ============================================================
# STREAM TEXT FROM JSONL
# ============================================================

def text_iterator():

    count = 0

    with open(
        TRAIN_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            record = json.loads(line)

            text = record.get(
                "text",
                ""
            )

            if text:
                count += 1
                yield text


print()
print("==============================================")
print(" TinyChat-F427 : TOKENIZER TRAINING")
print("==============================================")
print()
print("Vocabulary target :", VOCAB_SIZE)
print("Training source   :", TRAIN_FILE)
print("Base alphabet     : 256 byte symbols")
print("Special tokens    :", len(SPECIAL_TOKENS))
print()
print("Training tokenizer...")
print()


tokenizer.train_from_iterator(
    text_iterator(),
    trainer=trainer,
)


# ============================================================
# VERIFY VOCABULARY SIZE
# ============================================================

vocab_size = tokenizer.get_vocab_size(
    with_added_tokens=True
)

print()
print("Tokenizer training finished.")
print()
print("Vocabulary size :", vocab_size)
print("Expected        :", VOCAB_SIZE)

if vocab_size != VOCAB_SIZE:

    raise RuntimeError(
        "TOKENIZER SIZE FAILURE: "
        f"expected {VOCAB_SIZE}, got {vocab_size}"
    )


# ============================================================
# VERIFY SPECIAL TOKENS
# ============================================================

print()
print("Special token IDs:")

special_ids = {}

for token in SPECIAL_TOKENS:

    token_id = tokenizer.token_to_id(
        token
    )

    if token_id is None:
        raise RuntimeError(
            f"Missing special token: {token}"
        )

    special_ids[token] = token_id

    print(
        f"  {token:12s} -> {token_id}"
    )


# They should occupy the first five IDs.
expected_special_ids = {
    "<BOS>": 0,
    "<EOS>": 1,
    "<USER>": 2,
    "<ASSISTANT>": 3,
    "<EOT>": 4,
}

if special_ids != expected_special_ids:

    raise RuntimeError(
        "Unexpected special-token IDs.\n"
        f"Expected: {expected_special_ids}\n"
        f"Actual  : {special_ids}"
    )


# ============================================================
# SAVE
# ============================================================

tokenizer.save(
    TOKENIZER_JSON
)

# Also export traditional BPE artifacts.
saved_model_files = tokenizer.model.save(
    OUT_DIR,
    "tinychat"
)


# ============================================================
# ROUND-TRIP TESTS
# ============================================================

TEST_STRINGS = [
    "hello",
    "Hello, how are you?",
    "This is TinyChat-F427.",
    "2 + 2 = 4",
    "The quick brown fox jumps over the lazy dog.",
    "email@example.com",
    "STM32F427 @ 168 MHz",
    "don't can't won't",
    "₹250",
    "café",
    "తెలుగు",
    "日本語",
    "🙂",
    "line one\nline two",
    "\tspaces\tand\ttabs",
    "symbols: !@#$%^&*()[]{}<>",
]


print()
print("==============================================")
print(" ROUND-TRIP TESTS")
print("==============================================")


for text in TEST_STRINGS:

    encoding = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    decoded = tokenizer.decode(
        encoding.ids,
        skip_special_tokens=False,
    )

    passed = (
        decoded == text
    )

    print()
    print(
        "INPUT :",
        repr(text)
    )

    print(
        "TOKENS:",
        len(encoding.ids)
    )

    print(
        "IDS   :",
        encoding.ids[:20],
        "..." if len(encoding.ids) > 20 else ""
    )

    print(
        "OUTPUT:",
        repr(decoded)
    )

    print(
        "RESULT:",
        "PASS" if passed else "FAIL"
    )

    if not passed:

        raise RuntimeError(
            "ROUND-TRIP FAILURE"
        )


# ============================================================
# FULL BYTE TEST
#
# Test all byte values that are valid through UTF-8-safe
# Latin-1 mapping independently.
# ============================================================

print()
print("==============================================")
print(" BYTE COVERAGE TEST")
print("==============================================")


# Every raw byte must correspond to an initial ByteLevel
# alphabet symbol. This verifies the 256-symbol foundation.
alphabet = set(
    ByteLevel.alphabet()
)

if len(alphabet) != 256:

    raise RuntimeError(
        f"Byte alphabet size is {len(alphabet)}, expected 256"
    )

vocab = tokenizer.get_vocab()

missing_alphabet = [
    symbol
    for symbol in alphabet
    if symbol not in vocab
]

print(
    "Byte alphabet symbols :",
    len(alphabet)
)

print(
    "Missing from vocab    :",
    len(missing_alphabet)
)

if missing_alphabet:

    raise RuntimeError(
        "BYTE COVERAGE FAILURE"
    )


# ============================================================
# CORPUS COMPRESSION SAMPLE
# ============================================================

print()
print("==============================================")
print(" TOKENIZATION SAMPLE STATISTICS")
print("==============================================")


sample_chars = 0
sample_bytes = 0
sample_tokens = 0
sample_docs = 0

MAX_SAMPLE_DOCS = 10_000


with open(
    TRAIN_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        record = json.loads(line)
        text = record["text"]

        encoding = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        sample_chars += len(text)
        sample_bytes += len(
            text.encode("utf-8")
        )
        sample_tokens += len(
            encoding.ids
        )

        sample_docs += 1

        if sample_docs >= MAX_SAMPLE_DOCS:
            break


chars_per_token = (
    sample_chars / sample_tokens
)

bytes_per_token = (
    sample_bytes / sample_tokens
)


print(
    "Documents sampled :",
    f"{sample_docs:,}"
)

print(
    "Characters        :",
    f"{sample_chars:,}"
)

print(
    "UTF-8 bytes       :",
    f"{sample_bytes:,}"
)

print(
    "BPE tokens        :",
    f"{sample_tokens:,}"
)

print(
    "Characters/token  :",
    f"{chars_per_token:.3f}"
)

print(
    "Bytes/token       :",
    f"{bytes_per_token:.3f}"
)


# ============================================================
# HASH ARTIFACTS
# ============================================================

print()
print("==============================================")
print(" TOKENIZER ARTIFACTS")
print("==============================================")


artifact_paths = [
    TOKENIZER_JSON,
    *saved_model_files,
]


hashes = {}


for path in artifact_paths:

    with open(
        path,
        "rb"
    ) as f:

        digest = hashlib.sha256(
            f.read()
        ).hexdigest()

    hashes[path] = digest

    print()
    print(path)
    print("SHA256:", digest)


manifest = {
    "project": "TinyChat-F427",
    "vocab_size": VOCAB_SIZE,
    "special_tokens": special_ids,
    "byte_alphabet_size": 256,
    "sample_documents": sample_docs,
    "sample_characters": sample_chars,
    "sample_bytes": sample_bytes,
    "sample_tokens": sample_tokens,
    "characters_per_token": chars_per_token,
    "bytes_per_token": bytes_per_token,
    "sha256": hashes,
}


with open(
    os.path.join(
        OUT_DIR,
        "tokenizer_manifest.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        manifest,
        f,
        indent=2,
        ensure_ascii=False,
    )


print()
print("==============================================")
print(" TOKENIZER VERIFICATION")
print("==============================================")
print()
print("Vocabulary          : 1024 / PASS")
print("Byte alphabet       : 256 / PASS")
print("Special tokens      : 5 / PASS")
print("Special token IDs   : PASS")
print("Round-trip tests    : PASS")
print("Tokenizer trained   : FROM SCRATCH")
print()
print("TOKENIZER LOCK      : PASS")
print("==============================================")
