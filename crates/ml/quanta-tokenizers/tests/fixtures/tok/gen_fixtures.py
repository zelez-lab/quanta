#!/usr/bin/env python3
"""Generate the checked-in tokenizer conformance fixtures.

Provenance contract: this script is run ONCE by a maintainer with the
pinned HF `tokenizers` version below, and the resulting artifacts and
conformance-vector files are committed as the interchange ground truth.
CI never runs Python — the committed bytes are the contract, this
script documents where they came from.

Pinned provenance:

    python3 -m pip install tokenizers==0.21.4
    python3 gen_fixtures.py

Generated with Python 3.14.3. The reference semantics live in the
pinned Rust core of `tokenizers`; any Python >= 3.9 reproduces the
outputs byte-identically (emission is `ensure_ascii` JSON with fixed
key order, and the minimal artifacts are written by the reference
library's own serializer).

Fixture inventory (scope doc §9, the wave-two slice):

  Real anchors (immutable, hash-pinned, downloaded on first run):
    - gpt2.tokenizer.json               byte-level BPE, the
      most-trapped encode path
    - bert-base-uncased.tokenizer.json  WordPiece + BertNormalizer +
      pair templates, the normalization-trapped path

  Hand-built minimal artifacts, one per model family — each built by
  the pinned reference library and saved by ITS serializer, so even
  the minimal artifacts are reference-written bytes:
    - tiny_bpe.tokenizer.json        crafted merge chains + merge-order
      tie, byte_fallback over a full <0x00>..<0xFF> byte-token block,
      TemplateProcessing single/pair, specials + one lstrip/rstrip
      added token, ByteFallback+Fuse decoder chain
    - tiny_wordpiece.tokenizer.json  BertNormalizer (strip_accents
      follows lowercase) + BertPreTokenizer + BertProcessing (the
      fixed-form ancestor tag) + WordPiece decoder,
      max_input_chars_per_word=12 (a non-default, so the boundary is
      exercised by the long-word inputs)
    - tiny_unigram.tokenizer.json    Viterbi pieces with ▁ metaspace
      convention, Nmt normalizer, Metaspace pre-tokenizer + decoder,
      unk_id fallback
    - tiny_wordlevel.tokenizer.json  Lowercase + Whitespace + plain
      lookup, non-ASCII vocab entry (café), no decoder (the
      join-with-space default path)

  Conformance vectors (vectors/<name>.vectors.json), one file per
  artifact above: every curated input is run through the reference
  with add_special_tokens both true and false, plus pair encodings,
  and the FULL encoding record is committed — ids, tokens, offsets,
  type_ids, special_tokens_mask, attention_mask, word_ids — together
  with both decode round-trips (skip_special_tokens true/false).

Offsets are committed BYTE-BASED into the original input. The Python
binding reports char offsets (verified here at generation time: a
guard asserts every reported offset is a valid char index and would
fail loudly if the binding's unit ever changed); they are converted
deterministically against the known input via a cumulative UTF-8
byte-length map, per-sequence for pair encodings (`sequence_ids`
selects which text each token's span indexes). Special tokens carry
the reference's (0, 0) placeholder span, asserted, kept verbatim.

The curated input set is chosen to hit: merge chains and merge-order
ties, unknown tokens, byte-fallback on non-vocab bytes (incl. astral
emoji → 4 fallback tokens sharing one char span), Unicode boundaries
(combining accents, NFC/NFD-sensitive inputs, CJK, emoji + ZWJ
sequences, RTL text, astral-plane codepoints, Devanagari digits),
contractions, whitespace runs and leading/trailing space, control
characters (BertNormalizer clean_text strips them — the alignment
trap), the empty string, very long words (the WordPiece
max_input_chars boundary from both sides), special tokens appearing
literally in text (with and without surrounding whitespace), and the
lstrip/rstrip added-token edges.
"""

import hashlib
import json
import os
import urllib.request

import tokenizers
from tokenizers import (
    AddedToken,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
)

PINNED = "0.21.4"
assert tokenizers.__version__ == PINNED, (
    f"fixtures must be generated with tokenizers=={PINNED} "
    f"(found {tokenizers.__version__}); the committed bytes are the "
    "provenance contract"
)

HERE = os.path.dirname(os.path.abspath(__file__))

# ── Real anchors: immutable downloads, hash-pinned ──────────────────────

ANCHORS = [
    (
        "gpt2.tokenizer.json",
        "https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json",
        "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
    ),
    (
        "bert-base-uncased.tokenizer.json",
        "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/tokenizer.json",
        "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98",
    ),
]


def fetch_anchors():
    for name, url, sha in ANCHORS:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            print(f"downloading {name} from {url}")
            data = urllib.request.urlopen(url, timeout=120).read()
            with open(path, "wb") as f:
                f.write(data)
        data = open(path, "rb").read()
        got = hashlib.sha256(data).hexdigest()
        assert got == sha, (
            f"{name}: sha256 mismatch — expected {sha}, got {got}; the "
            "anchors are immutable, delete the file only to re-download "
            "the pinned bytes"
        )
        print(f"anchor {name}: {len(data)} bytes, sha256 {got}")


# ── Minimal artifacts, one per model family ─────────────────────────────


def build_tiny_bpe():
    chars = list("helowrdabc")
    merged = [
        "he", "hel", "hell", "hello",
        "wo", "wor", "worl", "world",
        "ab", "abc",
        # A deliberate late-rank merge: in "lol", "lo" applies; in
        # "hello", the earlier hel/hell chain outranks it — the
        # merge-order tie the vectors pin.
        "lo",
    ]
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    for i in range(256):
        vocab[f"<0x{i:02X}>"] = 3 + i
    next_id = 3 + 256
    for tok in chars + merged:
        vocab[tok] = next_id
        next_id += 1
    merges = [
        ("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o"),
        ("w", "o"), ("wo", "r"), ("wor", "l"), ("worl", "d"),
        ("a", "b"), ("ab", "c"), ("l", "o"),
    ]
    tok = Tokenizer(
        models.BPE(
            vocab=vocab,
            merges=merges,
            unk_token="<unk>",
            byte_fallback=True,
            fuse_unk=False,
        )
    )
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="<s> $A",
        pair="<s> $A </s> $B:1",
        special_tokens=[("<s>", 1), ("</s>", 2)],
    )
    tok.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.Fuse()])
    tok.add_special_tokens(
        [
            AddedToken("<unk>", special=True),
            AddedToken("<s>", special=True),
            AddedToken("</s>", special=True),
        ]
    )
    # One flagged non-special added token: lstrip/rstrip swallow the
    # surrounding whitespace into the matched span.
    tok.add_tokens([AddedToken("[area]", lstrip=True, rstrip=True, normalized=False)])
    return tok


def build_tiny_wordpiece():
    entries = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
        "the", "a", "cat", "hat", "in", "on", "sat", "mat", "play", "un",
        "##s", "##ing", "##ed", "##believ", "##able", "##a",
        "!", ",", ".", "'", "don", "##'", "##t",
        "cafe", "naive", "deja", "vu", "i", "##m",
        "中", "文", "1", "2", "##0", "hello", "world",
    ]
    vocab = {t: i for i, t in enumerate(entries)}
    tok = Tokenizer(
        models.WordPiece(vocab=vocab, unk_token="[UNK]", max_input_chars_per_word=12)
    )
    tok.normalizer = normalizers.BertNormalizer(
        clean_text=True, handle_chinese_chars=True, strip_accents=None, lowercase=True
    )
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.post_processor = processors.BertProcessing(sep=("[SEP]", 3), cls=("[CLS]", 2))
    tok.decoder = decoders.WordPiece(prefix="##")
    tok.add_special_tokens(
        [
            AddedToken(t, special=True)
            for t in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        ]
    )
    return tok


def build_tiny_unigram():
    pieces = [
        ("<unk>", 0.0), ("▁", -2.0),
        ("▁hello", -4.0), ("▁world", -4.2), ("hello", -5.0), ("world", -5.2),
        ("▁the", -3.5), ("▁cat", -4.1), ("▁a", -3.0),
        ("▁play", -4.4), ("ing", -3.8), ("s", -3.2), ("ed", -3.9),
        ("h", -6.0), ("e", -5.5), ("l", -5.6), ("o", -5.4), ("w", -6.1),
        ("r", -5.8), ("d", -5.7), ("t", -5.3), ("c", -6.2), ("a", -5.1),
        ("p", -6.3), ("y", -6.4), ("n", -5.9), ("i", -5.5), ("g", -6.5),
        ("▁h", -5.0), ("▁w", -5.2),
    ]
    tok = Tokenizer(models.Unigram(vocab=pieces, unk_id=0, byte_fallback=False))
    tok.normalizer = normalizers.Nmt()
    tok.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement="▁", prepend_scheme="always"
    )
    tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    tok.add_special_tokens([AddedToken("<unk>", special=True)])
    return tok


def build_tiny_wordlevel():
    entries = [
        "<unk>", "hello", "world", "the", "cat", "sat", "on", "mat",
        "a", "i", "'", "m", "don", "t", "!", ",", ".", "café", "1", "23",
    ]
    vocab = {t: i for i, t in enumerate(entries)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens([AddedToken("<unk>", special=True)])
    return tok


# ── Curated inputs (shared across every artifact) ───────────────────────

SINGLE_INPUTS = [
    # plain ASCII, merge chains, the tie word
    "hello world",
    "The quick brown fox jumps over the lazy dog.",
    "lol hello",
    # accents / NFC-vs-NFD-sensitive
    "café",
    "naïve déjà vu",
    "Ça va?",
    # CJK
    "中文 tokenizer 文本",
    # emoji + skin tone + ZWJ family sequence
    "👍🏽 ok 👨‍👩‍👧‍👦",
    # Devanagari digits
    "१२३ digits",
    # contractions
    "I'm here",
    "don't stop",
    # whitespace runs, leading/trailing, tabs/newlines, only-spaces
    "  hello   world  ",
    "tab\tand\nnewline",
    "   ",
    # empty string
    "",
    # very long words: the WordPiece max_input_chars_per_word=12
    # boundary from both sides, plus a natural monster
    "aaaaaaaaaaaa",
    "aaaaaaaaaaaaa",
    "supercalifragilisticexpialidocious",
    # special tokens literally in text, spaced and glued
    "<s> hello </s>",
    "a<s>b",
    "[CLS] the cat [SEP]",
    "<unk> token",
    # the lstrip/rstrip added-token edge
    "x [area] y",
    # byte-fallback bait: non-vocab letters, astral emoji
    "ζ 🦀",
    # control characters (clean_text strips them — alignment trap)
    "\x00\x07 bell",
    # RTL
    "مرحبا بالعالم",
    # astral-plane letters + hieroglyph
    "𝔘𝔫𝔦code 𓀀",
]

PAIR_INPUTS = [
    ("hello world", "how are you"),
    ("the cat", "a hat"),
    ("café", "naïve"),
    ("", "hello"),
    ("𝔘 astral", "中文"),
]


# ── Offset conversion: the binding's char offsets → byte offsets ────────


def byte_map(text):
    """Cumulative char-index → byte-index map for `text`."""
    acc = [0]
    for ch in text:
        acc.append(acc[-1] + len(ch.encode("utf-8")))
    return acc


def byte_offsets(enc, text, pair):
    """Convert the binding's char offsets to byte offsets, per sequence.

    Guards: every reported offset must be a valid char index of the
    sequence it belongs to (this is what pins "the binding reports
    char offsets" — a byte-reporting binding would trip it on any
    multibyte input), and special tokens (sequence_id None) must carry
    the (0, 0) placeholder, which is kept verbatim.
    """
    maps = [byte_map(text), byte_map(pair) if pair is not None else None]
    out = []
    for (a, b), sid in zip(enc.offsets, enc.sequence_ids):
        if sid is None:
            assert (a, b) == (0, 0), (
                f"special token carries a non-placeholder span {(a, b)}"
            )
            out.append([0, 0])
            continue
        m = maps[sid]
        assert m is not None and 0 <= a <= b < len(m), (
            f"offset {(a, b)} is not a valid char span (seq {sid}, "
            f"{len(m) - 1} chars) — the binding's offset unit changed?"
        )
        out.append([m[a], m[b]])
    return out


# ── Vector emission ─────────────────────────────────────────────────────


def record(tok, text, pair, add_special):
    enc = (
        tok.encode(text, add_special_tokens=add_special)
        if pair is None
        else tok.encode(text, pair, add_special_tokens=add_special)
    )
    case = {"kind": "single" if pair is None else "pair", "text": text}
    if pair is not None:
        case["text_pair"] = pair
    case.update(
        add_special_tokens=add_special,
        ids=enc.ids,
        tokens=enc.tokens,
        offsets=byte_offsets(enc, text, pair),
        type_ids=enc.type_ids,
        special_tokens_mask=enc.special_tokens_mask,
        attention_mask=enc.attention_mask,
        word_ids=enc.word_ids,
        decoded=tok.decode(enc.ids, skip_special_tokens=True),
        decoded_raw=tok.decode(enc.ids, skip_special_tokens=False),
    )
    return case


def emit_vectors(stem):
    """Run the curated set through the COMMITTED artifact bytes."""
    artifact = f"{stem}.tokenizer.json"
    tok = Tokenizer.from_file(os.path.join(HERE, artifact))
    cases = []
    for text in SINGLE_INPUTS:
        for add_special in (True, False):
            cases.append(record(tok, text, None, add_special))
    for first, second in PAIR_INPUTS:
        for add_special in (True, False):
            cases.append(record(tok, first, second, add_special))
    head = {
        "artifact": artifact,
        "generator": "gen_fixtures.py",
        "tokenizers_version": PINNED,
        "offsets": "bytes into the original input (converted from the "
        "binding's char offsets against the known text)",
    }
    lines = [json.dumps(head, ensure_ascii=True)[:-1] + ',"cases":[']
    for i, case in enumerate(cases):
        sep = "," if i + 1 < len(cases) else ""
        lines.append(json.dumps(case, ensure_ascii=True) + sep)
    lines.append("]}")
    path = os.path.join(HERE, "vectors", f"{stem}.vectors.json")
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    size = os.path.getsize(path)
    print(f"vectors {stem}: {len(cases)} cases, {size} bytes")
    return size


def main():
    fetch_anchors()
    os.makedirs(os.path.join(HERE, "vectors"), exist_ok=True)
    builders = {
        "tiny_bpe": build_tiny_bpe,
        "tiny_wordpiece": build_tiny_wordpiece,
        "tiny_unigram": build_tiny_unigram,
        "tiny_wordlevel": build_tiny_wordlevel,
    }
    total = 0
    for stem, build in builders.items():
        path = os.path.join(HERE, f"{stem}.tokenizer.json")
        build().save(path, pretty=True)
        size = os.path.getsize(path)
        total += size
        print(f"artifact {stem}: {size} bytes (reference-written)")
    for name, _, _ in ANCHORS:
        total += os.path.getsize(os.path.join(HERE, name))
    for stem in list(builders) + [name[: -len(".tokenizer.json")] for name, _, _ in ANCHORS]:
        total += emit_vectors(stem)
    print(f"total committed fixture bytes: {total}")


if __name__ == "__main__":
    main()
