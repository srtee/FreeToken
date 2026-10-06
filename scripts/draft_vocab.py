#!/usr/bin/env python3
"""Build --draft-vocab subsets from a tokenizer (docs/mtp-draft-vocab-plan.md).

Loads an HF tokenizer and classifies every non-special id by its DECODED
string — never the raw ``tokenizer.json`` vocab keys, which for byte-level
BPE are byte-mapped forms (Cyrillic surfaces as Latin-1 codepoints there).
Writes ``python/freetoken/engine/draft_vocab/<name>.json`` artifacts
(``{"name", "vocab_size", "ids"}``, sorted unique) that
``engine/spec_mtp.load_draft_vocab`` consumes.

Usage:
  python scripts/draft_vocab.py <model-or-tokenizer-path>            # all built-ins
  python scripts/draft_vocab.py <path> en-code                       # one subset
  python scripts/draft_vocab.py --inspect <path> en-code             # dry-run stats

Classification is text-only and deterministic (see classify_token); the pure
core takes no tokenizer dependency, so tests run it on a fabricated vocab.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "python" / "freetoken" / "engine" / "draft_vocab"

# ranges in codepoint order: name -> list of (lo, hi) beyond ASCII-printable
_RANGES: dict[str, list[tuple[int, int]]] = {
    "en-code": [],
    "cyrillic": [(0x0400, 0x04FF)],
    # CJK ideographs U+4E00-9FFF + ext-A U+3400-4DBF, kana U+3040-30FF,
    # CJK punctuation U+3000-303F, fullwidth forms U+FF00-FFEF
    "cjk": [(0x3000, 0x303F), (0x3040, 0x30FF), (0x3400, 0x4DBF),
            (0x4E00, 0x9FFF), (0xFF00, 0xFFEF)],
}
_ASCII = (0x20, 0x7E)
_ASCII_WS = (0x09, 0x0A, 0x0D)  # tab/LF/CR: indent + newline tokens are code text


def classify_token(text: str, subset: str) -> bool:
    """True when every codepoint is ASCII-printable, ASCII whitespace, or in
    the subset's ranges.

    A token's presence of a codepoint is what the DRAFT step gates, so a
    subset keeps any token whose full text it covers (e.g. ``cjk`` keeps
    "def foo():  # 注释").
    """
    lo, hi = _ASCII
    for ch in text:
        cp = ord(ch)
        if lo <= cp <= hi or cp in _ASCII_WS:
            continue
        if not any(rlo <= cp <= rhi for rlo, rhi in _RANGES[subset]):
            return False
    return True


def select_ids(vocab: dict[str, int], subset: str,
               special_ids: set[int]) -> list[int]:
    """Sorted ids of vocab entries whose decoded text classifies into ``subset``.

    Special tokens are excluded by ID: the trunk emitting one ends or
    structures generation; drafting it is pure waste.
    """
    return sorted(set(
        tid for tok, tid in vocab.items()
        if tid not in special_ids and classify_token(tok, subset)
    ))


def decoded_vocab(tokenizer) -> tuple[dict[str, int], set[int]]:
    """``{decoded_text: id}`` for every id + the special-token id set."""
    vocab = {}
    for tid in range(len(tokenizer)):
        text = tokenizer.decode([tid])
        vocab.setdefault(text, tid)
    special = set(tokenizer.all_special_ids)
    special.update(
        tid for tid, tok in tokenizer.added_tokens_decoder.items() if tok.special)
    return vocab, special


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tokenizer_path")
    ap.add_argument("subset", nargs="?", default=None,
                    help=f"one of {sorted(_RANGES)} (default: all)")
    ap.add_argument("--inspect", action="store_true",
                    help="print stats without writing artifacts")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    vocab, special = decoded_vocab(tokenizer)
    vocab_size = len(tokenizer)
    subsets = [args.subset] if args.subset else sorted(_RANGES)
    bad = [s for s in subsets if s not in _RANGES]
    if bad:
        ap.error(f"unknown subset(s) {bad}; choose from {sorted(_RANGES)}")

    for name in subsets:
        ids = select_ids(vocab, name, special)
        kept = len(ids)
        print(f"{name:10s} {kept:7d}/{vocab_size} ids "
              f"({100 * kept / vocab_size:.1f}%)")
        if args.inspect:
            continue
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"{name}.json"
        out.write_text(json.dumps(
            {"name": name, "vocab_size": vocab_size, "ids": ids},
            separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"           wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
