# MTP draft-vocab restriction — plan

Status: approved plan, 2026-10-06. Target: restrict the MTP draft head's proposals to a
vocab subset (`--draft-vocab`), Strata-style, adapted to our shared-head architecture.
All paths relative to the repo root.

## TL;DR

The drafter (`engine/spec_mtp.py:MTPDrafter.draft`) takes its argmax over the full
vocabulary. Restricting proposals to a language subset (English/code, Cyrillic, CJK)
cannot change outputs — greedy verify is lossless by construction — it only moves the
acceptance rate. Strata measured +1–2% on English/code with a 40K-id subset and 15–38%
on CJK by *widening* a too-narrow default; the risk side is acceptance collapse on
out-of-subset scripts. Default stays `full`; subsets are opt-in and derived from the
tokenizer alone.

## Why our version differs from Strata's

- Strata owns a *dedicated* draft head (~180 MiB), so restricting vocab shrinks VRAM.
  Our MTP heads share the trunk's lm_head by reference (`set_lm_head`,
  `mtp_use_dedicated_embeddings: false`; both `qwen3_5_moe/mtp.py` and
  `qwen4_exp/mtp.py`) and the trunk needs the full head for its own logits. A sliced
  copy would *add* VRAM (S x H bf16), the wrong trade on sm_70. So v1 is mask-only:
  the head GEMM stays full-vocab; the win is acceptance, not VRAM.
- Their "head does not fit" startup negotiation (#474) has no analogue here — nothing
  new has to fit.
- Vocab sizes differ per family (Qwen3.6: 248,320). Subsets are artifacts keyed by
  `vocab_size` and refused on mismatch.

## Design

One implementation site, family-agnostic: the **head's** `draft_step`. All three
draft paths — the scheduler's eager loop, the graphed draft capture, and the
eager reference — end at `head.draft_step(carry, tokens) -> (carry, logits)`
followed by an argmax *outside* the head, so the head applies a persistent
`-inf` bool mask over the full-vocab logits:

```python
carry, logits = self.forward_rows(carry, input_ids, with_logits=True)
if self.draft_vocab_mask is not None:
    logits = logits.masked_fill(self.draft_vocab_mask, float("-inf"))
return carry, logits
```

- `draft_vocab_mask: torch.Tensor | None` defaults to `None` in both MTPHead
  `__init__`s (`qwen3_5_moe`, `qwen4_exp`); `engine/spec_mtp.load_draft_vocab`
  builds the tensor and `MTPDrafter.__init__` installs it on the head BEFORE any
  graph capture — fixed shape, persistent buffer -> capture-safe, matching the
  graphed draft/verify path that `--spec-mtp` uses. (Masking inside
  `MTPDrafter.draft` would be a no-op: the scheduler calls the head directly and
  the graphed path never routes through the drafter.)
- `draft_vocab == "full"` keeps `masked_fill` out of the hot path entirely
  (`mask is None` guard) — bit-for-bit today's behavior.

### Subset derivation (v1: tokenizer-only, deterministic)

`scripts/draft_vocab.py` reads a tokenizer and classifies each non-special id by
its decoded string:
- `en-code`: every char ASCII-printable (0x20–0x7E) or ASCII whitespace
  (tab/LF/CR — indent and newline tokens are code text). Covers code, JSON,
  most prompts.
- `cyrillic`: ASCII-printable allowed plus U+0400–U+04FF.
- `cjk`: ASCII-printable allowed plus CJK ideographs U+4E00–U+9FFF, ext-A
  U+3400–4DBF, kana U+3040–U+30FF, and CJK/fullwidth punctuation
  U+3000–U+303F, U+FF00–U+FFEF.
- Added special tokens (`<|...|>`) are excluded: the trunk emitting one ends or
  structures generation; drafting it is pure waste.

Classification runs on the token's DECODED string, never the raw `tokenizer.json`
vocab keys: byte-level BPE stores byte-mapped forms there (Cyrillic surfaces as
Latin-1 codepoints), so key-based ranges silently miss every non-ASCII token —
discovered while building the first artifacts for the Qwen3.6 tokenizer.

Output: `python/freetoken/engine/draft_vocab/<name>.json` —
`{"name", "vocab_size", "ids"}` (sorted, unique). Checked in as package data
(pyproject `package-data` gains `engine/draft_vocab/*.json`, precedent:
`moe/configs/**/*.json`). Regenerate per tokenizer with the script when a family's
vocab differs; the loader's `vocab_size` guard catches a stale artifact.

## CLI / config

- `EngineConfig.draft_vocab: str = "full"` (next to `spec_mtp`).
- `--draft-vocab {full,en-code,cyrillic,cjk,<path.json>}` next to `--spec-draft-n`
  in `server/args.py`.
- Validation:
  - `__post_init__`: `draft_vocab != "full"` without `spec_mtp` -> ValueError
    (no silent no-op, mirrors the `--spec-draft-n` gate).
  - Loader: named subset must exist; `ids` sorted, unique, `max < vocab_size`
    (error names the flag, the artifact's `vocab_size`, and the model's).
- `docs/cli.md` gains the flag next to the spec section.

## Phases

1. `scripts/draft_vocab.py` — builder + `--inspect`; pure core
   (`classify_token(text) -> bool`, `select_ids(vocab, pred)`) so tests need no
   tokenizer. Generate and commit the three subsets for the Qwen3.6 tokenizer
   (248,320).
2. Engine wiring — `engine/config.py` field + gate, `server/args.py` flag,
   `load_draft_vocab()` + `MTPDrafter.__init__` mask install in
   `engine/spec_mtp.py`, the head-level `masked_fill` in both families'
   `draft_step`.
3. Tests — `tests/engine/test_draft_vocab.py`, pure-logic per the
   `test_spec_resolve.py` convention: builder classification and id selection on a
   fabricated vocab; loader errors (missing file, unsorted, dupes, out-of-range);
   drafter invariant (global argmax outside the subset -> best in-subset token;
   `full` path unchanged); losslessness note — the reject path for out-of-subset
   trunk tokens is the existing chain resolve, already covered.
4. Bench + docs — Qwen3.6-35B on the 5070 Ti: English and CJK prompts, `full` vs
   `en-code`; record tok/s and per-position acceptance (SpecStats) into
   `docs/tcq-baseline-numbers.md` style numbers. This decides whether any subset
   ever becomes a default (expected: no).

## Acceptance criteria

- `--draft-vocab full` (default): behavior and outputs identical to today.
- With a subset: every drafted token is in the subset; emitted sequences remain
  byte-identical to plain greedy decode (trunk-decided).
- Out-of-range or malformed artifacts fail fast at startup with an actionable error.

## Non-goals

Sliced-head copy (VRAM-for-draft-latency trade; revisit only if the head GEMM shows
in profiles), prompt-lookup drafting, automatic language detection, per-request
subset switching.

## Reference

Strata (`~/20llms/Strata`): `docs/DETAILS.md` (subsets, 15–38% CJK, #474
negotiation), `tools/draft_vocab.py`, `data/draft_vocab*.bin`, `mtp/rt/draft_vocab.bin`.
