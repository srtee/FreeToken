"""--draft-vocab: subset derivation + loader + mask invariant (pure logic).

CPU-only, deterministic, no tokenizer or checkpoint: classification runs on a
fabricated vocab, the loader on tmp_path artifacts, and the head's mask
invariant on crafted logits (docs/mtp-draft-vocab-plan.md phase 3).
"""

from __future__ import annotations

import json

import pytest
import torch

from freetoken.engine.config import EngineConfig
from freetoken.engine.spec_mtp import (
    BUILTIN_DRAFT_VOCABS,
    load_draft_vocab,
)
from scripts.draft_vocab import classify_token, select_ids


# ---------------------------------------------------------------------------
# builder: classify_token / select_ids
# ---------------------------------------------------------------------------

ASCII_TEXT = "def foo():\n    return {'k': 42}  # ok"


def test_en_code_keeps_ascii_and_whitespace_rejects_other_scripts():
    assert classify_token(ASCII_TEXT, "en-code")
    assert classify_token("line one\nline two", "en-code")  # whitespace stays
    assert not classify_token("café", "en-code")
    assert not classify_token("мир", "en-code")
    assert not classify_token("注释", "en-code")


def test_cyrillic_and_cjk_ranges_extend_ascii_not_each_other():
    assert classify_token("мир", "cyrillic")
    assert not classify_token("мир", "cjk")
    assert not classify_token("注释", "cyrillic")
    assert classify_token("注释", "cjk")
    assert classify_token("ひらがな", "cjk")
    assert classify_token(ASCII_TEXT, "cyrillic") and classify_token(ASCII_TEXT, "cjk")


def test_select_ids_is_sorted_unique_and_drops_specials():
    vocab = {"hello": 5, "world": 2, "мир": 7, "注释": 9,
             "<|endoftext|>": 6, "<|im_start|>": 3, "": 11}
    en = select_ids(vocab, "en-code", {6, 3})
    assert en == [2, 5, 11]  # sorted, unique, specials dropped by id
    cjk = select_ids(vocab, "cjk", {6, 3})
    assert cjk == [2, 5, 9, 11]  # subsets are ASCII supersets: hello/world ride along


# ---------------------------------------------------------------------------
# loader: load_draft_vocab (True = BANNED, matching the head's masked_fill)
# ---------------------------------------------------------------------------

def _write(tmp_path, payload, name="sub"):
    p = tmp_path / f"{name}.json"
    p.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    return str(p)


def test_full_and_empty_resolve_to_no_mask():
    assert load_draft_vocab("full", 16, 16, torch.device("cpu")) is None
    assert load_draft_vocab("", 16, 16, torch.device("cpu")) is None


def test_loader_builds_exact_banned_mask(tmp_path):
    spec = _write(tmp_path, {"name": "sub", "vocab_size": 8, "ids": [1, 4, 5]})
    mask = load_draft_vocab(spec, 8, 8, torch.device("cpu"))
    assert mask.dtype == torch.bool and mask.shape == (8,)
    assert mask.tolist() == [True, False, True, True, False, False, True, True]


def test_mask_width_is_logits_width_with_reserved_tail_banned(tmp_path):
    # Qwen3.6 shape: tokenizer len 248077 < config vocab_size 248320; here 16 vs 20.
    spec = _write(tmp_path, {"name": "sub", "vocab_size": 16, "ids": [1, 15]})
    mask = load_draft_vocab(spec, 16, 20, torch.device("cpu"))
    assert mask.shape == (20,) and not mask[15].item() and mask[19].item()


def test_loader_accepts_bare_id_list(tmp_path):
    spec = _write(tmp_path, [1, 2], name="bare")
    mask = load_draft_vocab(spec, 16, 16, torch.device("cpu"))
    assert mask.tolist() == [True, False, False] + [True] * 13


def test_loader_rejects_missing_file_and_names_builtins(tmp_path):
    with pytest.raises(ValueError, match=r"not a built-in subset.*en-code"):
        load_draft_vocab(str(tmp_path / "nope.json"), 16, 16, torch.device("cpu"))


def test_loader_rejects_malformed_artifacts(tmp_path):
    bad_payloads = [
        "{not json",                                    # invalid JSON
        {"name": "s", "vocab_size": 8},                 # missing ids
        {"name": "s", "vocab_size": 8, "ids": [2, 1]},  # unsorted
        {"name": "s", "vocab_size": 8, "ids": [1, 1]},  # duplicates
        {"name": "s", "vocab_size": 8, "ids": [1.5]},   # non-int id
        {"name": "s", "vocab_size": 8, "ids": [True]},  # bool is not an id
    ]
    for i, payload in enumerate(bad_payloads):
        spec = _write(tmp_path, payload, name=f"bad{i}")
        with pytest.raises(ValueError):
            load_draft_vocab(spec, 8, 8, torch.device("cpu"))


def test_loader_refuses_artifact_from_another_tokenizer(tmp_path):
    spec = _write(tmp_path, {"name": "sub", "vocab_size": 151936,
                             "ids": [151935]})
    with pytest.raises(ValueError, match=r"151936.*248077"):
        load_draft_vocab(spec, 248077, 248320, torch.device("cpu"))


def test_loader_refuses_tokenizer_longer_than_logits(tmp_path):
    spec = _write(tmp_path, {"name": "sub", "vocab_size": 320,
                             "ids": [1]})
    with pytest.raises(ValueError, match=r"exceeds"):
        load_draft_vocab(spec, 320, 300, torch.device("cpu"))


def test_builtins_constant_matches_shipped_names():
    assert BUILTIN_DRAFT_VOCABS == ("en-code", "cyrillic", "cjk")


# ---------------------------------------------------------------------------
# head invariant: -inf ban mask moves argmax inside the subset, never out
# ---------------------------------------------------------------------------

def test_masked_argmax_matches_subset_best_and_never_picks_banned(tmp_path):
    torch.manual_seed(0)
    ids = [1, 4, 5]
    spec = _write(tmp_path, {"name": "sub", "vocab_size": 8, "ids": ids})
    mask = load_draft_vocab(spec, 8, 8, torch.device("cpu"))
    logits = torch.randn(64, 8)
    masked = logits.masked_fill(mask, float("-inf"))
    picked = masked.argmax(dim=-1)
    # identical to restricting argmax to the subset then indexing back
    local = logits.index_select(1, torch.tensor(ids)).argmax(dim=-1)
    assert torch.equal(picked, torch.tensor(ids)[local])
    # a banned id holding the global max logit can never win the draft argmax
    logits[:, 2] = 10.0
    assert logits.masked_fill(mask, float("-inf")).argmax(dim=-1).ne(2).all()


# ---------------------------------------------------------------------------
# config gate
# ---------------------------------------------------------------------------

def _cfg(**kw):
    from freetoken.distributed import DistributedInfo

    base = dict(model_path="/tmp/x", tp_info=DistributedInfo(0, 1),
                dtype=torch.bfloat16)
    base.update(kw)
    return EngineConfig(**base)


def test_config_requires_spec_mtp_for_a_subset():
    with pytest.raises(ValueError, match="--draft-vocab requires --spec-mtp"):
        _cfg(draft_vocab="en-code")
    assert _cfg(draft_vocab="en-code", spec_mtp=True).draft_vocab == "en-code"
    assert _cfg().draft_vocab == "full"
