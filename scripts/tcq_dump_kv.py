"""Dump REAL prefill K/V from the 32B f16 engine to disk.

Runs the offline LLM with a patched MHAKVCache.store_kv that captures the
per-layer k/v rows written during the prefill of the divergence prompts
(soak P1/P2) plus a control prompt (soak P0). Saves per-layer tensors for
the first N layers (or all) as .pt shards under /tmp/tcq_kv_dump/.

Usage: .venv/bin/python scripts/tcq_dump_kv.py [--layers 0,1,2] [--max-tokens 8]
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch

from freetoken.kvcache.mha_pool import MHAKVCache
from freetoken.llm.llm import LLM
from freetoken.core import SamplingParams

OUT = "/tmp/tcq_kv_dump"
os.makedirs(OUT, exist_ok=True)

PROMPTS = {
    # control: turbo3 encode is fine on this one
    "P0": "Write a python function that deduplicates a list while preserving order.",
    # diverge from first decode token under turbo3_tcq
    "P1": "Explain what this does: print([x*x for x in range(10) if x%2])",
    "P2": "Refactor: def f(l):\n s=0\n for x in l:\n  s+=x\n return s",
}

_dump_state: dict = {"active": False, "tag": None}
_captured: dict = {}


_orig_store_kv = MHAKVCache.store_kv


def _patched_store_kv(self, k, v, out_loc, layer_id):
    if os.environ.get("TCQ_DEBUG"):
        print(f"store_kv layer={layer_id} active={_dump_state['active']}", flush=True)
    if _dump_state["active"]:
        slot = _captured.setdefault(layer_id, {"k": [], "v": [], "loc": []})
        slot["k"].append(k.detach().clone())
        slot["v"].append(v.detach().clone())
        slot["loc"].append(out_loc.detach().clone())
    return _orig_store_kv(self, k, v, out_loc, layer_id)


MHAKVCache.store_kv = _patched_store_kv


def main() -> None:
    layers_arg = os.environ.get("DUMP_LAYERS", "")  # "" = all
    want = {int(x) for x in layers_arg.split(",") if x} or None

    model = os.environ.get(
        "SOAK_MODEL",
        "/home/sherntee/.cache/huggingface/hub/models--bartowski--"
        "Qwen2.5-Coder-32B-Instruct-GGUF/snapshots/40b525506a4f98ed425882fa6"
        "dfc90cc8139065e/Qwen2.5-Coder-32B-Instruct-IQ3_XXS.gguf",
    )
    llm = LLM(model, dtype=torch.bfloat16, num_page_override=1024,
              cuda_graph_max_bs=0)

    print("pool type:", type(llm.engine.kv_cache).__name__,
              "store_kv is patched:", MHAKVCache.store_kv is not _orig_store_kv)
    for name, prompt in PROMPTS.items():
        _captured.clear()
        _dump_state["active"] = True
        try:
            llm.generate(
                [prompt],
                SamplingParams(max_tokens=1, temperature=0.0),  # prefill only
            )
        finally:
            _dump_state["active"] = False
        n_layers = len(_captured)
        for layer_id, slot in sorted(_captured.items()):
            if want is not None and layer_id not in want:
                continue
            k = torch.cat(slot["k"], dim=0).cpu()
            v = torch.cat(slot["v"], dim=0).cpu()
            loc = torch.cat(slot["loc"], dim=0)
            torch.save(
                {"k": k, "v": v, "loc": loc.cpu()},
                f"{OUT}/{name}_L{layer_id}.pt",
            )
        print(f"{name}: dumped {n_layers} layers, k shape {k.shape}")

    print("done ->", OUT)


if __name__ == "__main__":
    main()