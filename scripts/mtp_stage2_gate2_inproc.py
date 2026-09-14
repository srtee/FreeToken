"""Gate 2 in-process: plain and spec LLM generation in ONE process, sequential.

One process = one CUDA/distributed init (single-shot), so the two modes share
the process: run plain, capture outputs, then run spec, then diff. Between the
modes the spec-side LLM is freshly constructed (its draft family capture owns
the FI scratch); the plain LLM's graphs are destroyed first.

Run:  .venv/bin/python scripts/mtp_stage2_gate2_inproc.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch

from mtp_stage2_gate import CKPT, GREEDY, PROMPTS


def main() -> None:
    # ONE Engine per process (CUDA/distributed init is single-shot). Plain
    # first (no spec arm, no draft family); then REUSE the same Engine with
    # the spec arm: attach the MTP drafter + capture the draft family and
    # re-run the same prompts. Same weights, same greedy params — the only
    # difference between the two passes is the spec loop + graphed draft.
    from freetoken.core import SamplingParams
    from freetoken.llm import LLM

    sp = SamplingParams(max_tokens=256, ignore_eos=True, **GREEDY)
    llm = LLM(model_path=CKPT, dtype=torch.bfloat16,
              attention_backend="auto", max_running_req=1,
              spec_mtp=False, cuda_graph_max_bs=1,
              moe_strategy="offload", expert_load="parallel",
              kv_reserve_tokens=2048, moe_cache_auto=True,
              max_seq_len_override=1024)
    plain = [list(r["token_ids"]) for r in llm.generate(PROMPTS, sp)]
    print("plain done:", [len(t) for t in plain], flush=True)

    eng = llm.engine
    assert eng.draft_graph_runner is None and eng.mtp_drafter is None
    from freetoken.engine.spec_mtp import MTPDrafter
    from freetoken.engine.graph import MTPDraftGraphRunner

    eng.attn_backend.reset_capture()
    eng.model.attach_mtp_head()
    eng.mtp_drafter = MTPDrafter(eng)
    eng.draft_graph_runner = MTPDraftGraphRunner(
        stream=eng.stream,
        device=eng.device,
        mtp=eng.model.model.mtp,
        attn_backend=eng.attn_backend,
        max_running_req=1,
        cuda_graph_max_bs=1,
        max_seq_len=eng.page_table.shape[1],
        vocab_size=eng.config.model_config.vocab_size,
        hidden_size=eng.config.model_config.hidden_size,
        dtype=eng.config.dtype,
        dummy_req=eng.dummy_req,
    )
    spec = [list(r["token_ids"]) for r in llm.generate(PROMPTS, sp)]
    print("spec done:", [len(t) for t in spec], flush=True)
    verdict = all(a == b for a, b in zip(plain, spec))
    total = sum(len(a) for a in plain)
    print(json.dumps({"byte_identical": verdict, "tokens": total}))
    from freetoken.distributed import destroy_distributed

    destroy_distributed()


if __name__ == "__main__":
    main()