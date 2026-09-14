"""Gate-2 worker: one LLM generation pass in a given mode (plain | spec).

Printed as the last stdout line: a JSON list of token-id lists, one per prompt.
Run by scripts/mtp_stage2_gate.py (the parent diffs plain vs spec).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch
from mtp_stage2_gate import CKPT, GREEDY, PROMPTS


def main() -> None:
    mode = sys.argv[1]
    from freetoken.core import SamplingParams
    from freetoken.llm import LLM

    sp = SamplingParams(max_tokens=256, ignore_eos=True, **GREEDY)
    llm = LLM(model_path=CKPT, dtype=torch.bfloat16,
              attention_backend="auto", max_running_req=1,
              spec_mtp=(mode == "spec"), cuda_graph_max_bs=1,
              moe_strategy="offload", expert_load="parallel",
              kv_reserve_tokens=2048, moe_cache_auto=True,
              max_seq_len_override=1024)
    res = llm.generate(PROMPTS, sp)
    token_ids = [r["token_ids"] for r in res]
    print(json.dumps(token_ids))
    from freetoken.distributed import destroy_distributed

    destroy_distributed()


if __name__ == "__main__":
    main()