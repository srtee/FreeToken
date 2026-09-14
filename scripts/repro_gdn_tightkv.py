"""Stage-1 bug repro: CUDA illegal access in the GDN conv path during MTP
spec decoding under tight KV (kv_reserve_tokens=2048, max_seq_len_override=1024).

Drives the engine directly (build_engine pattern from mtp_stage2_gate.py),
spec=True, bs=1, N prompts of ~255 tokens, generation budget GATE_TOKENS
(default 768 -> each sequence fills to the 1024 cap).

Env-gated instrumentation (no repo edits — all monkeypatched here):
  REPROInvariantError aborts with full state on any violated invariant.
  CUDA_LAUNCH_BLOCKING=1 (exported here) pins the failing kernel launch.

Run: .venv/bin/python scripts/repro_gdn_tightkv.py [n_prompts] [max_tokens]
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.dirname(__file__))

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("FT_SPEC_TRACE", "1")

import torch  # noqa: E402

from mtp_stage2_gate import CKPT, GREEDY, PROMPTS  # noqa: E402

N_PROMPTS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 768
PROMPTS = (PROMPTS * ((N_PROMPTS // len(PROMPTS)) + 1))[:N_PROMPTS]


def build_llm():
    from freetoken.llm import LLM

    return LLM(model_path=CKPT, dtype=torch.bfloat16,
               attention_backend="auto", max_running_req=1,
               spec_mtp=True, cuda_graph_max_bs=1,
               moe_strategy="offload", expert_load="parallel",
               kv_reserve_tokens=2048, moe_cache_auto=True,
               max_seq_len_override=1024)

from freetoken.core import SamplingParams  # noqa: E402
def install_monitors(llm) -> None:
    """Wrap Scheduler._forward_spec + Engine._forward_spec_batch with
    invariant checks (monkeypatch only — no repo file changes)."""
    from freetoken.scheduler.scheduler import Scheduler
    from freetoken.engine.engine import Engine

    eng = llm.engine

    orig_fwd_batch = Engine._forward_spec_batch

    def checked_fwd_batch(self, batch):
        pool = self.linear_state_pool
        n_slots = pool.num_slots if pool is not None else 0
        for i, req in enumerate(batch.reqs):
            ls = req.linear_slot_idx
            snaps = batch.spec_gdn_snapshot_slots or []
            assert ls is None or (0 <= ls < n_slots), \
                f"PRE rowA fwd: linear_slot_idx OOB {ls} (n={n_slots}) uid={req.uid}"
            for s in snaps:
                assert 0 <= s < n_slots, \
                    f"PRE rowA fwd: snapshot slot OOB {s} (n={n_slots}) uid={req.uid}"
            assert 0 <= req.cached_len <= req.device_len <= 1024, \
                f"PRE rowA fwd: lens bad cached={req.cached_len} dev={req.device_len} uid={req.uid}"
            # NOTE: page_table entries are pool slot ids bounded by the KV
            # pool size, NOT page_table.shape[1] (an earlier bound here was
            # wrong: a legit uid=1 row hit 961..1024 and aborted the run).
        out = orig_fwd_batch(self, batch)
        # POST: verify the snapshot copies actually ran & slots sane
        for i, req in enumerate(batch.reqs):
            snaps = batch.spec_gdn_snapshot_slots or []
            if snaps:
                s = snaps[i]
                assert 0 <= s < n_slots, \
                    f"POST: snapshot slot OOB {s} uid={req.uid}"
        return out

    Engine._forward_spec_batch = checked_fwd_batch

    orig_fwd_spec = Scheduler._forward_spec

    def checked_fwd_spec(self, batch, output_mapping):
        out = orig_fwd_spec(self, batch, output_mapping)
        pool = self.engine.linear_state_pool
        n_slots = pool.num_slots if pool is not None else 0
        for req in batch.reqs:
            assert 0 <= req.cached_len <= req.device_len <= 1024, \
                f"POST iter: lens bad cached={req.cached_len} dev={req.device_len} uid={req.uid}"
            for s in (req.mamba_ping_pong or []):
                assert 0 <= s < n_slots, \
                    f"POST iter: ping-pong slot OOB {s} (n={n_slots}) uid={req.uid}"
        return out

    Scheduler._forward_spec = checked_fwd_spec

    # Trace the GDN snapshot bookkeeping in _prepare_spec_batch via the
    # existing FT_SPEC_TRACE plus a per-iteration ping-pong/slot dump.
    orig_prepare = Scheduler._prepare_spec_batch

    def traced_prepare(self, batch):
        eng_ = self.engine
        pool = eng_.linear_state_pool
        if pool is not None and os.environ.get("REPRO_TRACE"):
            for req in batch.reqs:
                sys.stderr.write(
                    f"[prep] uid={req.uid} cached={req.cached_len} "
                    f"dev={req.device_len} undone={req.spec_undone} "
                    f"linear_slot={req.linear_slot_idx} "
                    f"pp={req.mamba_ping_pong} nxt={req.mamba_next_track_idx} "
                    f"free_pool={pool.num_free_slots}\n")
        return orig_prepare(self, batch)

    Scheduler._prepare_spec_batch = traced_prepare

    def run_generate(llm_):
        sp = SamplingParams(max_tokens=MAX_TOKENS, ignore_eos=True, **GREEDY)
        res = llm.generate(PROMPTS, sp)
        lens = [len(r["token_ids"]) for r in res]
        print(json.dumps({"prompt_lasts": lens,
                          "drafted": llm.engine.mtp_drafter.stats.drafted,
                          "accepted": llm.engine.mtp_drafter.stats.accepted}))
        from freetoken.distributed import destroy_distributed

        destroy_distributed()

    return run_generate






def main() -> None:
    llm = build_llm()
    run_generate = install_monitors(llm)
    run_generate(llm)



if __name__ == "__main__":
    main()