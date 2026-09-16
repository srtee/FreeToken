"""Mini-gate: reproduce the stage-3 gate worker's bit-equality loop in a
configurable harness (the full gate costs ~25 min per attempt; this costs
one model load and N_STEPS fast iterations).

Differences vs the passing seqlen-sweep probe, each toggled by env:
  MINI_MIX=1   mixed seqlens within one row batch (row i at q + (i%2)),
               like the gate's row A / row B staging
  MINI_RANDBS=1  random bs in 1..MAX_BS per step (family interleave),
               like the gate's 200-step random-bs loop
Default (both off): uniform seqlen, fixed bs == the passing sweep probe.

Env: PROBE_BS (default 2), MINI_STEPS (default 50), MINI_MIX, MINI_RANDBS
"""
import os

CKPT = os.environ.get(
    "CKPT",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4"
        "/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5"))
import torch

MAX_BS = int(os.environ.get("PROBE_BS", 2))
Q = int(os.environ.get("MINI_Q", "100"))
MIX = os.environ.get("MINI_MIX", "0") == "1"
RANDBS = os.environ.get("MINI_RANDBS", "0") == "1"
N_STEPS = int(os.environ.get("MINI_STEPS", "50"))
BAND = int(os.environ.get("MINI_BAND", "5"))


def main() -> None:
    from freetoken.llm import LLM

    llm = LLM(model_path=CKPT, dtype=torch.bfloat16,
              attention_backend="auto", max_running_req=MAX_BS,
              spec_mtp=True, cuda_graph_max_bs=MAX_BS,
              moe_strategy="offload", expert_load="parallel",
              kv_reserve_tokens=8192, moe_cache_auto=True)
    eng = llm.engine if hasattr(llm, "engine") else llm
    runner = eng.verify_graph_runner
    assert runner is not None, "verify family required"
    pool = eng.linear_state_pool
    assert pool is not None
    device = eng.device
    print(f"pool.num_slots={pool.num_slots}", flush=True)
    vocab = eng.config.model_config.vocab_size

    gen = torch.Generator().manual_seed(7)
    from freetoken.attention.linear import FLAMetadata

    def run_one(step, bs, q):
        tokens = torch.randint(0, vocab, (bs,), generator=gen,
                               dtype=torch.int32) % 1000
        slots = torch.arange(bs, device=device, dtype=torch.int32) + 1
        pos = [q + (i % 2 if MIX else 0) for i in range(bs)]

        from freetoken.core import Batch, Req

        def make_batch(tag_uid, out_off, band_off):
            rows = [(int(tokens[i]), pos[i], int(slots[i]),
                     int(slots[i]) + out_off) for i in range(bs)]
            reqs = [Req(
                input_ids=torch.zeros(r[1] + 1, dtype=torch.int32),
                table_idx=eng.dummy_req.table_idx,
                cached_len=r[1], output_len=4, uid=tag_uid - i,
                sampling_params=None, cache_handle=None)
                for i, r in enumerate(rows)]
            b = Batch(reqs=reqs, phase="decode")
            b.padded_reqs = b.reqs
            b.input_ids = torch.tensor([r[0] for r in rows],
                                       dtype=torch.int32, device=device)
            b.positions = torch.tensor([r[1] for r in rows],
                                       dtype=torch.int32, device=device)
            b.out_loc = torch.tensor([r[3] for r in rows],
                                     dtype=torch.int32, device=device)
            b.linear_table_idx = slots + band_off
            return b

        # eager arm
        b = make_batch(-300 - step, 0, 0)
        b.fla_metadata = FLAMetadata(
            cu_seqlens=torch.arange(bs + 1, dtype=torch.int32, device=device),
            cache_indices=slots)
        eng.attn_backend.prepare_metadata(b)
        with eng.ctx.forward_batch(b):
            logits = eng.model.forward()
        torch.cuda.synchronize()
        e = logits[:bs].float().clone()

        # graphed arm (disjoint KV rows + GDN band, pool = 4*mr+1 slots)
        gb = make_batch(-400 - step, 100, BAND)
        g_logits, _ = runner.verify_row(gb)
        torch.cuda.synchronize()
        g = g_logits.float().clone()

        eq = bool(torch.equal(e.argmax(-1), g.argmax(-1)))
        d = (e - g).abs()
        print(f"step {step:>3} bs={bs} q={q}: argmax_equal={eq} "
              f"maxdiff={d.max().item():.6g} mean={d.mean().item():.6g}",
              flush=True)
        del e, g, d, logits, g_logits

    for step in range(N_STEPS):
        bs = int(torch.randint(1, MAX_BS + 1, (1,), generator=gen)) \
            if RANDBS else MAX_BS
        run_one(step, bs, Q)

    from freetoken.distributed import destroy_distributed
    destroy_distributed()


if __name__ == "__main__":
    main()