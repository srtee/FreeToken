"""Seqlen sweep: verify-row bit equality vs replay-time seqlen (bs=1).

Hypothesis under test: the captured FI decode kernels were planned at
CAPTURE time with the dummy req's seqlen=1; replay re-plans with real
seqlens. If the captured kernels can't honor a changed plan, the row is
wrong (bs=1) or IMAs (bs>=2) exactly when replay seqlen > capture seqlen.

Sweep q in {0, 1, 8, 32, 100}: q=0 replays with seqlen 1 == the capture
plan -> must be bit-equal; q>0 diverges iff the hypothesis holds.

Env:   PROBE_BS (default 1)
"""
import os

CKPT = os.environ.get(
    "CKPT",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4"
        "/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5"))
import torch
MAX_BS = int(os.environ.get("PROBE_BS", 1))
Q_LIST = [int(q) for q in os.environ.get("QLIST", "0,1,8,32,100").split(",")]


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
    H = eng.config.model_config.hidden_size
    vocab = eng.config.model_config.vocab_size

    gen = torch.Generator().manual_seed(7)
    bs = MAX_BS
    tokens = torch.randint(0, vocab, (bs,), generator=gen,
                           dtype=torch.int32) % 1000
    # GDN bands: eager uses slots, graphed uses slots+3 (hybrid pool = 4*mr+1)
    slots = torch.arange(bs, device=device, dtype=torch.int32) + 1

    from freetoken.attention.linear import FLAMetadata

    # Warm the MoE offload cache ONCE (the tail probe showed cache state
    # doesn't change the graphed arm's output; warming avoids noise).
    def make_batch(tag_uid, q, out_off):
        from freetoken.core import Batch, Req
        rows = [(int(tokens[i]), q, int(slots[i]), int(slots[i]) + out_off)
                for i in range(bs)]
        reqs = []
        for i, (tok, pos, slot, out) in enumerate(rows):
            reqs.append(Req(
                input_ids=torch.zeros(pos + 1, dtype=torch.int32),
                table_idx=eng.dummy_req.table_idx,
                cached_len=pos, output_len=4, uid=tag_uid - i,
                sampling_params=None, cache_handle=None))
        b = Batch(reqs=reqs, phase="decode")
        b.padded_reqs = b.reqs
        b.input_ids = torch.tensor([r[0] for r in rows],
                                   dtype=torch.int32, device=device)
        b.positions = torch.tensor([r[1] for r in rows],
                                   dtype=torch.int32, device=device)
        b.out_loc = torch.tensor([r[3] for r in rows],
                                 dtype=torch.int32, device=device)
        # Linear slots must be DISJOINT between arms (else the graphed row
        # inherits the eager arm's advanced GDN state) and IN-BOUNDS for the
        # pool (4*mr+1 slots; slots+out_off would be OOB). +bs satisfies both.
        b.linear_table_idx = slots + (bs if out_off else 0)
        return b, rows

    def run_eager(q):
        b, _ = make_batch(-300, q, 0)
        b.fla_metadata = FLAMetadata(
            cu_seqlens=torch.arange(bs + 1, dtype=torch.int32, device=device),
            cache_indices=slots)
        eng.attn_backend.prepare_metadata(b)
        with eng.ctx.forward_batch(b):
            logits = eng.model.forward()
        torch.cuda.synchronize()
        return logits[:bs].float().clone()

    def run_graph(q):
        gb, _ = make_batch(-400, q, 100)  # disjoint KV rows
        g_logits, _ = runner.verify_row(gb)
        torch.cuda.synchronize()
        return g_logits.float().clone()

    for q in Q_LIST:
        e = run_eager(q)
        e2 = run_eager(q)
        g = run_graph(q)
        g2 = run_graph(q)
        d_eg = (e - g).abs()
        d_ee = (e - e2).abs()
        d_gg = (g - g2).abs()
        eq = bool(torch.equal(e.argmax(-1), g.argmax(-1)))
        print(f"q={q:>3}: argmax_equal={eq} eager-vs-graph maxdiff={d_eg.max().item():.6g} "
              f"mean={d_eg.mean().item():.6g} | eager-vs-eager2 maxdiff={d_ee.max().item():.6g} "
              f"| graph1-vs-graph2 maxdiff={d_gg.max().item():.6g}", flush=True)
        # per-layer KV rows both arms wrote (eager: slots+0.., graph: +100)
        if q == Q_LIST[0]:
            for lid in [3, 39]:
                kc = eng.kv_cache.k_cache(lid)
                rows_e = torch.tensor([int(s) for s in slots], device=device)
                rows_g = rows_e + 100
                ke = kc[rows_e].cpu()
                kg = kc[rows_g].cpu()
                print(f"   kv layer {lid}: eager-vs-graph k maxdiff="
                      f"{(ke.float() - kg.float()).abs().max().item():.6g}", flush=True)
        del e, e2, g, g2, d_eg, d_ee, d_gg

    from freetoken.distributed import destroy_distributed
    destroy_distributed()


if __name__ == "__main__":
    main()