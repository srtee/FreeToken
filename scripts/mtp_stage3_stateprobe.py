"""State probe: does the captured verify graph read/write the GDN state pool
slots it was staged? Discriminates staging bug vs kernel computation bug.

Arms (bs from PROBE_BS, GDN layer 0 = pool.local_index(0)):
  eager: fresh pool -> eager forward, FLAMetadata(cache_indices=slots 1..bs)
  graph: fresh pool -> runner.verify_row (captures staged slots 1..bs)
Checks:
  - pool tensor data_ptr unchanged since capture (elastic-realloc detector)
  - which rec/conv slots each arm wrote (nonzero scan)
  - graph-vs-eager logits + state row diffs
  - replay determinism (same pristine pool, replay twice)
"""
import os

CKPT = os.environ.get(
    "MTP_CKPT",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/"
        "snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5"))
import torch
MAX_BS = int(os.environ.get("PROBE_BS", 2))

from freetoken.engine.graph import MTPVerifyGraphRunner as _VGR
_orig_capture = _VGR._capture_graphs
def _patched_capture(self, max_seq_len, vocab_size, hidden_size, dtype):
    from freetoken.core import get_global_ctx
    p = get_global_ctx().linear_state_pool
    _patched_capture.ptrs = (p.recurrent_states.data_ptr(), p.conv_states.data_ptr())
    _orig_capture(self, max_seq_len, vocab_size, hidden_size, dtype)
_VGR._capture_graphs = _patched_capture


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
    device = eng.device
    H = eng.config.model_config.hidden_size
    vocab = eng.config.model_config.vocab_size
    bs = MAX_BS
    pool = eng.linear_state_pool
    assert pool is not None
    li = pool.local_index(0)
    rec, conv = pool.recurrent_states, pool.conv_states
    print(f"pool rec {tuple(rec.shape)} {rec.dtype} ptr={rec.data_ptr():#x} | "
          f"conv {tuple(conv.shape)} ptr={conv.data_ptr():#x}", flush=True)
    print(f"capture-time ptrs: {_patched_capture.ptrs}", flush=True)
    print(f"table_bufs[{bs}] staged = {runner.table_bufs[bs].tolist()}", flush=True)

    gen = torch.Generator().manual_seed(7)
    tokens = (torch.randint(0, vocab, (bs,), generator=gen, dtype=torch.int64) % 1000).int()
    slots = torch.arange(bs, device=device, dtype=torch.int32) + 1

    from freetoken.core import Batch, Req
    from freetoken.attention.linear import FLAMetadata

    def make_batch(tag_uid, q):
        rows = [(int(tokens[i]), q, int(slots[i])) for i in range(bs)]
        reqs = [Req(input_ids=torch.zeros(pos + 1, dtype=torch.int32),
                    table_idx=eng.dummy_req.table_idx, cached_len=pos,
                    output_len=4, uid=tag_uid - i, sampling_params=None,
                    cache_handle=None)
                for i, (tok, pos, slot) in enumerate(rows)]
        b = Batch(reqs=reqs, phase="decode")
        b.padded_reqs = b.reqs
        b.input_ids = torch.tensor([r[0] for r in rows], dtype=torch.int32, device=device)
        b.positions = torch.tensor([r[1] for r in rows], dtype=torch.int32, device=device)
        b.out_loc = torch.tensor([r[2] for r in rows], dtype=torch.int32, device=device)
        b.linear_table_idx = slots
        return b

    def fresh_state():
        pool.clear_slots(list(range(pool._num_slots)))
        if eng.moe_offload_cache is not None:
            eng.moe_offload_cache.reset()

    def run_eager(q):
        fresh_state()
        b = make_batch(-300, q)
        b.fla_metadata = FLAMetadata(
            cu_seqlens=torch.arange(bs + 1, dtype=torch.int32, device=device),
            cache_indices=slots)
        eng.attn_backend.prepare_metadata(b)
        with eng.ctx.forward_batch(b):
            logits = eng.model.forward()
        torch.cuda.synchronize()
        return logits[:bs].float().clone()

    def run_graph(q, tag):
        fresh_state()
        nz = (rec[li].abs().amax(dim=(1, 2, 3)) > 0).nonzero().flatten().tolist()
        print(f"  [{tag}] pool pristine check — nonzero rec slots: {nz}", flush=True)
        gb = make_batch(-400, q)
        lg, hid = runner.verify_row(gb)
        torch.cuda.synchronize()
        return lg[:bs].float().clone(), hid[:bs].clone()

    q = 0
    e = run_eager(q)
    e_rec, e_conv = rec[li].clone(), conv[li].clone()
    g, _ = run_graph(q, "graph-1")
    g_rec, g_conv = rec[li].clone(), conv[li].clone()
    g2, _ = run_graph(q, "graph-2")
    g2_rec = rec[li].clone()

    print(f"eager logits[0] top3 {e[0].topk(3).values.tolist()}", flush=True)
    print(f"graph logits[0] top3 {g[0].topk(3).values.tolist()}", flush=True)
    print(f"graph-vs-eager logits maxdiff={(g - e).abs().max().item():.6g} "
          f"argmax_eq={torch.equal(e.argmax(-1), g.argmax(-1))}", flush=True)
    print(f"graph2-vs-graph logits maxdiff={(g2 - g).abs().max().item():.6g}", flush=True)
    print(f"rec determinism (graph2 vs graph1) maxdiff="
          f"{(g2_rec - g_rec).abs().max().item():.6g}", flush=True)
    nz_e = (e_rec.abs().amax(dim=(1, 2, 3)) > 0).nonzero().flatten().tolist()
    nz_g = (g_rec.abs().amax(dim=(1, 2, 3)) > 0).nonzero().flatten().tolist()
    print(f"eager wrote rec slots {nz_e}", flush=True)
    print(f"graph wrote rec slots {nz_g}", flush=True)
    for s in sorted(set(nz_e) | set(nz_g)):
        en = e_rec[s].abs().max().item()
        gn = g_rec[s].abs().max().item()
        d = (e_rec[s].float() - g_rec[s].float()).abs().max().item() \
            if s in nz_e and s in nz_g else float("nan")
        print(f"  slot {s}: eager|max|={en:.4g} graph|max|={gn:.4g} diff={d:.4g}", flush=True)
    nzce = [s for s in range(conv.shape[1]) if conv[li][s].abs().max() > 0]
    # conv after eager:
    print(f"conv eager wrote slots {nzce}", flush=True)
    nzc_g = [s for s in range(conv.shape[1]) if g_conv[s].abs().max() > 0]
    print(f"conv graph wrote slots {nzc_g}", flush=True)


if __name__ == "__main__":
    main()
