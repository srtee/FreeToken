"""bs=4 replay nondeterminism: find the capture-read buffer that moves.

The plain probe shows graph1-vs-graph2 != 0 for the verify family at bs=4
(bit-equal at bs<=2): the captured graph reads some memory that eager-side
work in between mutates. This script replays G, runs the eager arm E,
replays G again, and diffs full contents of every candidate persistent
buffer after each phase:

  FI plan outputs   capture.* device tensors (indptr/indices/lpl)
  FI workspaces     backend int/float workspace buffers
  wrapper buffers   graph_wrappers[bs] tensor attrs
  verify runner     buffer tensors, table_bufs, fla_cu_seqlens
  GDN pool          recurrent_states / conv_states (legit per-replay mutation)
  MoE offload       one-level tensor attr walk
  KV cache          rows at the verify slots for probed layers (best effort)

Legit mutations (GDN state, KV rows, MoE LRU) must show IDENTICAL deltas
across G1->G2 and G2->G3; anything else differs = the clobbered state.

Env: PROBE_BS (default 4), LAYER_KV (default "3,39").
"""
import os

CKPT = os.environ.get(
    "CKPT",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4"
        "/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5"))
import torch

MAX_BS = int(os.environ.get("PROBE_BS", 4))
KV_LAYERS = [int(x) for x in os.environ.get("LAYER_KV", "3,39").split(",")]


def tensor_walk(obj, prefix=""):
    """One-level attr walk collecting CUDA tensors: {name: tensor}."""
    out = {}
    for k, v in getattr(obj, "__dict__", {}).items():
        name = f"{prefix}.{k}" if prefix else k
        if isinstance(v, torch.Tensor) and v.is_cuda:
            out[name] = v
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, torch.Tensor) and vv.is_cuda:
                    out[f"{name}[{kk}]"] = vv
        elif isinstance(v, (list, tuple)):
            for ii, vv in enumerate(v):
                if isinstance(vv, torch.Tensor) and vv.is_cuda:
                    out[f"{name}[{ii}]"] = vv
    return out


def summarize(t: torch.Tensor):
    """On-device checksum, memory-bounded: chunked byte sum + byte min/max + sample.
    Byte views make it dtype-agnostic (fp8 has no min/max kernels)."""
    flat = t.reshape(-1).view(torch.uint8)
    acc = 0
    step = 4_000_000
    for i in range(0, flat.numel(), step):
        acc += int(flat[i:i + step].to(torch.int32).sum().item())
    sample = t.reshape(-1)[:: max(1, t.numel() // 65536)][:65536]
    return (acc, int(flat.min()), int(flat.max()),
            sample.float().cpu().sum().item())


def dump_state(eng, runner, slots_g):
    """Snapshot checksums of every candidate capture-read buffer."""
    be = eng.attn_backend
    s = {}
    for name, t in tensor_walk(be.capture, "fi.capture").items():
        s[name] = summarize(t)
    s["fi.int_ws"] = summarize(be.int_workspace_buffer)
    s["fi.float_ws"] = summarize(be.float_workspace_buffer)
    w = be.graph_wrappers[MAX_BS]
    for name, t in tensor_walk(w, f"fi.wrap{MAX_BS}").items():
        s[name] = summarize(t)
    for name, t in tensor_walk(runner.buffer, "vbuf").items():
        s[name] = summarize(t)
    for b, buf in runner.table_bufs.items():
        s[f"vrunner.table_bufs[{b}]"] = summarize(buf)
    s["vrunner.fla_cu_seqlens"] = summarize(runner.fla_cu_seqlens)
    pool = eng.linear_state_pool
    s["gdn.recurrent_states"] = summarize(pool.recurrent_states)
    s["gdn.conv_states"] = summarize(pool.conv_states)
    moe = eng.moe_offload_cache
    if moe is not None:
        for name, t in tensor_walk(moe, "moe").items():
            s[name] = summarize(t)
    kv = getattr(eng, "kv_cache_pool", None) or getattr(eng, "kv_cache", None)
    try:
        for lid in KV_LAYERS:
            s[f"kv[{lid}]@g"] = summarize(kv.layers[lid].k_buffer[slots_g])
    except AttributeError:
        print("  (kv row dump skipped: unknown pool layout)", flush=True)
    return s


def diff_states(a, b, label):
    print(f"--- {label} ---", flush=True)
    assert set(a) == set(b), "dump keys diverged"
    for name in a:
        x, y = a[name], b[name]
        if x != y:
            print(f"  DIFF {name}:", flush=True)
            for i, f in enumerate(("sum", "min", "max", "sample")):
                if x[i] != y[i]:
                    print(f"    {f}: {x[i]} -> {y[i]}", flush=True)

def main() -> None:
    from freetoken.llm import LLM

    llm = LLM(model_path=CKPT, dtype=torch.bfloat16,
              attention_backend="auto", max_running_req=MAX_BS,
              spec_mtp=True, cuda_graph_max_bs=MAX_BS,
              moe_strategy="offload", expert_load="parallel",
              kv_reserve_tokens=8192, moe_cache_auto=True)
    eng = llm.engine if hasattr(llm, "engine") else llm
    runner = eng.verify_graph_runner
    assert runner is not None
    device = eng.device
    vocab = eng.config.model_config.vocab_size
    bs = MAX_BS
    q = 0

    gen = torch.Generator().manual_seed(7)
    tokens = torch.randint(0, vocab, (bs,), generator=gen,
                           dtype=torch.int32) % 1000
    slots = torch.arange(bs, device=device, dtype=torch.int32) + 1

    from freetoken.core import Batch, Req

    def make_batch(tag_uid, out_off):
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
        b.linear_table_idx = slots if out_off == 0 else slots + out_off
        return b

    def run_eager():
        b = make_batch(-300, 0)      # eager rows: KV slots 1..4, linear 1..4
        eng.attn_backend.prepare_metadata(b)
        with eng.ctx.forward_batch(b):
            return eng.model.forward()[:bs].float().clone()

    def run_graph():
        b = make_batch(-400, 7)      # graph rows: KV slots 8..11, linear 8..11 (disjoint)
        lg, _ = runner.verify_row(b)
        torch.cuda.synchronize()
        return lg.float().clone()
    e1 = run_eager()
    g1 = run_graph()
    d1 = dump_state(eng, runner, slots + 7)
    e2 = run_eager()
    g2 = run_graph()
    d2 = dump_state(eng, runner, slots + 7)
    e3 = run_eager()
    g3 = run_graph()
    d3 = dump_state(eng, runner, slots + 7)
    zero_ws_cnt = [0]

    def zero_ws():
        for a in ("float_ws", "int_ws"):
            t = getattr(eng.attn_backend, a, None)
            if t is not None:
                t.zero_()
        zero_ws_cnt[0] += 1

    # Causal test: identical replays with a controlled (zeroed) FI workspace.
    zero_ws(); g4 = run_graph()
    e4 = run_eager()          # dirties the workspace like any eager step
    zero_ws(); g5 = run_graph()
    torch.cuda.synchronize()

    for i in range(bs):
        print(f"row{i}: g1g2={(g1[i] - g2[i]).abs().max().item():.6g} "
              f"g2g3={(g2[i] - g3[i]).abs().max().item():.6g} "
              f"e1g1={(e1[i] - g1[i]).abs().max().item():.6g} "
              f"e2g2={(e2[i] - g2[i]).abs().max().item():.6g}", flush=True)
    for i in range(bs):
        print(f"row{i}: g4g5={(g4[i] - g5[i]).abs().max().item():.6g} "
              f"e4g4={(e4[i] - g4[i]).abs().max().item():.6g} "
              f"g1g4={(g1[i] - g4[i]).abs().max().item():.6g}", flush=True)
    print(f"ws zeroed {zero_ws_cnt[0]} times", flush=True)

    diff_states(d1, d2, "G1 vs G2 (eager E2 interleaved)")
    diff_states(d2, d3, "G2 vs G3 (eager E3 interleaved)")

    from freetoken.distributed import destroy_distributed
    destroy_distributed()


if __name__ == "__main__":
    main()
