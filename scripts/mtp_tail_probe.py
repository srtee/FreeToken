"""Tail bisect: which post-attention op diverges under CUDA-graph replay?

Engine boots PLAIN (no MTP, no trunk graphs). For each tail op we:
  1. run it eagerly on a fixed input -> y_eager
  2. capture it in a CUDA graph on a static input buffer
  3. restage the same input, replay -> y_graph
  4. compare bit-exactly
Ops: layer-39 offload MoE, layer-39 o_proj... (via full mlp), final GemmaRMSNorm,
lm_head. If MoE diverges, bisect inside: router gate / shared expert / routed.
"""
import os

CKPT = os.environ.get(
    "CKPT",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4"
        "/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5"))
import torch


def diff(name, a, b):
    if a.shape != b.shape:
        print(f"{name}: SHAPE MISMATCH {a.shape} vs {b.shape}")
        return
    d = (a.float() - b.float()).abs()
    print(f"{name}: bit-equal={torch.equal(a, b)} maxdiff={d.max().item():.6g} mean={d.mean().item():.6g}")


def graph_op(fn, x_static, *rest_static):
    """Capture fn(x_static, *rest_static) once; return replay callable."""
    # warmup on a side stream (cuBLASLt init, marlin workspace)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(x_static, *rest_static)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn(x_static, *rest_static)
    return g


def main() -> None:
    from freetoken.llm import LLM

    llm = LLM(model_path=CKPT, dtype=torch.bfloat16,
              attention_backend="auto", max_running_req=1,
              spec_mtp=False, cuda_graph_max_bs=0,
              moe_strategy="offload", expert_load="parallel",
              kv_reserve_tokens=8192, moe_cache_auto=True)
    eng = llm.engine if hasattr(llm, "engine") else llm
    model = eng.model
    device = eng.device
    H = eng.config.model_config.hidden_size
    torch.manual_seed(0)
    x = (torch.randn(1, H, dtype=torch.bfloat16, device=device) * 0.5)

    # The offload-MoE forward reads ctx.batch (decode vs prefill branch);
    # provide a 1-row decode batch for every call below.
    from freetoken.core import Batch, Req, get_global_ctx
    _req = Req(input_ids=torch.zeros(2, dtype=torch.int32),
               table_idx=eng.dummy_req.table_idx, cached_len=1,
               output_len=4, uid=-1, sampling_params=None, cache_handle=None)
    _req.linear_slot_idx = eng.dummy_req.linear_slot_idx
    _b = Batch(reqs=[_req], phase="decode")
    _b.padded_reqs = _b.reqs
    _b.out_loc = torch.zeros(1, dtype=torch.int32, device=device)
    _b.positions = torch.zeros(1, dtype=torch.int32, device=device)
    _ctx_cm = get_global_ctx().forward_batch(_b)
    _ctx_cm.__enter__()

    layers = model.model.layers.op_list
    last = layers[-1]

    # ---- 1. layer-39 MoE block ----
    xs = x.clone()
    y_e = last.mlp.forward(xs.clone())
    g = graph_op(lambda t: last.mlp.forward(t), xs)
    xs.copy_(x)
    g.replay()
    torch.cuda.synchronize()
    # replay output lands where? capture returned a tensor allocated in the
    # graph pool; grab it via a static out: rerun capture storing output.
    # simpler: compare via a second eager call on same input for determinism first
    y_e2 = last.mlp.forward(x.clone())
    diff("moe eager-vs-eager", y_e, y_e2)
    # graph output: the captured graph's returned tensor is g's output buffer;
    # re-derive it by capturing with an explicit out copy.
    out_buf = torch.empty_like(y_e)
    xs2 = x.clone()
    g2 = graph_op(lambda t: out_buf.copy_(last.mlp.forward(t)), xs2)
    xs2.copy_(x)
    g2.replay()
    torch.cuda.synchronize()
    diff("moe eager-vs-graph", y_e, out_buf)

    # bisect inside MoE: router logits
    rs = x.clone()
    rl_e = last.mlp.gate.forward(rs.clone())
    rg = graph_op(lambda t: last.mlp.gate.forward(t), rs)
    rs.copy_(x)
    rg.replay()
    torch.cuda.synchronize()
    diff("router eager-vs-graph", rl_e, rl_e)  # placeholder shape check
    # (router graph output buffer: capture with out copy)
    rl_buf = torch.empty_like(rl_e)
    rs3 = x.clone()
    rg2 = graph_op(lambda t: rl_buf.copy_(last.mlp.gate.forward(t)), rs3)
    rs3.copy_(x)
    rg2.replay()
    torch.cuda.synchronize()
    diff("router eager-vs-graph", rl_e, rl_buf)

    # shared expert
    ss = x.clone()
    sh_e = last.mlp.shared_expert.forward(ss.clone())
    sh_buf = torch.empty_like(sh_e)
    ss2 = x.clone()
    sg = graph_op(lambda t: sh_buf.copy_(last.mlp.shared_expert.forward(t)), ss2)
    ss2.copy_(x)
    sg.replay()
    torch.cuda.synchronize()
    diff("shared expert eager-vs-graph", sh_e, sh_buf)

    # ---- 2. final norm (needs residual; fake one) ----
    res = (torch.randn(1, H, dtype=torch.bfloat16, device=device) * 0.5)
    xn = x.clone()
    n_e, _ = model.model.norm.forward_add_residual(xn.clone(), res.clone())
    n_buf = torch.empty_like(n_e)
    xn2 = x.clone()
    res2 = res.clone()
    # capture needs both statics: wrap
    s2 = torch.cuda.Stream()
    s2.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s2):
        for _ in range(3):
            model.model.norm.forward_add_residual(xn2, res2)
    torch.cuda.current_stream().wait_stream(s2)
    g3 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g3):
        n_buf.copy_(model.model.norm.forward_add_residual(xn2, res2)[0])
    xn2.copy_(x)
    res2.copy_(res)
    g3.replay()
    torch.cuda.synchronize()
    diff("final norm eager-vs-graph", n_e, n_buf)

    # ---- 3. lm_head on identical hidden ----
    xs4 = n_e.clone()
    l_e = model.lm_head.forward(xs4.clone())
    l_buf = torch.empty_like(l_e)
    xs5 = n_e.clone()
    g4 = graph_op(lambda t: l_buf.copy_(model.lm_head.forward(t)), xs5)
    xs5.copy_(n_e)
    g4.replay()
    torch.cuda.synchronize()
    diff("lm_head eager-vs-graph", l_e, l_buf)

    from freetoken.distributed import destroy_distributed
    destroy_distributed()


if __name__ == "__main__":
    main()