"""Layer bisect + sub-op bisect: localizing eager-vs-graph divergence (bs>=2).

Method (stage A, layer bisect): per-layer static output buffers written by
instance-level layer.forward wraps (layers are BaseOP, not nn.Module). Wraps
are installed (via a _capture_graphs monkeypatch) BEFORE the verify graph
family captures, so the buffer copies are recorded INTO the graph and refresh
on every replay. One eager run -> clone all layer snapshots; one graph replay
-> compare. The first divergent layer localizes the bug.

Stage B (sub-op bisect, PROBE_GDN=1): a class-level reimplementation of
Qwen3_5GatedDeltaNet.forward for layer 0 only, with graph-recorded snapshot
buffers for conv_in / a / b / mixed (conv decode out) / core (fla kernel out)
/ normed. Both arms run through the SAME patched forward, so eager-vs-graph
sub-op diffs localize the divergent kernel inside layer 0.

Stage C (eager cross-check, PROBE_XCHK=1): eager bs=1 (single row, same
token/position/slot as bs=2's row 0) vs the bs=2 eager run's row 0. If those
differ, the EAGER trunk path itself is bs-sensitive and the graph is not the
(bug) site.

Env: PROBE_BS (default 2), QLIST (default "0"), PROBE_GDN (default 1),
PROBE_XCHK (default 1).
"""
import os

CKPT = os.environ.get(
    "CKPT",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4"
        "/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5"))
import torch
from freetoken.core import get_global_ctx
MAX_BS = int(os.environ.get("PROBE_BS", 2))
Q_LIST = [int(q) for q in os.environ.get("QLIST", "0").split(",")]
PROBE_GDN = os.environ.get("PROBE_GDN", "1") == "1"
PROBE_XCHK = os.environ.get("PROBE_XCHK", "1") == "1"

# --- hook plumbing: installed before capture so copies are graph-recorded ----
snaps: dict[int, torch.Tensor] = {}
final_buf: list[torch.Tensor] = []
gdn_snaps: dict[str, torch.Tensor] = {}


def install_hooks(model, bs, device, H):
    from freetoken.engine.graph import MTPVerifyGraphRunner  # noqa: F401
    layers = model.model.layers.op_list
    for i, layer in enumerate(layers):
        snaps[i] = torch.zeros(bs, H, dtype=torch.bfloat16, device=device)

        def mk(idx, orig=layer.forward):
            def wrapper(hidden, residual, _idx=idx, _orig=orig):
                h, res = _orig(hidden, residual)
                n = h.shape[0]
                snaps[_idx][:n] = h[:n]
                return h, res
            return wrapper
        layer.forward = mk(i)
    final_buf.append(torch.zeros(bs, H, dtype=torch.bfloat16, device=device))

    norm = model.model.norm
    norm_orig = norm.forward_add_residual

    def norm_wrapper(hidden, residual, _orig=norm_orig):
        h, _ = _orig(hidden, residual)
        n = h.shape[0]
        final_buf[0][:n] = h[:n]
        return h, _
    norm.forward_add_residual = norm_wrapper

    if not PROBE_GDN:
        return
    # Layer-0 GDN sub-op snapshots, graph-recorded (the class-level patched
    # forward below writes them on every forward inside the capture).
    gdn = model.model.layers.op_list[0].linear_attn
    conv_dim = gdn.conv_dim
    num_v, hv = gdn.num_v_heads, gdn.head_v_dim
    Hsz = eng_hidden[0]
    gdn_snaps['conv_in'] = torch.zeros(bs, conv_dim, dtype=torch.bfloat16, device=device)
    gdn_snaps['a'] = torch.zeros(bs, num_v, dtype=torch.bfloat16, device=device)
    gdn_snaps['b'] = torch.zeros(bs, num_v, dtype=torch.bfloat16, device=device)
    gdn_snaps['mixed'] = torch.zeros(bs, conv_dim, dtype=torch.bfloat16, device=device)
    gdn_snaps['core'] = torch.zeros(bs, num_v, hv, dtype=torch.bfloat16, device=device)
    gdn_snaps['normed'] = torch.zeros(bs, num_v * hv, dtype=torch.bfloat16, device=device)


eng_hidden = [None]


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
    eng_hidden[0] = H
    pool = eng.linear_state_pool
    assert pool is not None

    gen = torch.Generator().manual_seed(7)
    tokens = torch.randint(0, vocab, (bs,), generator=gen,
                           dtype=torch.int32) % 1000
    # GDN bands: eager uses slots, graphed uses slots+3 (hybrid pool = 4*mr+1)
    slots = torch.arange(bs, device=device, dtype=torch.int32) + 1

    from freetoken.attention.linear import FLAMetadata

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
        b.linear_table_idx = slots if out_off == 0 else slots + 3
        return b, rows

    def fresh_state():
        # Zero ALL GDN slots + MoE offload cache so both arms start from an
        # identical pristine state each run.
        pool.clear_slots(list(range(pool._num_slots)))
        if eng.moe_offload_cache is not None:
            eng.moe_offload_cache.reset()
    def run_eager(q, arm_bs=None):
        fresh_state()
        if arm_bs is None:
            b, _ = make_batch(-300, q, 0)
            b.fla_metadata = FLAMetadata(
                cu_seqlens=torch.arange(bs + 1, dtype=torch.int32, device=device),
                cache_indices=slots)
        else:
            # dedicated bs=1 batch: single req, same token/pos/slot as row 0
            b, _ = make_batch(-500, q, 0)
            b.padded_reqs = b.reqs[:arm_bs]
            b.reqs = b.reqs[:arm_bs]
            b.input_ids = b.input_ids[:arm_bs]
            b.positions = b.positions[:arm_bs]
            b.out_loc = b.out_loc[:arm_bs]
            b.linear_table_idx = b.linear_table_idx[:arm_bs]
            b.fla_metadata = FLAMetadata(
                cu_seqlens=torch.arange(arm_bs + 1, dtype=torch.int32, device=device),
                cache_indices=slots[:arm_bs])
        eng.attn_backend.prepare_metadata(b)
        with eng.ctx.forward_batch(b):
            logits = eng.model.forward()
        torch.cuda.synchronize()
        n = bs if arm_bs is None else arm_bs
        return logits[:n].float().clone(), {i: snaps[i][:n].clone() for i in snaps}

    def run_graph(q):
        fresh_state()
        gb, _ = make_batch(-400, q, 100)  # disjoint KV rows
        g_logits, _ = runner.verify_row(gb)
        torch.cuda.synchronize()
        return g_logits.float().clone()

    # --- Stage C: eager bs=1 vs bs=2 row 0 (same token/pos/slot) ----------
    if PROBE_XCHK:
        e2, layers2 = run_eager(0)                       # eager bs=2
        e1, _ = run_eager(0, arm_bs=1)                   # eager bs=1, row 0
        d = (e2[0] - e1[0]).abs().max().item()
        print(f"XCHK eager bs=2 row0 vs eager bs=1 row0: logits maxdiff={d:.6g}",
              flush=True)
        for i in sorted(layers2):
            d2 = (layers2[i][0].float() - snaps[i][0].float()).abs().max().item()
            if d2 != 0.0:
                print(f"XCHK   first divergent layer vs bs=1: layer {i} "
                      f"maxdiff={d2:.6g}", flush=True)
                break

    # --- Stage A/B: eager-vs-graph per-layer (+ per-sub-op) ---------------
    for q in Q_LIST:
        e, e_layers = run_eager(q)
        e_final = final_buf[0].clone()
        e_gdn = {k: v.clone() for k, v in gdn_snaps.items()}
        g = run_graph(q)
        first = None
        for i in sorted(snaps):
            d = (e_layers[i].float() - snaps[i].float()).abs().max().item()
            tag = ""
            if first is None and d != 0.0:
                first = i
                tag = "  <== first divergence"
            print(f"q={q:>3} layer {i:>2}: maxdiff={d:.6g}{tag}", flush=True)
        if PROBE_GDN:
            print(f"q={q:>3} GDN layer-0 sub-ops:", flush=True)
            for k in gdn_snaps:
                if k not in e_gdn:
                    continue
                dk = (e_gdn[k].float() - gdn_snaps[k].float()).abs().max().item()
                print(f"   {k:>8}: maxdiff={dk:.6g}", flush=True)
        df = (e_final.float() - final_buf[0].float()).abs().max().item()
        dl = (e - g).abs()
        print(f"q={q:>3} final norm maxdiff={df:.6g} | logits maxdiff="
              f"{dl.max().item():.6g} mean={dl.mean().item():.6g} "
              f"argmax_equal={bool(torch.equal(e.argmax(-1), g.argmax(-1)))}",
              flush=True)
        del e, e_layers, e_final, e_gdn, g, df, dl

    from freetoken.distributed import destroy_distributed
    destroy_distributed()


# Patch capture to install hooks BEFORE warmup/capture (hooks on trunk decoder
# layers only fire on trunk forwards; the draft family never touches them).
from freetoken.engine.graph import MTPVerifyGraphRunner as _VGR

_orig_capture = _VGR._capture_graphs


def _patched_capture(self, max_seq_len, vocab_size, hidden_size, dtype):
    eng_hidden[0] = hidden_size
    install_hooks(self.model, self.max_graph_bs, self.device, hidden_size)
    _orig_capture(self, max_seq_len, vocab_size, hidden_size, dtype)


_VGR._capture_graphs = _patched_capture

if PROBE_GDN:
    # Class-level patched GDN forward with graph-recorded sub-op snapshots.
    from freetoken.models.qwen3_5_moe import gdn as _gdn_mod
    from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_decode_fla

    _gdn_orig_forward = _gdn_mod.Qwen3_5GatedDeltaNet.forward

    def _patched_gdn_forward(self, hidden_states):
        # Only layer-0 (the first divergent layer) gets the instrumented path;
        # every other GDN layer runs the original forward unchanged.
        if self.layer_id != 0 or 'conv_in' not in gdn_snaps:
            return _gdn_orig_forward(self, hidden_states)
        ctx = get_global_ctx()
        batch = ctx.batch
        pool_ = ctx.linear_state_pool
        total = hidden_states.shape[0]
        dtype = hidden_states.dtype
        fla = batch.fla_metadata
        if self._split_in_proj:
            qkvz = self.in_proj_qkvz.forward(hidden_states)
            conv_in, z = torch.split(qkvz, [self.conv_dim, self.value_dim], dim=-1)
            ba = self.in_proj_ba.forward(hidden_states)
            b_, a_ = torch.split(ba, [self.num_v_heads, self.num_v_heads], dim=-1)
        else:
            proj = self.in_proj.forward(hidden_states)
            conv_in, z, b_, a_ = torch.split(proj, self._in_proj_split, dim=-1)
        z = z.reshape(-1, self.head_v_dim)
        li = pool_.local_index(self.layer_id)
        gdn_snaps['conv_in'][:total] = conv_in
        gdn_snaps['a'][:total] = a_[:total]
        gdn_snaps['b'][:total] = b_[:total]
        mixed = self._conv_decode(conv_in, fla.cache_indices, pool_)
        gdn_snaps['mixed'][:total] = mixed
        B = mixed.shape[0]
        qf, kf, vf = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim],
                                 dim=-1)
        q = qf.reshape(1, B, self.num_k_heads, self.head_k_dim).to(dtype)
        k = kf.reshape(1, B, self.num_k_heads, self.head_k_dim).to(dtype)
        v = vf.reshape(1, B, self.num_v_heads, self.head_v_dim).to(dtype)
        core_out = gdn_decode_fla(
            q, k, v, a_, b_, A_log=self.A_log, dt_bias=self.dt_bias,
            state_source=pool_.recurrent_states[li],
            indices=fla.cache_indices,
            cu_seqlens=fla.cu_seqlens, scale=self.head_k_dim ** -0.5,
        )
        gdn_snaps['core'][:total] = core_out
        core_out = core_out.reshape(-1, self.head_v_dim)
        out = self.norm.forward(core_out, z).reshape(total, -1)
        gdn_snaps['normed'][:total] = out
        return self.out_proj.forward(out)

    _gdn_mod.Qwen3_5GatedDeltaNet.forward = _patched_gdn_forward

if __name__ == "__main__":
    main()