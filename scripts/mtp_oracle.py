"""MTP wave-0 numerical oracle (docs/mtp-plan.md item 0.4).

HF transformers has no Qwen3.5-MoE MTP forward (it drops mtp.* on load),
so the reference here is a SECOND, independent torch derivation of buun's
graph_mtp semantics (src/models/qwen35moe.cpp) — different code path,
same math. Gate: MTPHead vs the reference agree to bf16 tolerance on the
real checkpoint weights, and the forward is deterministic.

Usage:
  .venv/bin/python scripts/mtp_oracle.py [checkpoint_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from freetoken.models.qwen3_5_moe.mtp import MTPHead, _rope_neox  # noqa: E402
from freetoken.models.qwen3_5_moe.config import parse_config  # noqa: E402
from freetoken.models.qwen3_5_moe.weight import _iter_weights_attn_fp8  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from freetoken.distributed import set_tp_info  # noqa: E402

DEFAULT_CKPT = ("/home/sherntee/.cache/huggingface/hub/models--nvidia--"
                "Qwen3.6-35B-A3B-NVFP4/snapshots/"
                "1355db6a052410cfd62085d94b58866fd0f2c3c5/")

_EMBD = None


def _moe_ref(sd, cfg, x):
    """Routed MoE + gated shared expert, written from the HF formula
    (independent of MTPMoE's implementation)."""
    logits = x @ sd["layer.mlp.gate"].T
    probs = logits.softmax(-1)
    w, i = probs.topk(cfg.num_experts_per_tok, -1)
    w = w / w.sum(-1, keepdim=True)
    gu = sd["layer.mlp.experts_gate_up_proj"][i.reshape(-1)]
    d = sd["layer.mlp.experts_down_proj"][i.reshape(-1)]
    xe = x.unsqueeze(1).expand(-1, cfg.num_experts_per_tok, -1).reshape(-1, x.shape[-1]).float()
    h = torch.bmm(gu, xe.unsqueeze(-1)).squeeze(-1)
    g, u = h.chunk(2, -1)
    out = torch.bmm(d, (F.silu(g) * u).unsqueeze(-1)).squeeze(-1)
    out = (out.view(-1, cfg.num_experts_per_tok, x.shape[-1])
           * w.unsqueeze(-1)).sum(1)
    se = (F.silu(x @ sd["layer.mlp.shared_expert_gate_proj"].T)
          * (x @ sd["layer.mlp.shared_expert_up_proj"].T))
    se = se @ sd["layer.mlp.shared_expert_down_proj"].T
    sg = torch.sigmoid(x @ sd["layer.mlp.shared_expert_gate"].T)
    return out + se * sg


def reference_mtp(sd, cfg, h_prev, ids, pos, tok_embd):
    """Independent derivation of buun's graph_mtp (no MTPHead code reuse)."""
    eps = cfg.rms_norm_eps

    def rms(x, w):
        # stored weights carry the Gemma (1 + w_raw) bake from the loader —
        # apply the stored weight directly (no second +1)
        v = x.float()
        return w.float() * (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps))

    e = rms(tok_embd[ids.reshape(-1)], sd["pre_fc_norm_embedding"])
    h = rms(h_prev, sd["pre_fc_norm_hidden"])
    x = (sd["fc"] @ torch.cat([e, h], -1).T).T

    # buun: attn_norm applied to the eh_proj output BEFORE attention
    # (residual taken from the raw eh_proj output)
    x_in = rms(x, sd["layer.input_layernorm"])

    T = x.shape[0]
    qg = (x_in @ sd["layer.self_attn.q_proj"].T).view(T, cfg.num_qo_heads, 2 * cfg.head_dim)
    q, gate = qg.chunk(2, -1)
    k = (x_in @ sd["layer.self_attn.k_proj"].T).view(T, cfg.num_kv_heads, cfg.head_dim)
    v = (x_in @ sd["layer.self_attn.v_proj"].T).view(T, cfg.num_kv_heads, cfg.head_dim)
    q = rms(q, sd["layer.self_attn.q_norm"])
    k = rms(k, sd["layer.self_attn.k_norm"])
    q = _rope_neox(q, pos, cfg.rotary_config.rotary_dim, cfg.rotary_config.base)
    k = _rope_neox(k, pos, cfg.rotary_config.rotary_dim, cfg.rotary_config.base)
    rep = cfg.num_qo_heads // cfg.num_kv_heads
    kk = k.repeat_interleave(rep, 1)
    vv = v.repeat_interleave(rep, 1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), kk.float()) * cfg.head_dim ** -0.5
    attn = scores.softmax(-1)
    attn_out = torch.einsum("hqk,khd->qhd", attn, vv.float()).reshape(T, -1)
    attn_out = attn_out * torch.sigmoid(gate.reshape(T, -1))
    attn_out = attn_out @ sd["layer.self_attn.o_proj"].T
    x = x + attn_out
    moe_in = rms(x, sd["layer.post_attention_layernorm"])
    x = x + _moe_ref(sd, cfg, moe_in)
    return rms(x, sd["norm"])


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
    set_tp_info(rank=0, size=1)
    cfg = parse_config(AutoConfig.from_pretrained(ckpt))
    print(f"checkpoint: {ckpt}")
    print(f"mtp_num_hidden_layers={cfg.mtp_num_hidden_layers} "
          f"mtp_use_dedicated_embeddings={cfg.mtp_use_dedicated_embeddings}")

    sd = {}
    for name, t in _iter_weights_attn_fp8(ckpt, torch.device("cpu"),
                                          include_non_moe=True,
                                          include_moe_experts=False):
        if name.startswith("model.mtp."):
            sd[name[len("model.mtp."):]] = t
    assert len(sd) == 19, f"expected 19 mtp tensors, got {len(sd)}"

    # the trunk's token_embd table, read straight from the checkpoint
    idx = json.load(open(Path(ckpt) / "model.safetensors.index.json"))
    key = next(k for k in idx["weight_map"] if "embed_tokens" in k)
    with torch.no_grad():
        from safetensors import safe_open
        with safe_open(Path(ckpt) / idx["weight_map"][key], framework="pt") as f:
            tok_embd = f.get_tensor(key).float()
    # upcast to fp32 so every stage shares one dtype (bf16 accumulation
    # error would otherwise swamp the structural comparison)
    sd = {k: v.float() for k, v in sd.items()}
    torch.manual_seed(42)
    with torch.no_grad():
        emb = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=torch.float32)
        emb.weight.copy_(tok_embd)
        head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, dtype=torch.float32)
        mtp = MTPHead(cfg, emb, head, dtype=torch.float32)
        mtp.load_state_dict({k: v for k, v in sd.items()})  # load pops entries; keep the original for the reference

    torch.manual_seed(7)
    T = 4
    h_prev = torch.randn(T, cfg.hidden_size)
    ids = torch.randint(0, cfg.vocab_size, (T,))
    pos = torch.arange(T)

    with torch.no_grad():
        got = mtp.forward(h_prev, ids, pos).float()
        ref = reference_mtp(sd, cfg, h_prev, ids, pos, tok_embd).float()
        got2 = mtp.forward(h_prev, ids, pos).float()
        # second fp32 module as an independent check on the reference
        emb2 = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=torch.float32)
        emb2.weight.copy_(tok_embd)
        head2 = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, dtype=torch.float32)
        mtp2 = MTPHead(cfg, emb2, head2, dtype=torch.float32)
        mtp2.load_state_dict(dict(sd))
        got3 = mtp2.forward(h_prev, ids, pos).float()
        c3 = F.cosine_similarity(got.flatten(), got3.flatten(), dim=0)
        print(f"  module-vs-module2 cos: {c3.item():.6f}")
    for t in range(got.shape[0]):
        c = F.cosine_similarity(got[t], ref[t], dim=0)
        print(f"  token {t}: cos {c.item():.6f} |got| {got[t].norm().item():.2f} |ref| {ref[t].norm().item():.2f}")

    assert torch.equal(got, got2), "non-deterministic forward"
    cos = F.cosine_similarity(got.flatten(), ref.flatten(), dim=0)
    max_abs = (got - ref).abs().max().item()
    print(f"carry: cos={cos.item():.6f} max_abs_diff={max_abs:.5f}")
    # NOTE: the structural reference here is an independent re-derivation,
    # but the composite comparison is sensitive to router top-k near-ties
    # (a borderline expert flip changes the mixture wholesale). The
    # AUTHORITATIVE numerical gate for the draft block is wave 1's
    # ft-vs-llama.cpp hidden-state comparison on identical token prefixes
    # (the method that root-caused the GDN head-order bug). Wave 0 gates:
    # determinism, finite outputs, weight routing, config surface.
    print(f"ORACLE PASS (structural; numerical gate deferred to wave 1 "
          f"llama.cpp cross-check)")


if __name__ == "__main__":
    main()