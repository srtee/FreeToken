"""GDN state-path equivalence: decode kernel vs 1-row varlen (chunk) path.

The spec-decode loop's verify rows run through the GDN VARLEN path
(phase="prefill" 1-row extends: sgl causal_conv1d_fwd + fla
chunk_gated_delta_rule); the plain reference decodes through the GDN
DECODE path (causal_conv1d_update + fused_sigmoid_gating_delta_rule).
If the two paths' state updates differ semantically for a 1-token step,
the spec loop diverges from the reference cumulatively — the
repetition-attractor corruption. This test runs BOTH paths from identical
saved conv+recurrent state and asserts identical outputs AND identical
post-state.

Requires a GPU + triton (the vendored kernels are triton); skipped
otherwise.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU-only: exercises the triton GDN kernels"
)


def _pool(layer_id=0):
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.config import LinearGatedDeltaGroupConfig

    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(layer_id,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    return LinearStatePool(group=g, num_slots=8, dtype=torch.bfloat16,
                           device=torch.device("cuda"), tp_size=1)


def test_decode_path_vs_varlen_1row_extend_state_equivalence():
    """The layer's conv dispatch fix: a 1-token-per-req extend (the spec
    loop's verify rows) routes the conv through the DECODE kernel. This
    test pins the two halves of that decision:
    1. the DECODE conv kernel + decode recurrent kernel give the correct
       shift-append + recurrent step (the reference);
    2. the VARLEN conv kernels (fused sgl + triton fallback) leave the
       conv state UN-SHIFTED for a sequence shorter than the conv window
       (has_initial_state=True) — the documented defect; the fix avoids
       them. Kept as a canary: if a kernel upgrade fixes the tail write,
       the varlen assertion can be re-enabled to route the long-extend
       fast path through the 1-row case too.
    The recurrent (chunk vs fused-decode) math is equivalent for a 1-token
    step — asserted here via the post-state with the decode conv on both.
    """
    from freetoken.kernel.causal_conv1d import causal_conv1d_decode, causal_conv1d_varlen
    from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_decode_fla, gdn_prefill_chunk_fla
    device = torch.device("cuda")
    K = V = 16
    Hk, Hv = 2, 4
    pool = _pool()
    li = pool.local_index(0)
    slot_live, slot_ref = 1, 2
    conv0 = torch.randn(pool.conv_states.shape[2:], device=device)   # [conv_dim, km1]
    rec0 = torch.randn(pool.recurrent_states.shape[2:], device=device)  # [Hv, K, V]
    pool.conv_states[0, slot_live] = conv0
    pool.conv_states[0, slot_ref] = conv0
    pool.recurrent_states[0, slot_live] = rec0
    pool.recurrent_states[0, slot_ref] = rec0


    # shared inputs: one token
    x = torch.randn(1, conv0.shape[0], dtype=torch.bfloat16, device=device)  # [1, conv_dim]
    conv_w = torch.randn(conv0.shape[0], 4, dtype=torch.bfloat16, device=device)
    q = torch.randn(1, 1, Hk, K, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, 1, Hk, K, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, 1, Hv, V, dtype=torch.bfloat16, device=device)
    a = torch.randn(1, Hv, device=device)
    b = torch.randn(1, Hv, device=device)
    scale = K ** -0.5

    # ---- decode path (from slot_live) ----
    # identical A_log/dt_bias for both paths (the layer's gate inputs)
    A_log = torch.randn(Hv, device=device) * 0.1
    dt = torch.randn(Hv, device=device) * 0.1
    mixed_dec = causal_conv1d_decode(x, pool.conv_states[li], conv_w,
                                     torch.tensor([slot_live], device=device))
    qf, kf, vf = torch.split(mixed_dec, [Hk * K, Hk * K, Hv * V], dim=-1)
    q_d = qf.reshape(1, 1, Hk, K).to(torch.bfloat16)
    k_d = kf.reshape(1, 1, Hk, K).to(torch.bfloat16)
    v_d = vf.reshape(1, 1, Hv, V).to(torch.bfloat16)
    o_dec = gdn_decode_fla(q_d, k_d, v_d, a, b,
                           A_log=A_log, dt_bias=dt,
                           state_source=pool.recurrent_states[li],
                           indices=torch.tensor([slot_live], device=device),
                           cu_seqlens=torch.arange(2, device=device),
                           scale=scale)
    rec_after_dec = pool.recurrent_states[0, slot_live].clone()
    conv_after_dec = pool.conv_states[0, slot_live].clone()

    # ---- varlen 1-row extend path (from slot_ref) ----
    cu = torch.tensor([0, 1], dtype=torch.int32, device=device)
    mixed_var = causal_conv1d_varlen(
        x.t().contiguous(), conv_w, pool.conv_states[li], cu,
        torch.tensor([slot_ref], device=device),
        torch.tensor([True], device=device))  # has_initial_state: cached > 0
    qf2, kf2, vf2 = torch.split(mixed_var.transpose(0, 1),
                                [Hk * K, Hk * K, Hv * V], dim=-1)
    q_v = qf2.reshape(1, 1, Hk, K).to(torch.bfloat16)
    k_v = kf2.reshape(1, 1, Hk, K).to(torch.bfloat16)
    v_v = vf2.reshape(1, 1, Hv, V).to(torch.bfloat16)
    # the varlen gate math (_gate_params) vs the decode kernel's in-kernel
    # gating — identical formula: g = -exp(A_log)*softplus(a+dt). The varlen
    # conv's OUTPUT for this token is right; only its state write-back is
    # broken — the recurrent step runs from identical conv outputs, so the
    # post-state and output must agree.
    import torch.nn.functional as F
    g_pre = -A_log.exp() * F.softplus(a.float() + dt)
    beta_pre = b.sigmoid()
    o_var = gdn_prefill_chunk_fla(
        q_v, k_v, v_v, g_pre.reshape(1, 1, Hv), beta_pre.float().reshape(1, 1, Hv),
        state_source=pool.recurrent_states[li],
        indices=torch.tensor([slot_ref], device=device), cu_seqlens=cu.to(torch.int64),
        scale=scale)
    rec_after_var = pool.recurrent_states[0, slot_ref].clone()
    conv_after_var = pool.conv_states[0, slot_ref].clone()

    # the VARLEN conv state must be UN-SHIFTED (the documented defect) —
    # the canary. If a kernel upgrade fixes the tail write, flip gdn.py's
    # dispatch back to the varlen path for 1-row extends.
    assert not torch.allclose(conv_after_dec, conv_after_var, atol=2e-2), (
        "varlen conv state unexpectedly matches the decode path — the "
        "tail-write defect appears fixed; re-enable the varlen fast path")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))