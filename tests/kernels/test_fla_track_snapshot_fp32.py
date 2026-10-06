"""Hybrid-radix GDN/KDA boundary snapshots must be fp32-exact.

The ×CHUNK boundary snapshot a prefix-hit continuation resumes from must
be the BIT-EXACT fp32 state the kernel holds in registers at the boundary
-- not the bf16-rounded row the ``h`` buffer stores. A bf16 snapshot made
radix-hit continuations diverge from fresh prefills: every near-tie token
at the restore boundary could flip (observed as greedy byte-instability
on the first reuse of every long prompt; codec-independent, spec-independent
-- docs/tcq-baseline-numbers.md, 2026-09-21 matrix).

Property under test: the ``h_track`` side buffer written by the kernel
equals, bit-for-bit, the same kernel's boundary rows when ``h`` is stored
in fp32 (lossless). If either kernel ever rounds the tracked state (a
regression to the bf16 path), this fails.

Requires a GPU + triton (the vendored kernels are triton); skipped otherwise.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU-only: exercises the triton FLA kernels"
)


def _tiny_gdn_inputs(T=130, Hg=2, Hv=4, K=32, V=32, device="cuda"):
    g = torch.device(device)
    torch.manual_seed(7)
    k = torch.randn(1, T, Hg, K, dtype=torch.bfloat16, device=g)
    w = torch.randn(1, T, Hv, K, dtype=torch.bfloat16, device=g)
    u = torch.randn(1, T, Hv, V, dtype=torch.bfloat16, device=g)
    g_log = -torch.rand(1, T, Hv, dtype=torch.float32, device=g).log1p()
    cu = torch.tensor([0, T], dtype=torch.long, device=g)
    # the kernel loads initial_state_indices unconditionally (production
    # always passes the state pool); USE_INITIAL_STATE=False keeps it unread.
    state = torch.zeros(1, Hv, V, K, dtype=torch.float32, device=g)
    indices = torch.zeros(1, dtype=torch.int32, device=g)
    return k, w, u, g_log, cu, state, indices


def test_gdn_track_snapshot_bit_equals_fp32_h_rows():
    from freetoken.kernel.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h

    k, w, u, g_log, cu, state, indices = _tiny_gdn_inputs()
    # boundary after chunk 1 (row 1): the kernel stores h[i_t] = the state
    # ENTERING chunk i_t, so row 1 is the state after the first 64 tokens --
    # exactly the tracked boundary. pairs = (h_row, dst pool slot).
    pairs = torch.tensor([[1, 7]], dtype=torch.long, device=k.device)

    h_bf16, _, h_track = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g_log, cu_seqlens=cu, track_pairs=pairs,
        initial_state=state.clone(), initial_state_indices=indices)
    h_fp32, _, _ = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g_log, cu_seqlens=cu, track_pairs=pairs,
        h_dtype=torch.float32,
        initial_state=state.clone(), initial_state_indices=indices)

    assert h_track.dtype == torch.float32
    # bit-exact: h_track must carry the registers verbatim (fp32 ground truth),
    # NOT the bf16-rounded h row. Rows are [H, V, K] in both buffers.
    assert torch.equal(h_track[0], h_fp32[0, 1]), (
        "tracked boundary state diverges from the fp32 h row: the snapshot "
        "is being rounded again somewhere")
    assert not torch.equal(h_track[0], h_bf16[0, 1].float()), (
        "sanity: the bf16 h row must differ from the fp32 state for this "
        "test to defend anything (inputs too benign)")
    # tracking must not perturb the main compute path
    h_bf16_plain, _, ht_none = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g_log, cu_seqlens=cu, track_pairs=None,
        initial_state=state.clone(), initial_state_indices=indices)
    assert torch.equal(h_bf16, h_bf16_plain)
    assert ht_none is None
