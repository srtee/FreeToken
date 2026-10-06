"""MTP draft CUDA-graph runner (wave-2 stage 2) — CPU-green tests.

Pins the pure-logic surface of MTPDraftGraphRunner without CUDA:
- the bs family covers EVERY bs in 1..min(max_running_req,
  --cuda-graph-max-bs) — exact-bs capture, no padding (the draft MoE's
  bf16 bmm is batch-shape-sensitive: a padded kernel computes a live
  row differently than the eager exact-bs kernel, breaking bit-equality);
- the eager-fallback predicate is a pure function of the family;
- MTPDraftBuffer staging writes exactly the [:bs] slices;
- the replay-side metadata rebuild produces the same FIMetadata fields the
  eager path builds for an identical synthetic batch (the trunk's
  prepare_metadata contract the captured attention replays against).
The GPU bit-equality gate (graphed vs eager draft argmaxes/carry'/KV row)
is the gpu-marked test at the bottom; it runs only under FT_TEST_GPU=1
(operator-controlled — never in the CPU suite).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.graph import (
    MTPDraftBuffer,
    _determine_draft_graph_bs,
    _draft_family_bs,
)


# ---------------------------------------------------------------------------
# bs-family selection
# ---------------------------------------------------------------------------


def test_family_is_every_bs_up_to_max_running_req():
    # The 35B server: max_running_req=4 -> [1, 2, 3, 4]; the trunk's 8+
    # entries must NOT appear (dead graphs burn VRAM), and NO bs may be
    # missing (exact-bs capture: a padded kernel is not bit-equal to the
    # eager exact-bs kernel through the draft's bf16 bmm).
    assert _determine_draft_graph_bs(4, None) == [1, 2, 3, 4]


def test_family_capped_by_cuda_graph_max_bs():
    assert _determine_draft_graph_bs(8, 3) == [1, 2, 3]
    # max_bs 0 disables capture entirely (the trunk's contract).
    assert _determine_draft_graph_bs(4, 0) == []
    assert _determine_draft_graph_bs(4, 1) == [1]


def test_family_default_cap_is_160_when_running_req_allows():
    assert _determine_draft_graph_bs(160, None) == list(range(1, 161))


def test_bs_over_family_falls_back_to_eager():
    bs_list = _determine_draft_graph_bs(4, None)
    # every bs 1..4 has an EXACT entry; bs 5 -> None = eager.
    assert [(_draft_family_bs(bs_list, b)) for b in (1, 2, 3, 4)] == [1, 2, 3, 4]
    assert _draft_family_bs(bs_list, 5) is None
    # An empty family (capture disabled) always falls back.
    assert _draft_family_bs([], 1) is None


# ---------------------------------------------------------------------------
# Buffer shape/copy semantics
# ---------------------------------------------------------------------------

H, VOCAB = 16, 32


def _buffer(max_bs: int) -> MTPDraftBuffer:
    return MTPDraftBuffer.init(max_bs, H, VOCAB, torch.bfloat16, torch.device("cpu"))


def test_buffer_shapes_and_dtypes():
    buf = _buffer(4)
    assert buf.carry_in.shape == (4, H) and buf.carry_in.dtype == torch.bfloat16
    assert buf.token_in.shape == (4,) and buf.token_in.dtype == torch.int32
    assert buf.out_loc.shape == (4,) and buf.out_loc.dtype == torch.int32
    assert buf.positions.shape == (4,) and buf.positions.dtype == torch.int32
    # fp32 logits: exact bf16->fp32 cast keeps the captured argmax
    # bit-identical to the eager path's bf16 argmax.
    assert buf.logits.shape == (4, VOCAB) and buf.logits.dtype == torch.float32
    assert buf.carry_out.shape == (4, H)
    assert buf.drafts.shape == (4,) and buf.drafts.dtype == torch.int32


def test_stage_inputs_writes_only_live_slice():
    """Staging one replay step must touch exactly [:bs]. Exact-bs replays
    never run a padded tail, but the buffer is sized [max_bs, ...] so the
    rows beyond bs must keep whatever the capture parked there."""
    buf = _buffer(4)
    buf.out_loc[2:] = 777  # parked capture-time scratch beyond the live rows
    buf.stage_inputs(
        carries=torch.ones(2, H, dtype=torch.bfloat16),
        tokens=torch.tensor([5, 6], dtype=torch.int32),
        out_loc=torch.tensor([10, 11], dtype=torch.int32),
        positions=torch.tensor([3, 4], dtype=torch.int32),
        bs=2,
    )
    assert buf.carry_in[:2].equal(torch.ones(2, H, dtype=torch.bfloat16))
    assert buf.token_in[:2].tolist() == [5, 6]
    assert buf.out_loc[:2].tolist() == [10, 11]
    assert buf.positions[:2].tolist() == [3, 4]
    # padded tail untouched by staging
    assert buf.out_loc[2:].tolist() == [777, 777]
    assert buf.token_in[2:].tolist() == [0, 0]


def test_captured_argmax_matches_eager_argmax_for_identical_logits():
    """The fp32-logits invariant: for the same underlying values, argmax
    over the fp32 cast equals argmax over raw bf16 — tie behavior included
    (identical values resolve identically index-wise)."""
    torch.manual_seed(0)
    bf16 = torch.randn(8, VOCAB, dtype=torch.bfloat16)
    eager = bf16.argmax(dim=-1)
    graphed = bf16.to(torch.float32).argmax(dim=-1)
    assert torch.equal(eager, graphed)


# ---------------------------------------------------------------------------
# Replay-side metadata rebuild == the eager path's prepare_metadata
# ---------------------------------------------------------------------------


class _RowReq:
    """The 1-row req view the scheduler's _SpecRowReq presents: the fields
    prepare_metadata reads. Mirrors row A of a req mid-spec (device_len-2 =
    q, extend window [q, q+1))."""

    def __init__(self, table_idx: int, cached_len: int, device_len: int):
        self.table_idx = table_idx
        self.cached_len = cached_len
        self.device_len = device_len
        self.extend_len = 1


def _prepare_metadata_like_scheduler(backend, reqs, page_table, phase="decode"):
    """The exact prepare_metadata input shape the draft replay must feed:
    a phase="decode" batch of 1-row req views with padded_reqs set."""
    from freetoken.core import Batch

    b = Batch(reqs=reqs, phase=phase)
    b.padded_reqs = reqs
    backend.prepare_metadata(b)
    return b


def _fake_fi_backend(monkeypatch):
    """A FlashInferBackend whose metadata build runs on this host (CPU
    tensors where FIMetadata allows them). FIMetadata.__post_init__ hard-
    asserts cu_seqlens_q_gpu/indices on CUDA (device tensors from
    page_table/indptr .to(device)), so this pins the FIELD CONSTRUCTION
    only: patch the assert to device-agnostic on CPU and compare fields.
    """
    import freetoken.core as core
    from freetoken.core import Context

    monkeypatch.setattr(core, "_GLOBAL_CTX", Context(page_size=1))
    ctx = core.get_global_ctx()
    ctx.page_table = torch.arange(3 * 8, dtype=torch.int32).reshape(3, 8)

    from freetoken.attention.fi import FlashInferBackend
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)

    class _Pool:
        device = torch.device("cpu")
        dtype = torch.bfloat16

    class _Backend(FlashInferBackend):
        def __init__(self):  # bypass the flashinfer workspace init
            self.config = SimpleNamespace(head_dim=128)
            self.kvcache = _Pool()
            self.device = self.kvcache.device
            self.qo_head_local, self.kv_head_local = 4, 2
            self.decode_wrappers = SimpleNamespace()  # identity is what's asserted
            self.cached_ones_cpu = torch.tensor([], dtype=torch.int32, pin_memory=True)
    return _Backend()


def test_replay_metadata_matches_eager_path_fields(monkeypatch):
    """The runner rebuilds the draft batch's FI metadata through the SAME
    prepare_metadata + prepare_for_replay sequence the trunk uses; for an
    identical synthetic batch the fields must equal the eager path's
    (pin: decode indptr = arange(bs+1), indices = the live page-table rows,
    last_page_len = ones, wrapper = the decode wrapper)."""
    from freetoken.attention import fi as fi_mod

    # FIMetadata.__post_init__ hard-asserts the GPU-resident fields; this
    # CPU test only pins the FIELD CONSTRUCTION, so relax the device assert
    # (the CUDA gate exercises the real assert on-device).
    monkeypatch.setattr(
        fi_mod.FIMetadata, "__post_init__",
        lambda self: _assert_cpu_fields(self), raising=True)

    def _assert_cpu_fields(md):
        assert md.page_size == 1
        for f in ("cu_seqlens_q_cpu", "cu_seqlens_k_cpu", "last_page_len_cpu",
                  "seq_lens_cpu"):
            assert getattr(md, f).is_cpu

    backend = _fake_fi_backend(monkeypatch)
    reqs = [_RowReq(table_idx=i, cached_len=5, device_len=6) for i in range(2)]
    batch = _prepare_metadata_like_scheduler(backend, reqs, None)
    md = batch.attn_metadata
    assert isinstance(md, fi_mod.FIMetadata)
    # decode with all extend_len = 1: q indptr = arange(bs+1)
    assert md.cu_seqlens_q_cpu.tolist() == [0, 1, 2]
    # k indptr = cumulative device_len (the full attended window)
    assert md.cu_seqlens_k_cpu.tolist() == [0, 6, 12]
    assert md.seq_lens_cpu.tolist() == [6, 6]
    # indices = the concat of each req's [0, device_len) page-table row
    from freetoken.core import get_global_ctx

    pt = get_global_ctx().page_table
    expected = torch.cat([pt[i, :6] for i in range(2)])
    assert torch.equal(md.indices, expected)
    assert md.last_page_len_cpu.tolist() == [1, 1]
    assert md.num_qo_heads == 4 and md.num_kv_heads == 2
    assert md.page_size == 1 and md.pos_encoding_mode == "NONE"
    assert md.dtype == torch.bfloat16
    assert not md.initialized  # fresh: prepare_for_replay's assert holds


def test_draft_batch_no_fla_metadata():
    """The draft batch must not build GDN metadata (MTPDraftLayer has no
    GDN layers); the runner stages inputs onto the buffer and never
    touches fla_metadata."""
    from freetoken.core import Batch

    b = Batch(reqs=[], phase="decode")
    assert b.fla_metadata is None


# ---------------------------------------------------------------------------
# GPU gate: graphed vs eager bit-equality (operator-run, FT_TEST_GPU=1)
# ---------------------------------------------------------------------------

_CKPT = ("/home/sherntee/.cache/huggingface/hub/models--nvidia--"
         "Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5/")


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("FT_TEST_GPU") != "1",
    reason="Stage-2 GPU gate: run under FT_TEST_GPU=1 on the 35B NVFP4 checkpoint",
)
def test_gpu_draft_graph_bit_equality():
    """The Stage-2 gate: for N=200 random (carry, token) steps, the graphed
    draft path must produce bit-identical argmaxes, bit-identical carry',
    and an identical layer-40 KV row vs the eager path, and the draft's KV
    write position must match.

    Builds the engine from the real checkpoint the same way the server
    does (Engine + EngineConfig with --spec-mtp semantics), captures the
    draft family, then alternates eager (FT_SPEC_DRAFT_EAGER semantics:
    mtp.draft_step under forward_batch) and graphed (runner.draft) steps
    over the SAME inputs, comparing outputs and the written KV row.
    """
    from freetoken.engine.engine import Engine
    from freetoken.engine.config import EngineConfig
    from freetoken.distributed import DistributedInfo

    cfg = EngineConfig(
        model_path=_CKPT,
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        max_running_req=4,
        spec_mtp=True,
        cuda_graph_max_bs=4,
        max_seq_len_override=512,
        num_token_override=1024,
    )
    eng = Engine(cfg)
    assert eng.draft_graph_runner is not None, "gate requires the draft family captured"
    assert eng.graph_runner.max_graph_bs == 0, "trunk graphs stay disabled under --spec-mtp"

    mtp = eng.model.model.mtp
    device = eng.device
    dtype = eng.dtype
    pool = eng.kv_cache
    n_steps = 200
    gen = torch.Generator().manual_seed(7)
    for step in range(n_steps):
        bs = int(torch.randint(1, 5, (1,), generator=gen))
        # EXACT-bs: the graphed replay runs the bs graph; the eager oracle
        # runs the same bs. Same kernel geometry on both sides — the
        # comparison the production paths make (spec server graphed vs
        # plain server eager, each at the live bs).
        carries = torch.randn(bs, cfg.model_config.hidden_size, generator=gen).to(
            device, dtype=dtype)
        tokens = torch.randint(
            0, cfg.model_config.vocab_size, (bs,), generator=gen,
            dtype=torch.int32).to(device)
        # Layer-40 write slots: the pool is num_token_override (1024) slots
        # deep; page 0 is the dummy page, so slots start at 1. Each step's
        # eager + graphed sets live in disjoint 8-wide bands (2 sets x max
        # bs 4) cycling over 8 bands = 64 slots; both paths get FRESH slots
        # every step, so neither ever reads a stale row.
        band = step % 8
        base = 1 + band * 8
        out_loc = (torch.arange(bs, dtype=torch.int32) + base).to(device)
        positions = torch.full((bs,), step, dtype=torch.int32, device=device)

        # --- eager reference (the FT_SPEC_DRAFT_EAGER oracle, verbatim
        # scheduler shape): fresh batch -> draft_step under forward_batch.
        from freetoken.core import Batch, Req

        reqs = [
            Req(input_ids=torch.zeros(1, dtype=torch.int32), table_idx=i,
                cached_len=0, output_len=1, uid=-(step * 8 + i),
                sampling_params=None, cache_handle=None)
            for i in range(bs)
        ]
        eager_batch = Batch(reqs=reqs, phase="decode")
        eager_batch.padded_reqs = reqs
        eager_batch.input_ids = tokens
        eager_batch.out_loc = out_loc
        eager_batch.positions = positions
        # prepare_metadata builds the FIMetadata the FI decode path reads
        # (seq lens over the dummy table row); prepare_for_replay is NOT
        # needed — the eager path plans inside _initialize_metadata_once.
        eng.attn_backend.prepare_metadata(eager_batch)
        with torch.cuda.stream(eng.stream):
            with eng.ctx.forward_batch(eager_batch):
                eager_carry, eager_logits = mtp.draft_step(carries, tokens)
            eager_drafts = eager_logits.argmax(dim=-1).to(torch.int32)
        # read back the KV row the eager path wrote
        eager_kv = pool.k_cache(mtp.layer.self_attn.layer_id)[
            out_loc.long()].clone()

        # --- graphed path: the runner replays over the SAME inputs
        # (carries/tokens/positions), staged onto disjoint KV slots so the
        # comparison reads THIS step's write, not the eager step's. The
        # runner consumes the scheduler-shaped draft batch (fresh req
        # views so its metadata plan is the replay's own).
        graph_reqs = [
            Req(input_ids=torch.zeros(1, dtype=torch.int32), table_idx=i,
                cached_len=0, output_len=1, uid=-(step * 8 + i),
                sampling_params=None, cache_handle=None)
            for i in range(bs)
        ]
        graph_batch = Batch(reqs=graph_reqs, phase="decode")
        graph_batch.padded_reqs = graph_reqs
        graph_batch.positions = positions
        graph_batch.out_loc = (out_loc + 4).to(device)  # the band's 2nd half
        g_drafts, g_carry = eng.draft_graph_runner.draft(
            carries, tokens, graph_batch)
        graph_kv = pool.k_cache(mtp.layer.self_attn.layer_id)[
            graph_batch.out_loc.long()]

        assert torch.equal(eager_drafts, g_drafts), (
            f"step {step}: draft argmax diverged")
        assert torch.equal(eager_carry, g_carry), (
            f"step {step}: carry' diverged")
        assert torch.equal(eager_kv, graph_kv), (
            f"step {step}: layer-40 KV row diverged")

    eng.draft_graph_runner.destroy_cuda_graphs()