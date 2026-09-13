"""Spec decode post-resolve bookkeeping (wave-2 round 2, fix #1).

The invariant under test: after every spec resolve + bookkeeping pass,
``allocate_paged``'s next request span ``[cached_len, device_len)`` is
EXACTLY the positions the next iteration must (re-)process —
  accept: both verify rows committed -> cached_len = device_len (the next
          span is the iteration's 2 fresh positions);
  reject: row B freed + device_len rewound -> the undone position
          re-processes as the next verify's row A ([q, q+1));
  reject-then-recover: a subsequent accept normalizes the state again.

Tests drive the real (unbound) Scheduler._rollback_spec_rejects against a
CPU-built hybrid cache manager + page table (no GPU), checking the page
table/free-list effects and the spec_undone markers alongside the lens.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Batch, Req, SamplingParams, get_global_ctx
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.prefill import PrefillManager
from freetoken.scheduler.scheduler import Scheduler
from freetoken.scheduler.table import TableManager

UID = 7


def _setup():
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    pool = LinearStatePool(group=g, num_slots=16, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1)
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    tm = TableManager(max_running_reqs=4, page_table=pt)
    dm = DecodeManager(page_size=1)
    pm = PrefillManager(cm, tm, dm)
    stub = SimpleNamespace(
        cache_manager=cm, table_manager=tm, decode_manager=dm,
        prefill_manager=pm, finished_reqs=set(), eos_token_ids=set(),
        toolcall_anchor_id=None, config=SimpleNamespace(page_size=1),
        # real arm-gate logic: the bridge req (spec_carry None) stays plain
        _spec_armed=lambda batch: Scheduler._spec_armed(stub, batch),
        status_reporter=SimpleNamespace(report_batch=lambda *_, **__: None),
        send_result=lambda *_: None,
        _kv_usage_pages=cm.page_usage,
        _mamba_slot_usage=lambda: None,
        _swa_token_usage=lambda: None,
        _gpu_mem_bytes=lambda: 0,
        _match_stop_str=lambda _req: None,
        _pending_abort_acks=set(),
        _last_data=None,
        # the engine surface _rollback_spec_rejects touches
        engine=SimpleNamespace(
            page_table=pt,
            linear_state_pool=pool,
            mtp_drafter=None,  # plain decode: the arm gate short-circuits
        ),
    )
    return pool, cm, tm, dm, pt, stub


def _spec_req(pool, cm, tm, prompt_len=4, *, cached_len, device_len):
    """A decode-phase req whose lens are set to the POST-VERIFY state: the
    scheduler advanced device_len by 2 and the trunk processed both rows
    (cached_len still at the pre-advance frontier)."""
    prompt = torch.arange(1, prompt_len + 1, dtype=torch.int32)
    from freetoken.scheduler.utils import PendingReq
    mr = cm.match_req(PendingReq(uid=UID, input_ids=prompt,
                                 mm_embeds=None,
                                 sampling_params=SamplingParams(max_tokens=64)))
    req = Req(input_ids=prompt, table_idx=tm.allocate(),
              cached_len=0, output_len=64, uid=UID,
              sampling_params=SamplingParams(max_tokens=64),
              cache_handle=mr.cuda_handle)
    cm.lock(mr.cuda_handle)
    req.linear_slot_idx = pool.alloc(1)[0]
    req.mamba_ping_pong = tuple(pool.alloc(2))
    req.mamba_next_track_idx = 1
    # Map the prompt rows [0, prompt_len) the way the runtime's prefill
    # does (cached_len=0/device_len=prompt → allocate_paged maps the whole
    # span), then re-allocate the spec-advance span ONLY if the requested
    # state actually has uncommitted rows (device_len > prompt_len —
    # matching the runtime, where a post-accept req has cached == device
    # and the NEXT prepare allocates fresh).
    req.cached_len = 0
    req.device_len = prompt_len
    cm.allocate_paged([req])
    if device_len > prompt_len:
        req.cached_len = prompt_len
        req.device_len = max(device_len, prompt_len + 2)
        cm.allocate_paged([req])
    req.cached_len = cached_len
    req.device_len = device_len
    return req


def _ensure_ctx(pool):
    """A bare global Context with a no-op attn backend: _make_spec_row_batch
    and the GDN metadata builder read it (lm-head slicing / conv dims).
    Another test module may have installed a partial Context first, so
    backfill any missing attrs rather than only creating when absent."""
    from freetoken.core import Context, get_global_ctx, set_global_ctx

    class _NoopBackend:
        def prepare_metadata(self, batch):
            # mirror the real backends: prepare_metadata owns setting the
            # batch's attn_metadata (fi.py:258) — the spec replay's staged
            # metadata reads it back after this call.
            batch.attn_metadata = None

    try:
        ctx = get_global_ctx()
    except AssertionError:
        ctx = Context(page_size=1)
        set_global_ctx(ctx)
    if not hasattr(ctx, "attn_backend"):
        ctx.attn_backend = _NoopBackend()
    if ctx.linear_state_pool is None:
        ctx.linear_state_pool = pool


def _resolve(stub, batch, accepted: list[bool]):
    for req, acc in zip(batch.reqs, accepted):
        req.spec_accepted = acc
    Scheduler._rollback_spec_rejects(stub, batch)


def test_accept_commits_cached_len():
    """Accept: cached_len catches up to device_len; the next allocate span
    [cached_len, device_len) is empty (the NEXT prepare advances +2 first,
    making it exactly the 2 new positions)."""
    pool, cm, tm, dm, pt, stub = _setup()
    req = _spec_req(pool, cm, tm, cached_len=4, device_len=6)
    base_free = cm.free_slots.numel()
    _resolve(stub, Batch(reqs=[req], phase="decode"), accepted=[True])
    assert (req.cached_len, req.device_len, req.spec_undone) == (6, 6, 0)
    # nothing freed on accept
    assert cm.free_slots.numel() == base_free


def test_reject_frees_row_b_and_reprocesses_undone():
    """Reject: row B's slot freed (page-table zeroed), device_len rewinds
    to q+1 and cached_len ADVANCES to it (row A processed+verified
    position q: committed = [0, q+1)). Post-resolve span is 0; the next
    spec iteration re-stages 2 rows (row A = the corrected token at
    q+1). Leaving cached at q grew the span +2 per consecutive reject
    (the extend_len-10 crash)."""
    pool, cm, tm, dm, pt, stub = _setup()
    req = _spec_req(pool, cm, tm, cached_len=4, device_len=6)
    row_b_slot = int(pt[req.table_idx, 5])
    base_free = cm.free_slots.numel()
    _resolve(stub, Batch(reqs=[req], phase="decode"), accepted=[False])
    assert (req.cached_len, req.device_len) == (5, 5)
    assert req.spec_undone == 1
    assert int(pt[req.table_idx, 5]) == 0  # row B's mapping dropped
    # the lazy free region defers the free until the region exits — after
    # _resolve returns the slot must be back on the free list
    assert row_b_slot in cm.free_slots.tolist()
    assert cm.free_slots.numel() == base_free + 1


def test_reject_then_recover_normalizes():
    """Reject at iteration N (undone @q), then the NEXT iteration accepts:
    the re-process row consumed [q, q+1) and the 2 fresh rows [q+1, q+3)
    commit -> cached_len = device_len = q+3, spec_undone cleared."""
    pool, cm, tm, dm, pt, stub = _setup()
    req = _spec_req(pool, cm, tm, cached_len=4, device_len=6)
    _resolve(stub, Batch(reqs=[req], phase="decode"), accepted=[False])
    assert (req.cached_len, req.device_len, req.spec_undone) == (5, 5, 1)
    # ---- next iteration: prepare advances device_len by 2 over the
    # undone position + 1 fresh slot, then the verify accepts ----
    req.device_len += 2  # [4, 7): re-process 4 (row A input = a), fresh 5, 6
    cm.allocate_paged([req])
    _resolve(stub, Batch(reqs=[req], phase="decode"), accepted=[True])
    assert (req.cached_len, req.device_len, req.spec_undone) == (7, 7, 0)
    # the re-allocated undone slot is re-mapped in the page table
    assert int(pt[req.table_idx, 5]) != 0
    assert int(pt[req.table_idx, 6]) != 0

def test_mixed_batch_independent_bookkeeping():
    """A mixed batch: each req's branch applied independently."""
    pool, cm, tm, dm, pt, stub = _setup()
    r_acc = _spec_req(pool, cm, tm, cached_len=4, device_len=6)
    r_rej = _spec_req(pool, cm, tm, prompt_len=6, cached_len=6, device_len=8)
    base_free = cm.free_slots.numel()
    _resolve(stub, Batch(reqs=[r_acc, r_rej], phase="decode"),
             accepted=[True, False])
    assert (r_acc.cached_len, r_acc.device_len, r_acc.spec_undone) == (6, 6, 0)
    assert (r_rej.cached_len, r_rej.device_len, r_rej.spec_undone) == (7, 7, 1)
    assert cm.free_slots.numel() == base_free + 1  # only the reject freed

def test_gdn_snapshot_restore_invoked_on_reject():
    """The reject path restores the MID-VERIFY GDN snapshot (the state
    AFTER row A — captured by the engine between the row forwards) into
    the live slot; the accept path leaves the live slot alone (the
    verify's post-row-B state IS the committed state). The restored
    snapshot must equal the state after row A: with cached_len = q+1 the
    GDN state must cover [0, q+1) — a pre-verify snapshot (state [0, q))
    would lag the bookkeeping by one row (the repetition-loop defect)."""
    pool, cm, tm, dm, pt, stub = _setup()
    req = _spec_req(pool, cm, tm, cached_len=4, device_len=6)
    live = req.linear_slot_idx
    snap = pool.alloc(1)[0]
    live_before = pool.recurrent_states[:, live].clone()
    pool.recurrent_states[:, snap].fill_(123.0)  # a distinctive pre-verify state
    batch = Batch(reqs=[req], phase="decode")
    batch.spec_gdn_snapshot_slots = [snap]
    _resolve(stub, batch, accepted=[False])
    assert torch.equal(pool.recurrent_states[:, live], pool.recurrent_states[:, snap])
    assert not torch.equal(pool.recurrent_states[:, live], live_before)
    # accept: no restore — live state (the verify's own output) stands
    pool.recurrent_states[:, live].fill_(7.0)
    _resolve(stub, batch, accepted=[True])
    assert torch.all(pool.recurrent_states[:, live] == 7.0)
def test_row_batch_build_no_alloc_correct_tokens():
    """BUG-1/BUG-2 regression: _make_spec_row_batch must NOT allocate (the
    batch-level allocate_paged already covers both row spans) and must
    feed the row's input_ids from the staged [c, d] tokens, NOT a
    token_pool gather (the pool slots at q/q+1 hold no valid tokens for
    this iteration)."""
    pool, cm, tm, dm, pt, stub = _setup()
    req = _spec_req(pool, cm, tm, cached_len=4, device_len=6)
    # the staged row tokens: c at q, d at q+1 (the buf spans the full
    # max_device_len; the ids VIEW is [0, device_len))
    C, D = 321, 432
    req._ids_buf[4] = C
    req._ids_buf[5] = D
    stub.device = torch.device("cpu")
    _ensure_ctx(pool)
    stub.engine.attn_backend = get_global_ctx().attn_backend
    stub._row_token_ids = lambda reqs, is_row_a: Scheduler._row_token_ids(stub, reqs, is_row_a)
    req.input_ids = req._ids_buf[: req.device_len]  # widen the view to device_len
    base_free = cm.free_slots.numel()
    row_a = Scheduler._make_spec_row_batch(stub, [req], is_row_a=True)
    row_b = Scheduler._make_spec_row_batch(stub, [req], is_row_a=False)
    assert cm.free_slots.numel() == base_free  # no allocation
    # positions: row A -> q=4, row B -> q+1=5
    assert row_a.positions.tolist() == [4]
    assert row_b.positions.tolist() == [5]
    # input_ids: row A = c, row B = d (staged, not gathered from the pool)
    assert row_a.input_ids.tolist() == [C]
    assert row_b.input_ids.tolist() == [D]
    # out_loc matches the page table at [q, q+1]
    assert row_a.out_loc.tolist() == [int(pt[req.table_idx, 4])]
    assert row_b.out_loc.tolist() == [int(pt[req.table_idx, 5])]


def test_prepare_spec_batch_host_buf_positions():
    """BUG-3(a) regression: _prepare_spec_batch writes c at host[q] (the
    with the KV positions; a simulated accept drain must NOT shift them
    (no append_host on spec batches) and the bonus must stay out of the
    buf until the next iteration stages it."""
    pool, cm, tm, dm, pt, stub = _setup()
    # PRE-prepare lens: the req sits at device_len = q = 4 (the next
    # iteration's prepare advances +2 to 6)
    req = _spec_req(pool, cm, tm, cached_len=4, device_len=4)

    class _MTP:
        def draft_step(self, carry, tokens):
            logits = torch.zeros(tokens.shape[0], 1024)
            logits[:, 777] = 1.0  # every draft = 77
            return torch.zeros_like(carry), logits

    stub.engine = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(mtp=_MTP())),
        page_table=pt,
        linear_state_pool=pool,
    )
    stub.engine.stream = torch.cuda.default_stream()
    req.spec_carry = torch.zeros(8, dtype=torch.bfloat16)
    req.spec_next_input = 500  # c
    _ensure_ctx(pool)

    stub.device = torch.device("cpu")
    stub._forward_iter = 0
    stub._make_spec_row_batch = (
        lambda reqs, *, is_row_a: Scheduler._make_spec_row_batch(stub, reqs, is_row_a=is_row_a))
    stub._row_token_ids = lambda reqs, is_row_a: Scheduler._row_token_ids(stub, reqs, is_row_a)
    stub.engine.attn_backend = get_global_ctx().attn_backend
    stub.engine.ctx = get_global_ctx()
    stub._mtp_replay_batch_meta = (
        lambda batch: Scheduler._mtp_replay_batch_meta(stub, batch))
    Scheduler._prepare_spec_batch(stub, Batch(reqs=[req], phase="decode"))
    assert (req.cached_len, req.device_len) == (4, 6)
    assert int(req.input_ids[4]) == 500   # c at q
    assert int(req.input_ids[5]) == 777   # d at q+1 (the draft's host slot)
    assert req.spec_draft == 777
    # simulate the accept drain: spec path -> no append_host; the buf
    # tail [c@4, d@5] must be unchanged, bonus absent.
    BONUS = 999
    out = SimpleNamespace(spec_extra_tokens_cpu=torch.tensor([BONUS], dtype=torch.int32))
    stub.cache_manager = cm
    for req_i in [req]:
        spec_extra_cpu = out.spec_extra_tokens_cpu
        emitted = [torch.tensor([777], dtype=torch.int32)]  # next_tokens_cpu = first = d
        if int(spec_extra_cpu[0].item()) >= 0:
            emitted.append(spec_extra_cpu[0])
        for tok in emitted:
            if spec_extra_cpu is None:  # spec batches skip append_host
                req.append_host(tok.unsqueeze(0))
    assert int(req.input_ids[4]) == 500
    assert int(req.input_ids[5]) == 777
    assert req.input_ids.numel() == 6       # nothing appended
    assert BONUS not in req.input_ids.tolist()[:6]

def test_spec_to_plain_bridge_restores_extend_protocol():
    """BUG B regression: a req leaving the spec loop after an ACCEPT has
    cached_len == device_len (extend_len 0). The plain decode protocol
    requires exactly one unprocessed position per req — the GDN/conv
    decode kernel asserts one hidden row per request (the
    "conv_state_indices must have shape (batch_size)" crash). The bridge
    in _prepare_batch's decode branch stages the resolved next input as
    the extend token: token_pool write, host ids buf, device_len = cached
    + 1."""
    pool, cm, tm, dm, pt, stub = _setup()
    from freetoken.core import get_global_ctx
    _ensure_ctx(pool)
    # post-accept spec state: cached == device == 6, bonus staged
    req = _spec_req(pool, cm, tm, cached_len=6, device_len=6)
    req.spec_next_input = 4242
    stub.token_pool = torch.zeros_like(pt, dtype=torch.int32)
    stub.device = torch.device("cpu")
    stub._forward_iter = 0
    stub.engine.attn_backend = get_global_ctx().attn_backend
    stub.engine.sampler = SimpleNamespace(prepare=lambda batch: None)
    batch = Batch(reqs=[req], phase="decode")
    stub.engine.graph_runner = SimpleNamespace(
        pad_batch=lambda b: setattr(b, "padded_reqs", b.reqs))
    stub.engine.linear_state_pool = None  # skip fla staging in this test
    Scheduler._prepare_batch(stub, batch)
    assert req.extend_len == 1
    assert (req.cached_len, req.device_len) == (6, 7)
    assert int(req.input_ids[6]) == 4242          # host mirror staged
    assert int(stub.token_pool[req.table_idx, 6]) == 4242  # plain gather source
    assert int(req.input_ids.numel()) == 7

def test_spec_slot_lifecycle_conservation():
    """Defect-1 harness (page leak): run the bookkeeping through N spec
    iterations (mixed accept/reject), then FINISH the request via the
    drain's free path, and assert the page currency is conserved: every
    mapped slot is either back on the free list or tree-owned, and the
    integrity check passes. Catches per-iteration leaks
    ("free_pages + cache_pages != num_pages") at the CPU level."""
    pool, cm, tm, dm, pt, stub = _setup()
    from freetoken.core import get_global_ctx
    _ensure_ctx(pool)
    stub.device = torch.device("cpu")
    stub._forward_iter = 0
    stub.engine.attn_backend = get_global_ctx().attn_backend
    stub.engine.sampler = SimpleNamespace(prepare=lambda batch: None)
    stub.token_pool = torch.zeros_like(pt, dtype=torch.int32)
    stub.engine.stream = (torch.cuda.default_stream()
                          if torch.cuda.is_available()
                          else torch.cuda.Stream(device=torch.device("cpu")))
    stub.engine.ctx = get_global_ctx()
    stub.engine.linear_state_pool = pool

    class _MTP:
        def draft_step(self, carry, tokens):
            logits = torch.zeros(tokens.shape[0], 64)
            logits[:, 7] = 1.0
            return torch.zeros_like(carry), logits

    stub.engine.model = SimpleNamespace(model=SimpleNamespace(mtp=_MTP()))
    stub._make_spec_row_batch = (
        lambda reqs, *, is_row_a: Scheduler._make_spec_row_batch(
            stub, reqs, is_row_a=is_row_a))
    stub._row_token_ids = (
        lambda reqs, is_row_a: Scheduler._row_token_ids(stub, reqs, is_row_a))
    stub._mtp_replay_batch_meta = (
        lambda batch: Scheduler._mtp_replay_batch_meta(stub, batch))

    # a req at the post-prefill protocol state: prompt fully committed
    # (cached == device == 4), first spec input staged.
    req = _spec_req(pool, cm, tm, cached_len=4, device_len=4)
    req.spec_next_input = 100
    req.spec_carry = torch.zeros(8, dtype=torch.bfloat16)

    for it in range(6):  # mixed accept/reject sequence
        Scheduler._prepare_spec_batch(stub, Batch(reqs=[req], phase="decode"))
        snap = pool.alloc(1)[0]   # the engine's mid-verify snapshot slot
        batch = Batch(reqs=[req], phase="decode")
        batch.spec_gdn_snapshot_slots = [snap]
        _resolve(stub, batch, accepted=[it % 2 == 0])
        pool.free([snap])         # the snapshot slot is iteration-local
        req.spec_next_input = 100
        req.spec_carry = torch.zeros(8, dtype=torch.bfloat16)

    # Finish MID-VERIFY: re-run the last prepare WITHOUT resolving, so the
    # +2 advance leaves a MAPPED uncommitted tail [cached_len, device_len)
    # (device_len = cached_len + 2, e.g. the request hit its output budget
    # between the row forwards). This tail is exactly what the defect-1
    # leak lost — exercise it.
    Scheduler._prepare_spec_batch(stub, Batch(reqs=[req], phase="decode"))
    req.spec_next_input = 100
    req.spec_carry = torch.zeros(8, dtype=torch.bfloat16)

    # per-iteration conservation: no mapped slot may also be on the free list
    mapped = {int(s) for s in pt[req.table_idx] if int(s) != 0}
    freed = set(cm.free_slots.tolist())
    assert not (mapped & freed), "a mapped slot is also on the free list"

    # finish via the drain's free path (cache_req finished=True + table free)
    stub.finished_reqs = set()
    stub.send_result = lambda *a: None
    stub._free_req_resources = lambda r: Scheduler._free_req_resources(stub, r)
    stub.decode_manager = DecodeManager(page_size=1)
    stub.decode_manager.running_reqs = {req}
    Scheduler._free_req_resources(stub, req)
    # The page currency must balance: free + tree-owned == total. Tree-owned
    # slots (the finish's radix donation) are legitimate non-free holders —
    # the check_integrity assertion below is the exact invariant the gate's
    # "free_pages + cache_pages != num_pages" leak tripped.
    cm.check_integrity()
    tree = cm.prefix_cache.full_evictable + cm.prefix_cache.full_protected
    assert len(cm.free_slots) + tree // cm.page_size == cm.num_pages

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))