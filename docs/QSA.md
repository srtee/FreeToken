# Speculative decode on QSA models (page-64 pools) — plan

**Status: proposed, not started.** The MTP port itself is complete and serving
(see `docs/qwen38-mtp-verification.md`); this plan covers the one remaining
lever: making the spec arm actually engage on QSA-backed models
(Qwen3.8-Flash-Next), where it currently silently degrades to plain decode.

## Why the arm is gated off

The stage-1 gate (`scheduler.py:_spec_armed`) requires `eng.ctx.page_size == 1`:

```python
# scheduler.py:973
and eng.ctx.page_size == 1
```

The QSA backend declares `page_sizes=(64,)` (`attention/__init__.py:142`), so
the engine coerces page_size to 64 at config time — every decode batch takes
the plain path. The comment on the gate ("reject frees mid-page slot
otherwise") is the whole reason: **the reject rollback frees individual token
slots**, and a page-64 pool owns memory in 64-token pages.

## What the reject path actually does today (page-1 pool)

`Scheduler._rollback_spec_rejects` (`scheduler.py:1031`), per rejected req:

1. `cm._free(page_table[table_idx, head:tail])` — free the unverified draft
   rows' slots; zero their page-table entries so the next `allocate_paged`
   re-maps them.
2. GDN snapshot restore (`pool.copy_from(snapshot_slot_lists[i][k], ...)`) —
   GDN-only, irrelevant for QSA (the MTP layer is QSA, the trunk is
   36 GDN + 12 QSA; the snapshot machinery is generic).
3. Rewind `device_len = cached_len = q + k + 1`.

Also `_prepare_spec_batch` advances `device_len` by n+1 and allocates page
slots for the whole draft span *up front* (`cm.allocate_paged`).

## Why page-64 is not actually a blocker

Key insight: a reject rolls back a contiguous **tail** of the req's KV, and a
page-64 pool hands out pages per request — the req's tail slots are the tail of
the req's last partial page, then whole pages. `cm._free` operates on page
indices from the page table. The page-table entries `page_table[table_idx,
head:tail]` for the *token* range `head..tail` hold the **slot ids** backing
those tokens (one per token, page_size entries per page). Freeing a tail
span of tokens therefore frees the pages they map to — the existing
`cm._free` + page-table zeroing already works on slot granularity, and the
allocator refcounts pages underneath.

The genuinely hard parts are not the free path:

1. **Partial last page**: the spec span `q+k+1 .. q+n` (n ≤ 2 with the current
   depth gate, so at most 2 draft slots) usually lands *inside* the req's last
   live page. Freeing that page would drop the committed tokens sharing it. So
   the reject free must be **slot-scoped, page-aware**: keep the page alive
   while any committed slot in it is live; only free pages whose every token is
   beyond the rewind frontier. Concretely:
   - pages fully inside `[head, tail)` → free,
   - the boundary page → free only the slots `[head − page_base, tail)` by
     masking them in the free list (or leave the boundary page allocated to the
     req and let the next `allocate_paged` hand back the still-empty slots —
     the pool's per-slot free list already supports that if `_free` is
     slot-granular; verify in `mha_pool`/`cache.py` which of the two holds).
2. **Index-slab rows for the discarded tokens are never reclaimed** — but they
   are also never *required* to be: the slab is an append-only score cache
   ("written rows are never cleared again", `qsa_pool.py:113`), visibility is
   clamped by `kvlen // index_ratio` in the score kernel. Rolling back `kvlen`
   makes the stale tail rows unreachable. The pending ring, however, is
   position-indexed (`ring_row = slots * cap + positions % cap`) — a rejected
   draft row **did** write ring members for positions that are being rewound.
   Those stale members must not feed the next forward's straddling group.
   Mitigation: the ring is a bounded window; after a reject, the next decode
   step re-writes the same ring rows for the rewound positions (positions
   `q+k+1 ..` are re-decoded). Any row whose position was discarded will be
   overwritten before it is read again — **but only if** the read side never
   looks at positions ≥ the rewound frontier within one step. Verify with the
   QSA group-closing math (`closing = out_loc % ratio == ratio − 1`): a rewound
   decode step rewrites exactly the rows it would read. This needs a
   proof-by-test, not by argument (Stage-3 lesson: pinned-state theories die
   under probe matrices).

## Plan

### Stage A — gate lift, eager path first (no graphs)

Scope: decode-only, `--spec-draft-n 1` (depth 2 is orthogonal and stage-4
proven), bs=1..2. CUDA graphs stay off for spec batches (they already are —
the eager draft path is the FT_SPEC_DRAFT_EAGER oracle, proven bit-exact).

1. **`_spec_armed`**: drop the `page_size == 1` clause; add a pool-capability
   check instead (QSA/BSA pools: fine; anything else keeps the gate). Gate on
   `isinstance(eng.kv_cache, QSAKVCache)` OR page_size == 1, not on page_size
   alone.
2. **`_prepare_spec_batch`**: the up-front `allocate_paged` for the draft span
   already works for page-64 (it allocates fresh pages for the tail as today).
3. **Reject rollback**: replace the slot-scope free with the page-aware
   boundary logic above. The QSA pool's `_free` is page-granular; the change is
   in which slots are passed and whether the boundary page is freed or retained.
   The page-table zeroing must match (zero only freed slots).
4. **KV row writes**: none needed — `store_kv`/index writes go through
   `batch.out_loc` which the scheduler stages for the spec replay rows already;
   the rows written by rejected drafts are simply stale, unreachable rows (same
   as the slab).
5. **Verify row batches**: the 2 sequential 1-row verify batches work unchanged
   (each row is its own extend window; QSA metadata is rebuilt per row batch —
   the `_idx_slot` map covers layer 48 since the pool bound fix `7ca7273`).

Acceptance: bit-equality — plain vs spec at depth 1 on QSA, 3 prompts, ≥500
tokens each, byte-identical (the stage-3 gate harness, pointed at the qwen38
serve). Plus acceptance-rate telemetry ≥ 0.4 (QSA MTP draft quality is the
unknown; measure before optimizing).

### Stage B — graphs (only if Stage A holds)

The spec verify path is eager for the 35B because GDN snapshots + host gathers
break capture. On QSA there is no GDN in the draft layer (QSA layer), but the
host-pinned expert gather still forces eager. Leave graphs off; revisit only
after measuring. The stage-3 machinery (graph family for verify) can be reused
later if the PCIe gather moves into a persistent kernel.

### Stage C — depth 2

Stage-4 chain logic is scheduler-generic (advance span + resolve). Port cost:
mostly re-running the stage-4 gates on QSA. Only after Stage A proves
acceptance ≥ 0.55 on real prompts.

## Risks

- **Pending-ring correctness under reject** (highest): stale ring rows could
  corrupt a straddling group's index key. Kill criterion: bit-equality gate.
  Fallback if it fails: on reject, re-stage the ring rows for the rewound span
  from the slab (the slab rows for those positions were written by the
  committed prefix and are still valid — read-back is a gather over known
  rows).
- **Slab visibility**: `_select` scores *complete* blocks only; a rewound kvlen
  shrinks the visible block count. The score kernel reads unmasked tail rows
  relying on zero-fill — stale nonzero rows beyond kvlen must be excluded by the
  clamp (`kvlen // index_ratio`), which is already the invariant.
- **QSA pool free granularity**: if `MHAKVCache._free` turns out to be
  strictly page-granular (no slot masking), the boundary-page-retention variant
  is required; that leaks at most `page_size − 1` slots per reject cycle per
  req, reclaimed at finish — bounded, acceptable, and worth checking first
  before building slot masking.

## Sequencing

1. Page-aware rollback + gate lift (~1 day incl. the bit-equality gate run).
2. Acceptance measurement on real serve (telemetry already wired: the
   `report_spec_window` counters) (~half day).
3. Decide depth 2 / graphs by measurement (deferred).