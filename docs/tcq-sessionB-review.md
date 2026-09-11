# gguf-wave3 (Session B) review of Session A's turbo KV work — 2026-09-11

## Verified correct
- All 5 python files + .cu parse clean (ast-checked).
- launchers: TurboQuantLaunch/TurboDequantLaunch/TurboCodebookUpload match the
  python-side signatures in kernel/turbo_kv.py (launch(src,dst,locs,is_v);
  dequant(src,dst,locs,codec,is_v); upload(cb3k,cb3v,cb2k,cb2v)).
- locs dtype: scheduler builds batch.out_loc from page_table rows -> int64
  (page_table itself is int32 but _make_input_tuple builds int64). The quant
  kernel reads ((const int64_t*)p.locs)[row] and the TVM matcher accepts
  int32/int64 — BUT the kernel unconditionally reads int64 while the matcher
  also admits int32. If locs is ever int32 the kernel reads garbage. Two
  options: narrow matcher to int64 only, or cast in TurboKVCache.store_kv.

## Bugs found
1. **locs dtype mismatch risk (medium)**: matcher allows int32; kernel reads
   int64. TurboKVCache.store_kv should do `out_loc.to(torch.int64)` if
   out_loc.dtype != int64 (currently passes through).
2. **store_kv dst shape contract (high)**: turbo_quantize's dst matcher wants
   (Rows, H, kBlockBytes) but TurboKVCache.store_kv passes
   self._k_packed[dense] which IS (tokens, heads, bb) — OK — but the kernel
   writes dst_base = dst + dst_row*H*bb + head*bb. Consistent. However
   turbo_quantize in turbo_kv.py docstring says locs selects rows; the pool
   passes out_loc (length L). Matches. OK — no bug, confirmed.
3. **materialize scratch aliasing (high)**: k_scratch = self._k_scratch[:n]
   is sliced from a persistent buffer; fine — but materialize returns views
   consumed asynchronously by flashinfer; a subsequent batch's store_kv may
   overwrite _k_scratch while the previous attention kernel still reads it.
   Needs either per-layer scratch or a sync/event. Recommend: allocate scratch
   per (layer, n) or record a CUDA event on the compute stream before reuse.
4. **fi.py seq_lens_cpu transfer**: metadata.seq_lens_cpu.to(indices.device)
   is a blocking H2D each step (small, acceptable) — OK but note it.
5. **rebuild() allocates dummy _kv_buffer fp16 slab (2*layers*pages*heads*128*2B)**
   just to keep bookkeeping — doubles apparent memory and defeats the VRAM
   saving. kv_cost returns the packed cost, but the engine's pool accounting
   may use _kv_buffer. Recommend computing bookkeeping fields without
   allocating the fp16 slab.

## Test status
- No unit tests exist for turbo_pool/turbo_kv (tests/kvcache has none).
- Suggested minimal test: tests/kvcache/test_turbo_pool.py — quantize random
  (8,4,128) fp16 -> dequant, assert max rel err < codec gate from
  docs/tcq-baseline-numbers.md (turbo4 <1e-3, turbo3_tcq <2e-2), and
  store_kv/materialize roundtrip through a fake page_table.

## Addendum after deeper trace (graph path)

6. **CUDA-graph capture is the load-bearing hazard (critical)**: decode runs
   `graph_runner.replay(batch)` → captured `model.forward()` → FA forward →
   `pool.materialize()`. Captured inside the graph:
   - `int(cache_seqlens.max().item())` — device→host sync; at capture it reads
     the dummy batch's seqlen; at replay the captured scratch shape/CPU value
     is stale (scratch slice shape frozen at capture size).
   - `metadata.seq_lens_cpu.to(device)` in fi.py — an H2D copy captured at
     capture time with capture-time data; replay ignores new seq lens.
   Both mean decode-with-graphs + turbo codec = silently wrong attention.
   Fix options (A's call): (i) force `use_graph=False` when kv_codec != f16
   (engine.py:914 `use_graph = ... and codec == "f16"`), or (ii) make
   materialize graph-safe: capture a fixed (max_bs, max_pages) scratch once,
   gather with the captured page_table buffer, dequant with grid sized to the
   frozen shape (padded rows read a dummy page). (i) is one line; (ii) needed
   for perf parity later.

7. **locs matcher vs kernel mismatch (from main review, confirmed real)**:
   `.with_dtype<int32_t, int64_t>` admits int32 but kernel reads int64.
   page_table is int32, so if anyone ever passes a page_table-derived locs
   directly it corrupts. TurboKVCache.store_kv should assert/cast int64.

## Exact gate suggestion (verified against source)

GraphRunner has no engine config, so the cleanest disable point is in
`GraphRunner.__init__` (python/freetoken/engine/graph.py:94): accept
`kv_codec: str = "f16"` and after `self.graph_bs_list = sorted(cuda_graph_bs)`
add:

    if kv_codec != "f16":
        self.max_graph_bs = 0
        self.graph_bs_list = []
        # (log: cuda graphs disabled for turbo KV codec — materializer is not
        #  graph-safe yet)

and pass `kv_codec=config.kv_codec` at both GraphRunner construction sites
(engine.py:420 and engine.py:897 rebuild path). `can_use_cuda_graph` then
returns False and every decode runs eager through materialize().

## CORRECTION — locs bug upgraded to CRITICAL (verified end-to-end)

Traced the dtype: `batch.out_loc = page_table[input_mapping]` (scheduler.py:786)
with `input_mapping` int64 tuples → advanced indexing returns **int32** (the
page_table's dtype). The quant kernel reads `((const int64_t*)p.locs)[row]`
(turbo_kv.cu:148, 268) → pairs consecutive int32s → garbage row indices on the
VERY FIRST store_kv. Every turbo write corrupts.

Fix (pick one):
- Python (1 line, tiny cost): in `TurboKVCache.store_kv`, first statement
  `out_loc = out_loc.to(torch.int64)` (L is small; capture-safe).
- CUDA (cleaner): make the kernel read via the element size — replace both
  `((const int64_t*)p.locs)[row]` with a helper that checks
  `p.locs_dtype` (add field) or just read int32 and widen.
Python-side cast is the safe immediate fix; matcher can stay as-is.
