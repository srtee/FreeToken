# TCQ wave 3b — delegation notes (3.2 TP, 3.3 ctl cache, 3.4 FTW)

Grounded facts for the remaining plan items (all verified in-repo this session).
Plan: `docs/tcq-kv-plan.md` items 3.2–3.4. InnerQ (3.1) and docs (3.5) shipped
in `dec615b`.

## 3.2 Multi-GPU (TP>1) — document + unit test; no TP=2 box

- KV slabs shrink per-rank inside `MHAKVCache.__init__`
  (`kvcache/mha_pool.py:36`): `local_kv_heads = div_even(num_kv_heads,
  tp_info.size, allow_replicate=True)`. `TurboKVCache` inherits this and packs
  per `(token, local_head)` row.
- Quantization is per-128-group with no cross-rank reduction: the K/V a backend
  hands `store_kv` are already the rank-local head slices (same contract the
  f16 pool serves). The InnerQ accumulators are per-rank `__device__` globals —
  each rank calibrates from its own heads; scales stay decode-local, so
  per-rank divergence is correct behavior, not a bug.
- No TP guards exist anywhere in `turbo_pool.py` / `kernel/turbo_kv.py`.
- Single GPU on this box → per plan: "assert single-rank correctness and
  document". The honest equivalent test: two pools each holding half the heads
  (TP=2 simulation) must pack the same rows a full pool packs for those heads —
  per-head quantization means head-splitting is exact. Pin that in
  `tests/kernels/test_turbo_innerq.py` (or a sibling turbo test file), plus a
  doc note in `docs/models.md` (KV codecs section): TP supported, scales are
  per-rank, replication (`allow_replicate=True` for odd head counts) works
  unchanged since quantization never crosses heads.

## 3.3 ft ctl cache — live resize + codec fields

- Live resize ALREADY WORKS structurally: engine `_resize_kv_pool` →
  `rebuild_from_config` (inherited, `num_pages + 1` dummy-page convention,
  matches `create_kv_pool` line 121) → `TurboKVCache.rebuild` reallocs the
  packed slabs (codec-uniform → straight byte gather; comment in the method).
  The budget fit-check (`BaseKVCachePool.validate_rebuild`) dispatches
  `type(self).kv_cost` → codec-aware packed pricing (now guarded: non-128
  head_dim falls back to the f16 rate so it can never under-budget).
- Calibration state survives rebuild: scales live in JIT device symbols, not
  in the packed slabs; `_num_pages`-only realloc doesn't touch them.
- MISSING piece 1 — geometry codec fields: `/v1/cache/status` geometry
  (`api_server.py:710 cache_geometry`) exposes `kv_codec` only inside
  `unit_bytes`. Add top-level `kv_codec` + `kv_codec_tune` +
  `innerq_calibrated` (bool: read off the pool — e.g. a pool property like
  `is_turbo`/`codec`; False for f16 pools). Defensive `getattr` guards like
  the existing `num_experts` block (dummy configs must not 500 the poll).
- MISSING piece 2 — display: `cache_report.format_cache_status` should surface
  the codec (e.g. a dedicated line or row suffix) so `ft ctl cache status` and
  the shell `/cache` show it. `control_cli.py` needs no change (it renders the
  report).
- MISSING piece 3 — proof: start the 14B GGUF with `--kv-codec turbo4
  --kv-codec-tune innerq`, `ft ctl cache status` shows the codec fields, then
  `ft ctl cache rebuild --kv <smaller>` succeeds and generation still works.
  Orchestrator runs the serve smoke (GPU); the agent wires fields + display +
  a unit test for `cache_geometry`/`CachePools` with a fake state.

## 3.4 FTW runtime-only note

- Verified: `checkpoint/ftw.py` stores `kind="weight"` tensors only;
  `convert.py` never touches KV. The KV codec is a runtime pool decision
  (`--kv-codec`), applicable equally to HF- and FTW-served models.
- Add one sentence to `docs/cli.md` (the `ft checkpoint` section, after the
  FTW caveats reference): FTW carries weights only; KV storage codec is
  runtime-only (`--kv-codec`), so converted checkpoints serve quantized KV
  without reconversion.

## Gotchas for the agent

- `TurboKVCache.kv_cost` reads `config.kv_codec` via `getattr` — a config
  without the field silently prices f16 (bit us twice).
- head_dim is hard-gated to 128 at pool construction (`turbo_pool.py:56`); the
  qwen35moe spec (head_dim 256, 2 kv heads, 10 full layers) resolves to
  TurboKVCache but then REJECTS at construction — `resolve_pool_class` alone
  is not a construction proof. The kv_cost non-128 fallback (added today) is
  the pricing guard.
- Compression math (128-dim heads, K+V): f16 = 512 B/head-token; turbo8
  130×2 → 2.0x, turbo4 66×2 → 3.9x, turbo3_tcq 52×2 → 4.9x, turbo2_tcq 36×2
  → 7.1x. The models.md table now states exactly these.
- Tests: `tests/kernels/test_turbo_innerq.py` fixture uploads identity scales
  in teardown — keep that pattern; suites must stay full-suite-safe.