# MTP speculative-decode semantics — frozen against buun (Stage 0.1)

Reference: `/home/sherntee/20llms/buun-llama-cpp` (read-only), file:line
references as of this stage. Companion pure-logic tests:
`tests/engine/test_mtp_loop.py`.

Scope: Qwen3.5/3.6-MoE style single-block MTP (buun
`common_speculative_impl_draft_mtp`, `n_mtp_layers == 1`,
`is_mem_shared == false`, `chain_heads == false` — the same shape as
FreeToken's `MTPHead`).

Source map (buun):

- `common/speculative.cpp:60-99` — `common_speculative_mtp_carry_lifecycle`
  (the ready/cold state machine).
- `common/speculative.cpp:2591-3142` — `common_speculative_impl_draft_mtp`
  (`process` 2773-2940, `draft` 2942-3109, `accept` 3111-3142).
- `common/speculative.cpp:6701-6713` — `common_speculative_rollback_dft`.
- `tools/server/server-context.cpp:20058-20173` — the caller: accept bookkeeping,
  target rollback, then `common_speculative_rollback_dft`.
- `src/models/qwen35moe.cpp:554-742` — `graph_mtp` (the MTP draft graph).

---

## Q1 — Layer-40 (MTP layer) KV row ownership per position

**Answer: the MTP layer needs a KV row at every position its attention
attends over, and buun writes it by REPLAYING the whole target batch
through the MTP graph in `process()`, not by per-position ownership.**

Positions, precisely: with verify rows `[c@q, d@q+1]`, row A's hidden is
at position q, row B's at position q+1. Under accept the next draft
consumes the carry from row B (the newest verified position, q+1); under
reject row B is rolled back, so the newest verified position is q and the
next draft consumes row A's hidden.

Evidence:

- The MTP graph attends against the draft context's paged KV
  (`build_attn_inp_kv` + `build_attn(..., il)` with
  `il = hparams.n_layer()`, qwen35moe.cpp:606, 563, 663-665). So any
  position in the KV is visible to any future draft step at a greater
  position: rows must exist densely, gaps are not skip-able in the graph
  itself.
- buun's `process(batch)` (speculative.cpp:2841-2918) replays **the
  entire target batch** (prefill rows or the verify rows `[c@q, d@q+1]`)
  through the draft context, one MTP graph pass per `n_mtp_layers`:
  - it first rewinds the draft KV to the target batch's first position
    (`llama_memory_seq_rm(ctx_dft, seq, pos_first, -1)`,
    speculative.cpp:2844-2849) — M-RoPE requires stored positions <=
    incoming positions;
  - rows `[1..n-1]` get the **shifted target hidden** `h_tgt`
    (speculative.cpp:2865-2867: `embd[1..n-1] = h_tgt[0..n-2]`, i.e. row
    i processes token i with the predecessor's target hidden);
  - row 0 gets the pending carry `pending_h` (speculative.cpp:2885);
  - then `llama_decode(ctx_dft, batch)` (speculative.cpp:2903) — this
    writes layer-40 KV rows for **all batch positions** at once.
- The draft loop (`draft`, speculative.cpp:2942-3109) writes one more row
  per draft step: it decodes `dp.id_last` at `dp.n_past` plus one row per
  drafted token (speculative.cpp:2975, 3078), each with the preceding
  row's `h_nextn` as `embd`.
- On accept, `accept(seq, n_accepted)` (speculative.cpp:3111-3142) only
  re-seats the carry (`pending_h = verify_h[min(n_accepted, n_rows-1)]`);
  KV rows the target already verified are simply left in place — no
  per-position re-write is needed because `process()` had already written
  rows for the verified positions.
- On reject, the target rolls back the rejected rows in the trunk pool
  (server-context.cpp:20158-20171) and
  `common_speculative_rollback_dft(...)` (speculative.cpp:6701-6713)
  trims the DRAFT KV to `slot.prompt.n_tokens()`:
  `llama_memory_seq_rm(ctx_dft, seq, n_past, -1)`, then re-seats the
  carry via `accept()`. The next `process()` then re-writes the replayed
  region (the seq_rm at 2844-2849 does the same job each iteration).

**Resolved semantics (the model the tests encode):** layer-40 rows exist
for every position `< frontier`. The draft writes position q's row (the
input token of the iteration); the verify's row A re-writes it (same
position, deterministic content) and row B writes position q+1. Under
accept, positions q and q+1 both stand verified and stay. Under reject,
row B's position q+1 row is freed and `device_len` rewinds by 1; the next
iteration's verify row A re-writes position q+1 with the corrected token.

**The gap question — settled: there is NO gap under this loop.** The
plan's worry ("the accepted-draft position, whose row nobody writes")
dissolves in buun's formulation because the verify forward is a single
2-row batch that writes layer-40 rows for BOTH positions (row A re-writes
q; row B writes q+1) in the same decode. The accepted-draft position q+1
has its row written by verify row B itself — nobody has to catch up.
buun's equivalent of a gap only appears across iterations: `process()`
rewinds to the target frontier and re-replays (2844-2849) precisely
because the draft may have advanced the KV past it (2966-2969); that
rewind is the mechanism that keeps the layer-40 rows dense without any
extra catch-up decode step.

## Q2 — Carry source per resolve case

**Answer: accept → row-B hidden (the hidden at the newest accepted
position); reject → row-A hidden (the hidden at the last retained
position). Both come from the TARGET's verify forward, not from the
draft.**

- The target verify forward fills `verify_h[seq]` with one `h_nextn` row
  per verify row (speculative.cpp:2920-2937), and `pending_h` is set to
  the LAST row (2934-2936) — the pre-acceptance default.
- `accept(seq_id, n_accepted)` (speculative.cpp:3134-3141):
  `i_h = min(n_accepted, n_rows - 1)`,
  `pending_h = verify_h[i_h]`.
  - accept (n_accepted = 1, n_rows = 2): `i_h = 1` → **row B's hidden**
    (`verify_h[1]`).
  - reject (n_accepted = 0): `i_h = 0` → **row A's hidden**
    (`verify_h[0]`).
- The lifecycle flips to ready at every target process
  (`target_process_refreshed`, 2936) and the carry is nulled on any
  rewind/restore event (`sequence_transition`, 86-99: prompt_rewind,
  restore, live_range_shift, ... → `ready = false`), in which case the
  next target-only batch re-seats it (2815-2838) or zero-cold-starts at
  position 0 (2879-2883).

So the carry that the NEXT draft consumes is always the target hidden at
the newest position whose token the trunk itself committed — row B under
accept (position q+1, since the bonus `b` is committed at position q+2),
row A under reject (position q, after row B's rollback; the next input
`a` re-enters at position q+1).

## Q3 — Bonus-token emission

**Answer: yes — under accept, the draft token AND the trunk's row-B
argmax are both emitted.**

- The trunk verify's row-B argmax is the trunk's own prediction for
  position q+2 given the (now accepted) draft — in buun the verification
  pipeline is `common_sampler_sample_and_accept_n(ctx_tgt, ..., slot.spec_i_batch, slot.spec_draft)`
  (server-context.cpp:19992-19994), which returns `ids` = the accepted
  chain PLUS one bonus token sampled from the verify row that follows the
  last accepted draft (the standard target-verifier contract; depth-1 MTP
  has verify rows `[c, d]` → `ids` = `[d, bonus]` on accept).
- The server then appends the whole `ids` chain to the prompt
  (server-context.cpp:20064-20067) and emits it
  (server-context.cpp:20175-20199).
- Under reject, only the trunk's own argmax `a` is emitted
  (`ids = [a]`); nothing extra.

So per iteration: accept → emit `[d, b]`; reject → emit `[a]`.

## Q4 — First decode step after prefill

**Answer: after prefill, `common_speculative_process` replays the
prefill batch through the MTP graph once (row 0 gets a zero carry via
`cold_zero`), and the first draft consumes the carry re-seated from the
prefill's LAST target hidden row.**

- Prefill's target decode is followed by `common_speculative_process`
  on the target batch (speculative.h:133-140 "after EVERY llama_decode";
  wired at server-context.cpp:3210-3214 and 19405-19407).
- In `process()`, the lifecycle's `target_process_mode(pos_first)`
  (speculative.cpp:69-76) returns `cold_zero` iff `!ready && pos == 0`.
  That zeroes `pending_h` (speculative.cpp:2881-2882) and feeds it as
  row 0's embedding.
- After the prefill replay, `pending_h` = the LAST target hidden row and
  `target_process_refreshed()` re-arms drafting
  (speculative.cpp:2934-2936).
- The first `draft()` consumes exactly that: carry = prefill's final
  `h_nextn` row, token = the first sampled token
  (`dp.id_last`, 2942-2983). If the carry is not ready (e.g. a
  rewind/restore in between), `draft_carry` returns nullptr and the
  draft is skipped for that iteration (2960-2964) — the loop degrades to
  plain decode for one step, never drafts from a stale carry.
- Note: buun does NOT zero the carry "at prefill" specially; the zero
  row only exists for a cold start at position 0. For FreeToken (always
  starting from a prefill at pos 0) the effective first-iteration
  sequence is: prefill → process() (cold_zero row-0 replay) → carry =
  prefill's last hidden → draft.

---

## Resolved loop pseudocode (per iteration, mirroring the plan's corrected loop)

State per sequence: `q` (next position to write), `c` (certain input
token at position q), `H` (carry: target hidden at the newest verified
position), layer-40 KV `rows: pos -> slot`, `device_len`.

```
iteration(H@q-1, c@q):                     # H = carry, c = certain input
  d = draft(H, c)                          # 1-row MTP forward; writes KV[q] (row for c)
  [rowA, rowB] = trunk_forward([c@q, d@(q+1)])   # 2-row verify; writes KV[q] (rowA, same pos) and KV[q+1] (rowB)
  a = argmax(rowA)   # trunk's true token for q+1  — verification of d
  b = argmax(rowB)   # trunk's prediction for q+2  — the bonus
  if a == d:                               # ACCEPT
      emit [d, b]
      next_input = b     # position q+2
      carry = rowB.hidden  # row-B hidden (position q+1)
      device_len += 2     # rows q, q+1 committed
      q += 2
  else:                                    # REJECT
      emit [a]
      next_input = a     # position q+1
      carry = rowA.hidden  # row-A hidden (position q)
      free KV[q+1]'s slot; device_len -= 1   # roll back row B
      q += 1
  # layer-40 invariant: rows exist (densely) for every position < q
  # (the accept case's rowB at q+1 satisfies this directly; the reject
  # case's rollback + next iteration's rowA at q+1 restores it)
```

Equivalences checked against buun:

- "carry = rowB/rowA hidden" ⇔ `accept()`'s `i_h = min(n_accepted,
  n_rows-1)` into `verify_h` (speculative.cpp:3139-3141).
- "free KV[q+1] on reject" ⇔ target seq_rm on the rejected suffix
  (server-context.cpp:20158-20160) + `common_speculative_rollback_dft`
  trimming the draft KV (speculative.cpp:6709) — in FreeToken this is
  the plan's "free page slot + rewind device_len" for the layer-40 pool.
- "rowA re-writes position q" ⇔ buun's `process()` rewind + replay of
  the verify batch (2844-2918); the FreeToken loop reaches the same
  state because row A IS the input row of the verify forward.
- "first draft consumes prefill's last hidden" ⇔ `pending_h` refresh in
  `process()` (2934-2936) and `draft_carry` (64-67).

## Open items for Stage 1

1. **GDN reject-path mechanism** (plan already flags it): the 2-row
   verify pollutes the trunk's GDN recurrent state at row B; buun solves
   it with a full sequence-image backup + restore + re-decode of accepted
   tokens (server-context.cpp:20069-20157) or a GPU tape replay (DFlash
   only). The plan's candidates (snapshot/restore + 1-row replay vs two
   sequential 1-row GDN steps vs accept-path-only refresh) remain a
   Stage-1 decision by measurement — the semantics doc only pins the
   layer-40 (attention) side, which is trimmable in place.
2. **Multi-request interleaving of process()**: buun's `process()` is
   called on the whole target batch (server-context.cpp:19405-19407) and
   asserts contiguous per-seq row groups (speculative.cpp:2860-2863).
   FreeToken's equivalent must decide where the replay lives
   (`engine.forward_batch` vs scheduler hook) — semantics identical, the
   ownership is a code-layout question.
3. **`p_min`-style confidence gating**: buun may stop drafting early on
   low draft confidence (speculative.cpp:3038-3043) and has adaptive
   depth capping (3116-3131). Depth-1 FreeToken starts without these;
   if added later they change only the draft side, not the semantics
   above.
4. **Verify-batch abort**: what happens on abort between draft and
   verify (plan's Stage-1 gate) has no buun analogue to copy directly —
   the invariant to maintain is "after any abort, layer-40 rows are
   dense up to the committed frontier and the carry matches the newest
   verified position" (rollback + next-iteration rowA rewrite restores
   it, per the loop above).
5. **Sampler semantics**: buun verifies with `common_sampler_sample_and_accept_n`
   (joint distribution over the chain), which for greedy reduces to
   argmax-equality — the lossless formulation FreeToken uses
   (`verify_chain`). If non-greedy sampling is ever added, the acceptance
   rule changes; out of scope for wave 2 (greedy).
---

## Stage 0.3 — row-cost bench (2026-09-13, measured on the 35B)

Setup: Qwen3.6-35B-A3B-NVFP4, `--moe-strategy offload`, triton backend,
graphs off, 8192 pages, greedy, ~150-token prompts, 256-token generations,
3-run means, expert cache warmed.

- **c1 (1-row decode unit)**: 78.2 tok/s at bs=1 → 12.8 ms/step.
- **c2 (2-row forward / verify shape)**: bs=2 concurrent decode sustained
  156.4 tok/s aggregate — **2 rows cost the same wall time as 1 row**
  (c2 ≈ 1.00). The 35B decode is expert-fetch bound (MoE offload), so
  the second row rides free PCIe/CPU cycles already paid by the first.
- **cd (draft step)**: not directly measurable outside the engine
  (wave-1 MTPHead needs engine context); analytic bound: lm_head GEMV
  [152k × 2048] bf16 ≈ 0.6 GB → ~1.2 ms + 1-layer attn/MoE/fc ≈ 2–4 ms
  total ≈ **0.15–0.3 c1 units**.

**Ceiling math (depth 1, eager)**: speedup = (1 + r) / (c2 + cd) with
c2 = 1.0: at r = 0.7 → **1.52×**; at r = 0.55 → **1.39×**; even
r = 0.3 → 1.15×. **GO**: on this hardware the depth-1 eager loop is
already clearly profitable; graphs (Stages 2–3) mostly shave launch
overhead and make the draft step cheaper.

Bench caveats: bs=2 concurrency approximates a 2-row single-req verify
(same row count per forward, same expert-fetch pattern; per-req
attention shapes differ slightly — the verify attends over the req's
own KV twice rather than two reqs' KV once each). Stage 1's E2E
acceptance telemetry gives the exact realized numbers.
