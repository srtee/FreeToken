# qwen35moe Debug Plan — Layered Attack on the Incoherent-Output Bug

## Problem statement

The qwen35moe (Qwen3.5/3.6 hybrid GDN+MoE) GGUF adapter loads and serves,
but generation diverges: the first decode token is often plausible, then
output degenerates within a few tokens. All static weight mappings have
been verified line-by-line against llama.cpp's
`src/models/qwen35moe.cpp` (which serves the same GGUF coherently on this
machine, GPU + expert-offload). Ground-truth greedy continuations are
captured and serve as regression targets:

| Prompt | llama.cpp ground truth | ft (current) |
|---|---|---|
| `The capital of France is` | ` Paris, a city renowned for` | `PStournjepCoiel` |
| `1, 2, 3,` | ` 4, 5,` | ` 0, 8` |
| `def fibonacci(n):` | `\n    if n <= ` | ` affiteqezensika` |

Established facts (do not re-derive):
- Dense NVFP4 dequant is element-exact vs gguf-py; per-tensor `.scale`
  global semantics match llama.cpp's `build_lora_mm` output-multiplier.
- Norms: GGUF stores pre-baked `(1+w)`; adapter passes through (fixed).
- `ssm_a` stores `-exp(A_log)`; adapter converts `A_log = log(-rate)` (fixed).
- Partial rope: `rotary_dim = 64` from `rope.dimension_count` (fixed).
- `FREETOKEN_ZERO_EXPERTS=1` (zeroed expert globals) still produces
  garbage → **the bug is in the dense path** (attention / GDN / shared
  expert / router), NOT the routed-expert banks.
- The fla GDN decode kernel matches `gdn_reference.recurrent_gated_delta_rule`
  in isolation (verified; earlier 11.3x "diff" was my test's scale error).
- First decode token plausible + later divergence ⇒ prefill attention over
  the prompt KV is *approximately* right; corruption enters through
  decode-step state updates or the KV of generated tokens.

## Attack layers (in order — stop at the first that localizes the bug)

### Layer 0 — Logit fingerprint harness (build once, use everywhere)
A tiny script `scripts/logit_probe.py` that, given a model path and a
token-id prefix, runs ONE forward and prints the top-8 logits/ids for the
next token. Two backends:
- `--engine llama`: llama.cpp `/completion` with `n_probs=8` on the same
  token ids (bypass chat template; feed raw ids via `--tokenize` off or
  use the `/tokenize` endpoint then `/completion` with `prompt` as
  control chars? — llama.cpp accepts raw byte prompts; ensure identical
  ids via the `/tokenize` endpoint round-trip).
- `--engine ft`: in-process — build the engine via the normal serve path
  in a subprocess, hit `/v1/completions` with `prompt` = detokenized ids
  AND `echo=True`-style prefix; or better, add a debug flag to the ft
  server that accepts explicit prompt token ids (small patch to
  `GenerateRequest`, guarded by an env var).
Deliverable: `logit-diff` tool printing ft vs llama.cpp top-k for any
prefix. This converts the vague "incoherent" into a per-token, per-layer
measurement.

### Layer 1 — Prefix-length sweep (prefill vs decode discrimination)
With the harness: run prefixes of length 1..N on the same text. If ft's
FIRST predicted token is wrong already at short prefix (pure prefill),
the bug is in prefill. If the first token matches llama.cpp up to prefix
length L then diverges after decode steps begin, the bug is decode-state.
Previous evidence says the latter, but a sweep makes it conclusive and
gives the exact divergence step.

### Layer 2 — Layer-0 GDN-only probe (bypass the model, test the state machine)
Drive `Qwen3_5DecoderLayer`/`Qwen3_5GatedDeltaNet` directly on GPU with a
minimal `Context` + a stub batch (the earlier attempt failed only on
missing `ctx.batch`/FLA metadata — build the FLAMetadata by hand: it is
just cu_seqlens/cache_indices/has_initial_state tensors). Compare against
`gdn_reference.recurrent_gated_delta_rule` STEP BY STEP:
1. prefill of T tokens via `_conv_prefill` + `gdn_prefill_chunk_fla` —
   check the written `pool.recurrent_states[li]` and `conv_states[li]`
   against the reference's final state (this is the handoff).
2. T decode steps via `_conv_decode` + `gdn_decode_fla`, comparing the
   output AND the state after every step against a step-by-step reference
   loop.
If step-by-step matches in isolation but not in the engine, the bug is in
the metadata plumbing (cache_indices, slot selection, pool indexing) —
compare `fla.cache_indices` values against the actual request slots.

### Layer 3 — Attention KV probe
Same direct-layer approach for the full-attention layers: run layer 3
with a fixed 8-token input, then 7 decode steps, dumping the KV cache
rows written per step (k_cache/v_cache rows at the out_loc positions) and
the attention output. A wrong `out_loc` during decode (e.g. one-off, or
the same slot reused) would corrupt step 2+ exactly as observed.
Cross-check the written K/V vectors against llama.cpp's computed K/V for
the same token (extract via its `/embedding`-style debug or by computing
qkv_proj.forward(token_embedding) directly — that part already works).

### Layer 4 — Suspect-ranked fixes
Apply fixes in this order of likelihood (from the evidence so far):
1. **GDN conv-state layout in decode** — `causal_conv1d_decode` expects
   `[B, conv_dim]` input and updates `conv_states[li]` in place; the
   prefill path writes the trailing window via `conv_win` gather in
   `_write_track_snapshot` (hybrid-radix only) and the varlen prefill
   writes states internally. Compare the prefill-written conv state
   against a manual roll of the conv input window (Layer 2 gives this).
2. **`softplus` threshold mismatch** — kernel uses
   `softplus_beta=1.0, softplus_threshold=20.0` with a hand-rolled
   `log(1+exp(x))` (no linear-region shortcut — line 174 computes
   `log(1+exp(beta_x))` even for large x; harmless numerically but check
   the a+dt_bias magnitudes: if `a+dt_bias` is large, `exp` overflows in
   fp32 → inf/nan gating. torch F.softplus switches to linear above 20.
   The kernel's `tl.log(1 + tl.exp(x))` OVERFLOWS for x>88 — print the
   actual a+dt_bias stats for this checkpoint).
3. **FLA decode kernel's `o[0]` squeezing** — `o[0]` returns all B tokens
   of the FIRST head-chunk (NK dim); if NK≠1 under some batching, the
   output selection could be wrong for B>1. Test B=1 vs B=2 decode.
4. **Router/shared-expert interaction** — only after 1-3 are cleared;
   the zero-expert test says the dense path is broken, but the router
   bias/top-k renormalization config (`norm_topk_prob=True`) could still
   amplify.

### Layer 5 — Regression armor (do regardless of outcome)
- Add the three captured ground-truth continuations to
  `tests/models/test_qwen35moe_gguf.py` as an E2E coherence test marked
  `@pytest.mark.gpu` (skipped when no GPU/model), asserting ≥5-token
  greedy match against the llama.cpp targets.
- Keep the synthetic structural tests as-is (they caught 4 real bugs).

## Operational guardrails (learned this session, non-negotiable)
- Serve with `sudo -u sherntee bash -c "ulimit -l unlimited && exec
  nohup .venv/bin/ft serve ..."` — banks are mlocked; MEMLOCK must be
  unlimited (systemd override installed).
- `--cuda-graph-max-bs 0` (graph capture illegal-accesses on this model).
- llama.cpp ground truth: `llama.cpp/build/bin/llama-server -ngl 999 -ot
  "exps=CPU"` — but NOT simultaneously with ft (30G RAM; the ft banks +
  llama.cpp model exceed it → systemd-oomd kills things silently).
- Every ft worker crash can leave a VRAM-holding zombie: sweep with
  `python3 -c "os.kill(pid,9)"` over `ss -tlnp` port-1920 holders; two
  crashes = reboot (driver .58 still wedges on kill-during-DMA).
- Never `pkill -9` mid-load; kill via python os.kill only (bash tool
  timeouts interrupt before the signal lands otherwise).