# Plan: generalizing FreeToken GGUF serving beyond gemma4

Status: proposal (2026-09-10). High-level decisions here; per-arch coding is
delegated to the local coder model (`omp --model freetoken/qwen3.6-35b-a3b`).

## Goal

`ft serve --model <file>.gguf` works for the common open-weight GGUFs, not just
gemma4: dense llama/qwen2 (e.g. Qwen2.5-Coder-14B Q3_K_M) and qwen3.5-moe-style
hybrid MoE (e.g. Qwen3.6-35B-A3B NVFP4 as `qwen35moe`).

## What exists (verified in-tree)

- `models/gguf/`: reader (gguf-py), config shim, tokenizer builder, torch
  reference dequant for F32/F16/BF16/Q4_0/Q8_0/Q6_K only.
- `models/gemma4/gguf.py`: the single wired GGUF family
  (`GGUF_ARCH_TO_REGISTRY = {"gemma4": "Gemma4GGUFForCausalLM"}`).
- `layers/gguf.py`: `GGUFLinear` / `GGUFEmbedding` with MMVQ/MMQ dispatch
  limited to Q4_0, Q8_0, Q6_K; `GGUFTiedLMHead` for the tied-Q6_K head.
- `kernel/csrc/gguf/`: vendored kernels actually support dequant + MMVQ + MMQ +
  MoE for Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2_K..Q6_K and the iq* types; the Python
  layer just doesn't expose most of them.
- `moe/`: Q4_0 expert offload banks + `fused_experts_gguf_q4_0` + CPU executor
  W4A8 path (`_WFMT_IDS` includes `q4_0`).
- FTW: `source_metadata.gguf` sidecar + `freetoken.output_weight_present` KV
  (config/tokenizer travel inside the GGUF header).
- utils/hf.py already routes tokenizer / eos / sampling through GGUF metadata.

## Gap analysis (from a real qwen2 Q3_K_M and qwen35moe NVFP4 GGUF)

1. One family wired. `qwen2` and `llama` are dense; `qwen35moe` is MoE with
   separate `ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps` tensors (NOT the
   fused `ffn_gate_up_exps` gemma4 expects) plus `*_shexp` shared-expert
   tensors and per-tensor `scale` / `input_scale` sidecar tensors (ggml_type 40
   = a llama.cpp NVFP4-style layout FreeToken does not read).
2. Quant-type coverage: `BLOCK_SHAPE` / `_MMVQ` / `_MMQ` / `_DEQUANT` in
   `layers/gguf.py` cover 6 of the ~20 types the vendored kernels support.
   Q3_K_M (the common 14B-coder quant) needs Q3_K (id 11); Q4_K (12) and
   Q5_K (13) are the other common K-quants. All already exist in the CUDA
   kernels and in gguf-py's `GGML_QUANT_SIZES`; only the Python tables and a
   torch reference dequant are missing.
3. Qwen2 GGUF carries F32 `attn_q/k/v.bias` - `GGUFLinear` allocates a bias
   slot but the current kernel wrappers take no bias; bias must be added in
   Python after the quantized matmul (it already is, line 87) - the gap is the
   loader, not the kernel: `iter_gguf_weights` for qwen2 must fuse the three
   packed bias vectors alongside the packed weights.
4. Untied head: qwen2 ships `output.weight` (Q6_K here, any type in general);
   the shim already detects it via tensor presence + the FTW KV. Needs a
   `GGUFUntiedLMHead` (a `GGUFLinear` in the lm_head slot) instead of the tied
   one.
5. Vocab sizing: verified correct on both local files. `_vocab_size` reads
   `token_embd.weight`'s torch shape from `iter_gguf_tensors`, which already
   reverses ggml order, so `shape[-1]` is the vocab (152064 / 248320 both
   match `len(tokenizer.ggml.tokens)`). No change needed.
6. `qwen35moe` is out of reach for wave 1: NVFP4-style ggml_type 40 tensors,
   GDN (ssm_*) linear-attention layers, `attn_qkv` fused-in-file, 256 experts,
   per-expert scales. Wave 4.

## Phases

### Wave 1 - dense llama + qwen2 (the coder model's first assignment)

1. `GGUF_ARCH_TO_REGISTRY`: add `"llama": "LlamaGGUFForCausalLM"`,
   `"qwen2": "Qwen2GGUFForCausalLM"`.
2. Per family, in `models/<fam>/gguf.py`:
   - `parse_gguf_config(shim)` mirroring `models/<fam>/config.parse_config`
     but reading `{arch}.block_count`, `embedding_length`,
     `attention.head_count{,_kv}`, `rope.freq_base`, `context_length`,
     `attention.layer_norm_rms_epsilon`, `feed_forward_length`.
   - `iter_gguf_weights` mapping per the verified table:
     - `token_embd.weight` -> `model.embed_tokens.qweight`
     - `output.weight` (untied) -> `lm_head.qweight`
     - `output_norm.weight` -> `model.norm.weight`
     - `blk.N.attn_norm.weight` / `blk.N.ffn_norm.weight` -> layer norms
     - `blk.N.attn_q/k/v.weight` -> fused `qkv_proj.qweight` (packed-row
       concat, same in_features) + the three F32 biases fused into
       `qkv_proj.bias`
     - `blk.N.attn_output.weight` -> `o_proj.qweight`
     - `blk.N.ffn_gate/up.weight` -> fused `gate_up_proj.qweight`
     - `blk.N.ffn_down.weight` -> `down_proj.qweight`
   - `convert_<fam>_to_gguf(model, config)`: swap dense linears +
     embedding; if untied, swap lm_head to a `GGUFLinear`-based head; if
     tied, reuse `GGUFTiedLMHead` generalized to the actual embedding quant
     type (read it off `token_embd.weight`, don't hardcode Q6_K).
3. `is_gguf_model` must stop keying on `moe_weight_format == "q4_0"` (dense
   models have no experts): move the flag to an explicit
   `weight_format: "gguf"` field on ModelConfig set by every `parse_gguf_config`.
4. Engine: skip offload/expert-bank setup for dense GGUF (`is_moe` False
   already short-circuits most paths; verify `shared_offload_method` and
   `_CPU` gating tolerate a GGUF model with no MoE layers).
5. `is_gguf_model` consumers in engine/engine.py need the same
   re-keying (they check `moe_weight_format` today).
6. Tests: extend `tests/models/test_gemma4_gguf_rope.py` pattern -
   registry resolution test drives the new specs automatically; add a
   mapping test per family (real GGUF gated by env var, like
   `FREETOKEN_GEMMA4_GGUF_GLOB`).

### Wave 2 - quant coverage (mechanical, kernels already exist)

1. Extend `BLOCK_SHAPE` + `GGML_NAME` + torch reference dequant + `_MMVQ` /
   `_MMQ` / `_DEQUANT` for Q4_1(3), Q5_0(6), Q5_1(7), Q2_K(10), Q3_K(11),
   Q4_K(12), Q5_K(13), plus dequant-only iq* types (16-23, 29).
   Sizes come from gguf-py `GGML_QUANT_SIZES` (no guessing).
2. Per-tensor type dispatch: today `iter_gguf_weights` assumes one quant type
   for all projections. Q3_K_M mixes Q3_K and Q4_K/Q5_K tensors, so the swap
   helper must read `t.ggml_type` per tensor and pass it into `GGUFLinear`.
3. Test: round-trip each new type through `ggml_dequantize` vs the torch
   reference on synthetic blocks (pattern already used by the q4_0 tests).

### Wave 3 - MoE GGUFs (qwen3_moe / qwen2_moe / deepseek2 style)

1. Expert tensors arrive as `ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps`
   `[n_embd, ff, n_expert]` 3D packed tensors. Add a provider in
   `models/weight.py::load_q4_0_moe_expert_sources`-style shape for the
   three-tensor layout and a `_BANK_SCHEMAS["q4_0_3t"]` (gate_up assembled by
   concat along dim 1 at load, or store three banks).
2. `general.sampling.*` keys already flow (`utils/hf.py`).
3. Gating: `expert_used_count`, `expert_count`, `expert_feed_forward_length`
   metadata keys per arch.

### Wave 4 - qwen35moe / NVFP4-in-GGUF (explicit non-goal for now)

ggml_type 40 + `scale`/`input_scale` sidecars are a llama.cpp extension, not
plain GGUF quants; would need kernel work. Revisit after waves 1-3.

## Risks / decisions held by the human

- **Bias in GGUFLinear**: kernel wrappers are bias-free; adding bias means
  the fused-qkv path stores an F32 bias tensor next to the packed rows.
  Accepted; matches Qwen2 HF layout.
- **TP>1 stays rejected** for GGUF (packed rows don't shard) - keep the
  loud `NotImplementedError`, document in `docs/models.md`.
- **`moe_weight_format` overloading** is already leaking ("q4_0" doubles as
  "is GGUF"); wave 1 introduces `weight_format` and migrates the two checks
  (`gemma4/gguf.py:is_gguf_model`, `engine/engine.py`) in the same PR.
- **CPU executor for dense GGUF**: dense models never hit the expert path;
  no new CPU work needed for wave 1/2.

## Verification gates (per AGENTS.md)

- `uv run pytest tests/models -m "not slow"` after each wave.
- Registry test (exists) must pass for the new specs: every `parse_gguf_config`
  / `iter_gguf_weights` attr must exist on the module.
- Real-GGUF gate: `FREETOKEN_GGUF_GLOB` env var per family, mirroring
  `FREETOKEN_GEMMA4_GGUF_GLOB`; smoke test is
  `ft serve --model <gguf>` + one chat completion.
- `ft checkpoint` on a GGUF must keep producing a loadable FTW dir
  (`source_metadata.gguf` path already covers config/tokenizer).