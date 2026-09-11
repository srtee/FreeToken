# Wave 3 part A: qwen3moe GGUF adapter (single file task)

Work in /home/sherntee/20llms/FreeToken on branch gguf-serve. Do NOT commit.
Create ONE file: `python/freetoken/models/qwen3_moe/gguf.py`, then edit
`python/freetoken/models/gguf/config.py` (one line) and
`python/freetoken/models/qwen3_moe/__init__.py` (add exports).

## Context (no need to read other files)

GGUF flow for arch strings: `python/freetoken/models/gguf/config.py` maps
`general.architecture` -> registry key via `GGUF_ARCH_TO_REGISTRY`, e.g.:

```python
GGUF_ARCH_TO_REGISTRY: dict[str, str] = {
    "gemma4": "Gemma4GGUFForCausalLM",
    "llama": "LlamaGGUFForCausalLM",
    "qwen2": "Qwen2GGUFForCausalLM",
}
```

`register.py` already has (do NOT touch register.py):

```python
"Qwen3MoeForCausalLM": ModelSpec(
    "freetoken.models.qwen3_moe",
    "Qwen3MoeForCausalLM",
    packed_modules_mapping=_DENSE_PACKED + _EXPERTS_PACKED,
),
```

Your task adds `"qwen3moe": "Qwen3MoeGGUFForCausalLM"` to the registry dict plus
a ModelSpec in register.py pointing at module `freetoken.models.qwen3_moe`,
class `Qwen3MoeForCausalLM`, `parse_config="parse_gguf_config"`,
`iter_weights="iter_gguf_weights"` (copy the shape of the Qwen2GGUFForCausalLM
entry; read register.py lines 75-85 for the exact pattern).

Qwen3-MoE module layout (the weights you must yield; from models/qwen3_moe/):
- `model.embed_tokens` (VocabParallelEmbedding), `model.norm` (RMSNormFused),
  per layer: `input_layernorm`, `self_attn.{qkv_proj,q_norm,k_norm,rotary}`,
  `post_attention_layernorm`, `mlp.gate` (LinearReplicated), `mlp.experts`.
- Attention has QK-norm (has_qk_norm=True) and q/k BIAS (attn_q.bias/attn_k.bias
  exist in GGUF for qwen3 — map them; the qkv_proj takes has_attn_bias).
- lm_head: `Qwen3MoeForCausalLM` has `lm_head` (ParallelLMHead); untied qwen3
  checkpoints have `output.weight` in GGUF -> yield `lm_head.qweight`.

GGUF tensor names (llama.cpp qwen3moe conversion): `token_embd.weight`,
`output_norm.weight`, `output.weight`, per layer `blk.N.{attn_norm.weight,
attn_q.weight, attn_k.weight, attn_v.weight, attn_q.bias, attn_k.bias,
attn_output.weight, post_attention_norm.weight, ffn_norm.weight,
ffn_gate_inp.weight, ffn_gate_exps.weight, ffn_up_exps.weight,
ffn_down_exps.weight}`.

Expert tensors (`ffn_*_exps.weight`) and `ffn_gate_inp.weight` are OUT OF
SCOPE: skip them silently in iter_gguf_weights (they are handled by a later
converter task). Note the skip in the docstring.

## Reference implementation to mirror (llama dense adapter, models/llama/gguf.py)

Write your file with the same structure as this (paraphrased reference):

```python
def _tensor_types(model_path: str) -> dict:
    # single pass over iter_gguf_tensors(model_path): record ggml_type of
    # token_embd/output_norm/output, per-layer attn_q/k/v types, bias presence
    # (name.endswith(".bias")), expert types for ffn_gate_inp if you want.

def parse_gguf_config(shim) -> ModelConfig:
    m = shim.metadata; arch = shim.model_type  # "qwen3moe"
    def g(key): val = m.get(f"{arch}.{key}"); assert val is not None; return val
    hidden = int(g("embedding_length")); n_q = int(g("attention.head_count"))
    n_kv = int(g("attention.head_count_kv")); head_dim = hidden // n_q
    # qwen3 GGUF has no per-layer head_count_kv divergence; scalar key.
    return ModelConfig(
        num_layers=int(g("block_count")), num_qo_heads=n_q, num_kv_heads=n_kv,
        head_dim=head_dim, hidden_size=hidden, vocab_size=int(shim.vocab_size),
        intermediate_size=int(g("feed_forward_length")), hidden_act="silu",
        rms_norm_eps=float(g("attention.layer_norm_rms_epsilon")),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        rotary_config=RotaryConfig(head_dim=head_dim, rotary_dim=head_dim,
            max_position=int(g("context_length")), base=float(g("rope.freq_base")),
            scaling=None),
        num_experts=int(g("expert_count")),
        num_experts_per_tok=int(g("expert_used_count")),
        moe_intermediate_size=int(g("expert_feed_forward_length")),
        norm_topk_prob=True, model_type=arch,
        architectures=list(shim.architectures),
        moe_enabled=True, weight_format="gguf",
        gguf_type_table=_tensor_types(shim.model_path),
    )
```

iter_gguf_weights(model_path, device, *, include_moe_experts,
include_non_moe) — mirror llama's: assert include_non_moe; TP1 guard
(`_require_tp1` like llama's, copy it); buffer q/k/v packed rows per layer,
fuse `qkv_proj.qweight` when types equal else yield separate
`.qkv_proj.{q,k,v}_proj.qweight` shards (GGUFSplitQKV path — copy llama's
logic verbatim, using config.gguf_type_table); `ffn_gate_inp.weight` ->
`mlp.gate.weight` is bf16-dequantizable, but SKIP it with the experts (part B
owns the router wiring) — add it to the skipped set; `token_embd.weight` ->
`model.embed_tokens.qweight`; `output.weight` -> `lm_head.qweight`;
`output_norm.weight` -> `model.norm.weight` (bf16); `attn_q.bias` /
`attn_k.bias` -> yield as `self_attn.qkv_proj.{q,k}_bias` — CHECK: if unsure
of the exact bias param naming, read python/freetoken/layers/linear.py
LinearQKVMerged to find the bias param names; norms via `_to_bf16`.

## Acceptance

1. `uv run python -c "import freetoken.models.qwen3_moe"` clean.
2. `uv run pytest tests/models/test_models_registry.py -q` still passes.
3. New test file `tests/models/test_qwen3_moe_gguf.py`:
   build a tiny GGUF with the `gguf` python package (GGUFWriter, arch
   "qwen3moe", 2 layers, embedding_length 64, 4 q heads, 2 kv heads,
   expert_count 4, f16 tensors so types are uniform) in tmp_path; call
   parse_gguf_config via a GgufConfigShim and iter_gguf_weights; assert
   embed/norm/qkv fusion/o_proj yields and that no expert tensor is yielded.
   Model the writer usage on llama.cpp's gguf package docs; keep the test
   CPU-only.
4. Report: files changed, arch string registered, skipped tensor list, real
   pytest output.