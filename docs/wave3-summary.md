Wave 3 (MoE GGUF / qwen3moe) is complete and verified end-to-end. Summary of
what landed in the working tree (branch gguf-serve, uncommitted per the
coordination rule):

## Implementation (Session B, direct — local coder retired)

- `python/freetoken/models/qwen3_moe/gguf.py` (new, ~490 lines):
  - `parse_gguf_config`: qwen3moe metadata -> ModelConfig (per-layer kv-head
    list, expert geometry, expert_quant="q4_0", use_qk_norm).
  - `iter_gguf_weights`: packed qkv fusion (mixed-type -> GGUFSplitQKV), q/k
    bias fusion with zero-padded v segment, qk-norms, router gate
    (ffn_gate_inp -> mlp.gate.weight), expert tensors skipped.
  - `load_q4_0_expert_sources` + `dummy_q4_0_expert_sources`: three-tensor
    (gate/up/down) expert banks for the offload cache, with a per-role type
    probe pass; DOWN projection normalized to Q4_1 (see fixes).
  - `convert_qwen3moe_to_gguf`: dense-layer GGUF ops swap (embed, qkv, o_proj,
    lm_head), same pattern as qwen2/gemma4.
- `qwen3_moe/model.py`: GGUF convert hook in __init__.
- `models/gguf/config.py` + `models/register.py`: "qwen3moe" arch registry.
- `models/weight.py`: expert-source dispatch returns (sources, ggml_types).
- `moe/expert_banks.py` + `moe/offload_cache.py` + `layers/moe.py` +
  `moe/fused_q4_0.py`: per-role ggml quant types flow to the fused MoE kernel
  (`down_qt`), kernel already had Q4_1 paths.
- `models/gguf/tokenizer.py`: "qwen3moe" -> qwen2 tokenizer converter.
- `kernel/aot_models.py`: Qwen3-30B-A3B entry claims the registry key.
- `tests/models/test_qwen3_moe_gguf.py`: 4 tests (config geometry, dense
  yields + expert skip + gate mapping, split-shard path, unmapped guard).

## Real-GGUF smoke (17GB unsloth Qwen3-30B-A3B-Q4_0.gguf)

Served clean on :1918, `--moe-strategy offload`, 14.6 GiB GPU. Coherent
generation verified (finish=stop, "Paris"). Fixes forced by the real file:
- DOWN experts mix Q4_1 (first 6 layers) and Q4_0 (rest) -> normalized to
  Q4_1 at load with `m = fp16(-8*d)`; verified 0.0 max diff vs the Q4_0 torch
  reference on 262144 real rows.
- tokenizer arch map + GGUFUntiedLMHead vocab_size arg.

## Test status

119 passed across the qwen3moe/registry/moe suites; tests/models 156 passed /
80 skipped with 3 pre-existing failures unrelated to wave 3 (AOT parity test's
pre-existing Llama/Qwen2 GGUF keys, muse_glimmer disk-quota,
glm5_next_kda_snapshot collection error).

## TCQ-KV (other worker) review findings

Two CRITICAL bugs documented in docs/tcq-sessionB-review.md: (1) out_loc is
int32 but their kernel reads int64; (2) materialize() captured into CUDA
graphs. Their graph-disable gate landed; the int64 cast is still open.