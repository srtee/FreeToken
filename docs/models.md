# Supported models

FreeToken loads HF safetensors checkpoints directly. The checkpoints below are known-good — the prebuilt kernels are tuned
for them; other checkpoints of the same architectures work too.

| Model | HF checkpoints |
|---|---|
| DeepSeek-V4 | [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) |
| GLM-5.3-Flash | [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) |
| GLM-5.2 | [nvidia/GLM-5.2-NVFP4](https://huggingface.co/nvidia/GLM-5.2-NVFP4) |
| GLM-4.7 | [nvidia/GLM-4.7-NVFP4](https://huggingface.co/nvidia/GLM-4.7-NVFP4) |
| Qwen3.8-Flash-Next | [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8), [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4), [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) |
| Qwen3.6 / Qwen3.5 MoE | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) ([-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)), [nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4), [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) ([-FP8](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8)) |
| Qwen3.8 / Qwen3.6 dense | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) ([-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)), [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4), [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) ([-FP8](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)), [nvidia/Qwen3.6-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4) |
| Qwen3-MoE | [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Qwen3-VL | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), [Qwen/Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) |
| gpt-oss | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b), [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) |
| Gemma-4 | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it), [nvidia/Gemma-4-26B-A4B-NVFP4](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4), [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it), [nvidia/Gemma-4-31B-IT-NVFP4](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4) .. |
| MiniMax-M2.5 | [nvidia/MiniMax-M2.5-NVFP4](https://huggingface.co/nvidia/MiniMax-M2.5-NVFP4) |
| MiniMax-M3 | [nvidia/MiniMax-M3-NVFP4](https://huggingface.co/nvidia/MiniMax-M3-NVFP4) |
| Muse-Glimmer | [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B), [RedHatAI/Muse-Glimmer-30B-NVFP4](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-NVFP4) |

### Image input

These families accept image input by default; pass `--text-model-only` to skip the vision encoder. The flags are described in the
[CLI reference](cli.md#image-input); each family reads them in its own units.

| Family | Image tokens | `--image-min-tokens` / `--image-max-tokens` | `--mm-processor-kwargs` example |
| --- | --- | --- | --- |
| Qwen3.6 (both variants, every listed weight format), Qwen3.8-Flash-Next, Qwen3-VL | one token per 32x32 pixels of the resized image, dynamic resolution | pixel areas in `size.shortest_edge` / `longest_edge`; checkpoint defaults 64 to 16384 tokens | `{"size": {"longest_edge": 1048576}}` |
| Gemma-4 26B-A4B, 31B (`gemma4`: ViT tower, streamed under `--mm-encoder-weights host`) | one of the soft-token budgets 70 / 140 / 280 / 560 / 1120, every image scaled to its budget as far as the aspect ratio allows | the maximum picks the largest budget within it, below 70 is refused at start-up; the minimum has no effect | `{"max_soft_tokens": 1120}` |
| Gemma-4 12B (`gemma4_unified`: linear patch embedder, resident under either placement) | same budgets, one 48x48 super-patch per soft token | same as the tower releases | same |
| GLM-5.3-Flash (`glm5_next`: ViT tower, streamed under `--mm-encoder-weights host`) | one token per 28x28 pixels of the resized image, dynamic resolution on a canvas zero-padded to a 28-multiple | token counts, passed through as the processor's `min_image_tokens` / `max_image_tokens`; checkpoint defaults 16 to 8000 tokens | `{"max_image_tokens": 2048}` |
| Muse-Glimmer-30B (windowed ViT tower, streamed under `--mm-encoder-weights host`) | one token per 28x28 pixels of the resized image, aspect ratio kept under a token cap; checkpoint default 4096 tokens | the maximum is the cap (`max_image_tokens`); the minimum has no effect | `{"max_image_tokens": 1024}` |
| MiniMax-M3 (`minimax_m3`: CLIP-style ViT tower, streamed under `--mm-encoder-weights host`) | one token per 28x28 pixels of the resized image, dynamic resolution | pixel areas in `size.shortest_edge` / `longest_edge`; checkpoint defaults 4 to 576 tokens | `{"size": {"longest_edge": 1048576}}` |

## MoE strategies

`ft serve --moe-strategy {auto,fused,offload,cpu,hybrid}` (`--moe-backend` is the deprecated old spelling):

- **fused** — experts resident on GPU (needs the VRAM); never auto-selected.
- **offload** — experts live in host RAM, an LRU cache of expert slots on GPU;
  misses stream over PCIe.
- **cpu** — misses are computed on the CPU instead of fetched.
- **hybrid** — per step, fetches some misses over PCIe and computes the rest on
  CPU, overlapped. Run `ft bench bw` once per machine to calibrate the split.
- **auto** — dense models always resolve to `fused`; MoE models resolve to
  `offload`, upgraded to `hybrid` when a cached `ft bench bw` profile
  recommends it.

## KV storage codecs

`ft serve --kv-codec {f16,turbo8,turbo4,turbo3_tcq,turbo2_tcq}` packs the KV
cache with TurboQuant: each 128-value rotation group is L2-normalized, rotated
with the signed FWHT, then scalar-quantized (turbo8: 8-bit absmax grid;
turbo4: 4-bit Lloyd-Max) or trellis-quantized (turbo3_tcq / turbo2_tcq: Viterbi
over a convolutional codebook). One fp16 norm scalar per group carries the
group magnitude.

| Codec | Bits/value | K+V compression vs f16 | Quality |
|---|---|---|---|
| turbo8 | 8.125 | 2.0x | near-lossless at any model scale |
| turbo4 | 4.125 | 3.9x | calibrated on 27B+; degrades past ~10 decode tokens on 14B-class |
| turbo3_tcq | 3.25 | 4.9x | 27B+; TCQ bitstream is byte-exact vs the torch oracle |
| turbo2_tcq | 2.25 | 7.1x | 27B+; experimental |

Constraints: head_dim % 128 == 0 (one rotation group per 128 values —
wider heads such as Qwen3.6-35B's 256-dim full-attention heads carry
independent groups), page_size 1, CUDA. Hybrid GDN models (qwen3.5/3.6)
are supported — only the full-attention layers carry KV, so the
compression applies to that subset; linear layers cost no KV at all.

Decode reads the packed slabs directly (Triton fused-decode kernels: the
dequant inverse rotation is folded into the query and the output
accumulator, so per-token decode work is byte-unpack + dot). Prefill goes
through a dequantizing materializer. Use `--attention-backend triton` with
turbo codecs: it is the only backend with the fused decode and CUDA-graph
capture; fi/fa materialize in the decode path and force graphs off.

With `--kv-codec-tune innerq`, per-channel K/V scales are calibrated over the
first ~2048 stored tokens and channels are equalized before quantization —
recovers accuracy when a few channels dominate the group (common on
anisotropic K).

Tensor parallelism needs no special handling: KV slabs are per-rank (kv heads
split across ranks), quantization is per-128-group and never crosses heads, and
InnerQ scales are per-rank — each rank calibrates from its own heads'
statistics, which is correct since scales are decode-local.

## MTP speculative decoding

Qwen3.5/3.6 checkpoints that ship an `mtp.*` draft head decode speculatively
with `--spec-mtp`: each iteration drafts `--spec-draft-n` tokens (default 1)
with the MTP head, then verifies with n+1 sequential one-row trunk forwards.
Both the draft step and the verify forwards run as CUDA graphs (the trunk
graph family is excluded instead). Greedy-lossless: output is byte-identical
to plain decode at either depth — rejects roll back cleanly.

Whether it pays is a concurrency question, not just an acceptance one: the
verify costs n+1 trunk forwards per iteration, so spec wins only where
per-iteration token yield beats the extra forward cost — long generations at
low concurrency. Under batched decoding (several concurrent requests) plain
decode amortizes the forward over many sequences and usually wins; the
stage-5 soak numbers in [mtp-baseline-numbers.md](mtp-baseline-numbers.md)
are the reference. Acceptance telemetry rides the decode log line
(`#drafted: N, #accepted: k (rate r)`); r ≥ 0.55 on prose is healthy.

## GGUF checkpoints

`ft serve --model <file>.gguf` loads GGUF files directly — config and tokenizer
travel inside the file's metadata header, no HF repo needed. `ft checkpoint`
also accepts a GGUF: the resulting FTW keeps a `source_metadata.gguf` sidecar
so the config/tokenizer still resolve from the original header.

| Family | Examples |
|---|---|
| llama / qwen2 (dense) | Qwen2.5-Coder-14B Q3_K_M |
| qwen3-moe | Qwen3-30B-A3B GGUFs (three-tensor expert banks) |
| qwen35moe | Qwen3.5/3.6-35B-A3B GGUFs — hybrid GDN layers, NVFP4-in-GGUF (llama.cpp extension with per-tensor `scale` sidecars) |
| gemma4 | google/gemma-4 GGUF releases (fused `ffn_gate_up_exps`, Q4_0 experts) |

Quant coverage: legacy Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 and all K-quants Q2_K–Q6_K run
the packed-weight MMVQ/MMQ kernels (dequant-in-kernel; no bf16 weight copy is
ever materialized). Mixed-quant files such as Q3_K_M (Q3_K attention + Q4_K/Q5_K
FFN) dispatch per tensor. The `iq*` types are MMVQ/dequant-only: small batches
use the vector kernel, larger batches fall back to dequant-then-matmul.

Restriction: GGUF serving is TP=1 — packed quant rows and expert banks do not
shard; TP>1 fails fast with a clear error.

## Notes

- `ft checkpoint` conversion is optional — it pre-converts a checkpoint into
  FreeToken's fast-load format, and `ft serve --model` auto-detects the result.
- FTW files converted by builds before the quantization refactor may fail to load;
  see [ftw-hotfix.md](ftw-hotfix.md) for the affected checkpoints and the repair tool.
- An FTW converted before its family served images holds no vision encoder: `ft serve`
  refuses it unless started with `--text-model-only` (or `--mm-disable vision`); reconvert it
  with `ft checkpoint`, or add the encoder in place with [scripts/ftw_hotfix.py](ftw-hotfix.md).
- DeepSeek-V4 checkpoints must keep the `inference/config.json` subdir — the
  authoritative model args are read from there.
- Qwen3.8-Flash-Next keeps a 47.7 GiB PLE n-gram table pinned in host RAM.
