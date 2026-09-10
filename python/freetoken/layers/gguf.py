"""Native-GGUF quantized layers: weights stay in their packed block layout and are
dequantized *inside* the borrowed llama.cpp kernels (no bf16 copy ever materialized).

Mirrors vLLM/sglang's ``GGUFLinearMethod`` / ``GGUFEmbeddingMethod`` dispatch, ported
onto FreeToken's ``BaseOP``. FreeToken keeps fused projections (qkv, gate_up) as a
single tensor: because Q4_0/K-quants pack each *output row* independently over the
input dim, the loader can concatenate the per-shard packed rows along dim 0 (they
share an input dim, hence the same ``row_bytes``), so a fused layer is still one
``[out, row_bytes]`` qweight -- no per-shard padding bookkeeping needed.

TP is assumed to be 1 (the gemma4 GGUF path restricts to TP=1, like the HF path).
"""

from __future__ import annotations

import torch

from freetoken.models.gguf.dequant import (
    BLOCK_SHAPE,
    GGML_BF16,
    GGML_F16,
    GGML_F32,
    GGML_NAME,
    GGML_Q2_K,
    GGML_Q3_K,
    GGML_Q4_0,
    GGML_Q4_1,
    GGML_Q4_K,
    GGML_Q5_0,
    GGML_Q5_1,
    GGML_Q5_K,
    GGML_Q6_K,
    GGML_Q8_0,
    row_bytes,
)

from .base import BaseOP

# ggml type groups for kernel dispatch (everything the vendored kernels build).
_UNQUANTIZED = {GGML_F32, GGML_F16, GGML_BF16}
# Quant types with both an MMVQ (small-batch GEMV) and an MMQ (large-batch) kernel:
# the scalar quants Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 and the K-quants Q2_K..Q6_K.
_MMVQ = {GGML_Q4_0, GGML_Q4_1, GGML_Q5_0, GGML_Q5_1, GGML_Q8_0,
         GGML_Q2_K, GGML_Q3_K, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K}
_MMQ = _MMVQ
_DEQUANT = _MMQ
# iq* types: MMVQ + dequant kernels only (no MMQ in the vendored source); small
# batches take MMVQ, large batches fall through to the dequant-then-matmul path.
_IQ_TYPES = {16, 17, 18, 19, 20, 21, 22, 23, 29}

# Below this token count, the MMVQ GEMV kernel wins (matches vLLM's heuristic).
_MMVQ_SAFE = 6


def fused_mul_mat_gguf(x: torch.Tensor, qweight: torch.Tensor, qweight_type: int) -> torch.Tensor:
    """y = x @ dequant(qweight).T, dispatched by batch size and quant type."""
    from freetoken.kernel.gguf import (
        ggml_dequantize,
        ggml_mul_mat_a8,
        ggml_mul_mat_vec_a8,
    )

    out_features = qweight.shape[0]
    if x.shape[0] == 0:
        return x.new_empty((0, out_features))
    if qweight_type in _UNQUANTIZED:
        return x @ qweight.T
    if x.shape[0] <= _MMVQ_SAFE and qweight_type in _MMVQ:
        return ggml_mul_mat_vec_a8(qweight, x, qweight_type, out_features)
    if qweight_type in _MMQ:
        return ggml_mul_mat_a8(qweight, x, qweight_type, out_features)
    if qweight_type in _DEQUANT or qweight_type in _IQ_TYPES:
        # _DEQUANT types reach here only for batches above the MMQ crossover;
        # iq* types have no MMQ at all and always dequantize at large batch.
        block, type_size = BLOCK_SHAPE[qweight_type]
        in_features = qweight.shape[1] // type_size * block
        weight = ggml_dequantize(qweight, qweight_type, out_features, in_features, x.dtype)
        return x @ weight.T
    raise NotImplementedError(f"unsupported GGUF type {GGML_NAME.get(qweight_type, qweight_type)}")


class GGUFLinear(BaseOP):
    """Linear whose weight is a native GGUF block-quantized ``[out, row_bytes]`` tensor."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        quant_type: int,
        has_bias: bool = False,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self._quant_type = quant_type
        self.qweight = torch.empty(out_features, row_bytes(in_features, quant_type), dtype=torch.uint8)
        self.bias = torch.empty(out_features) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = fused_mul_mat_gguf(x, self.qweight, self._quant_type)
        if self.bias is not None:
            out = out + self.bias
        return out



class GGUFSplitQKV(BaseOP):
    """q/k/v attention projection for GGUF checkpoints whose shards quantize with
    *different* ggml types (Q3_K_M: q/k = Q3_K, v = Q4_K/Q5_K). Packed-row fusion into a
    single ``qweight`` requires one type per fused tensor, so mixed checkpoints keep three
    ``GGUFLinear`` shards; uniform checkpoints keep the fused path. Same duck type as the
    fused projection: ``forward(x) -> [tokens, qo + 2*kv]`` (the caller splits by dims).
    """

    def __init__(
        self,
        in_features: int,
        q_out: int,
        kv_out: int,
        q_type: int,
        k_type: int,
        v_type: int,
        has_bias: bool = False,
    ):
        self.q_proj = GGUFLinear(in_features, q_out, q_type, has_bias=has_bias)
        self.k_proj = GGUFLinear(in_features, kv_out, k_type, has_bias=has_bias)
        self.v_proj = GGUFLinear(in_features, kv_out, v_type, has_bias=has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.q_proj.forward(x), self.k_proj.forward(x), self.v_proj.forward(x)], dim=-1
        )


class GGUFEmbedding(BaseOP):
    """Vocab embedding stored as a native GGUF block-quantized table.

    The full table is never dequantized: only the looked-up rows are gathered (in
    packed form) and dequantized per lookup, matching vLLM's ``_apply_gguf_embedding``.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        quant_type: int,
        embed_scale: float | None = None,
    ):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self._quant_type = quant_type
        self.qweight = torch.empty(
            num_embeddings, row_bytes(embedding_dim, quant_type), dtype=torch.uint8
        )
        self._embed_scale = embed_scale
        self._embed_scale_t: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.gguf import ggml_dequantize

        flat = x.flatten()
        rows = self.qweight.index_select(0, flat)  # [n, row_bytes] packed
        y = ggml_dequantize(rows, self._quant_type, flat.shape[0], self.embedding_dim, torch.bfloat16)
        y = y.view(*x.shape, self.embedding_dim)
        if self._embed_scale is not None:
            if self._embed_scale_t is None:
                self._embed_scale_t = torch.tensor(self._embed_scale, dtype=y.dtype, device=y.device)
            y = y * self._embed_scale_t
        return y




class GGUFTiedLMHead:
    """Tied LM head over a native block-quant embedding table (logits via ggml matmul).

    Holds only a reference to the GGUF embedding (no params of its own -> empty
    state_dict). TP=1 only.
    """

    def __init__(self, embedding, quant_type: int):
        self._embedding = embedding
        self._quant_type = quant_type

    def state_dict(self, *, prefix: str = "", result=None):
        return result if result is not None else {}

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False):
        state_dict.pop(f"{prefix}.weight", None)
        state_dict.pop(f"{prefix}.bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.core import get_global_ctx

        batch = get_global_ctx().batch
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(batch.size)
            x = x[indices].contiguous()
        return fused_mul_mat_gguf(x, self._embedding.qweight, self._quant_type)


class GGUFUntiedLMHead(BaseOP):
    """Untied LM head over a native block-quant weight (logits via ggml matmul).

    Same interface as GGUFTiedLMHead but owns its packed qweight; the loader
    fills it through the "lm_head.qweight" state-dict name. TP=1 only.
    """

    def __init__(self, in_features: int, out_features: int, quant_type: int):
        self._quant_type = quant_type
        self.qweight = torch.empty(out_features, row_bytes(in_features, quant_type), dtype=torch.uint8)

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False):
        item = state_dict.pop(f"{prefix}.qweight", None)
        assert item is not None and item.shape == self.qweight.shape and item.dtype == self.qweight.dtype
        self.qweight = item

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.core import get_global_ctx

        batch = get_global_ctx().batch
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(batch.size)
            x = x[indices].contiguous()
        return fused_mul_mat_gguf(x, self.qweight, self._quant_type)


__all__ = ["GGUFLinear", "GGUFSplitQKV", "GGUFEmbedding", "GGUFTiedLMHead", "GGUFUntiedLMHead", "fused_mul_mat_gguf"]
