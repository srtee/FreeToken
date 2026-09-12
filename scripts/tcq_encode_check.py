"""Offline: CUDA turbo3_tcq encode/decode vs torch oracle on REAL K/V.

For each dumped layer: run the CUDA roundtrip and the oracle roundtrip on
the same k rows, report per-group relative error, worst groups, and
whether CUDA == oracle at the byte level.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch

from freetoken.kernel import turbo_kv
from freetoken.kernel.turbo_oracle import TurboCodec, turbo_rotate

CODEC = os.environ.get("TCQ_CODEC", "turbo3_tcq")
DUMP = "/tmp/tcq_kv_dump"
bb = turbo_kv.CODEC_SPECS[CODEC][1]


def cuda_roundtrip(k):
    L, heads, _ = k.shape
    locs = torch.arange(L, dtype=torch.int32, device="cuda")
    kd = torch.zeros(L, heads, bb, dtype=torch.uint8, device="cuda")
    khat = torch.zeros(L, heads, 128, dtype=torch.float16, device="cuda")
    turbo_kv.turbo_quantize(CODEC, k.contiguous(), kd, locs, is_v=False)
    turbo_kv.turbo_dequantize(CODEC, kd, khat, locs, is_v=False)
    return kd, khat


def oracle_roundtrip(k):
    codec = TurboCodec(CODEC, is_v=False)
    flat = k.reshape(-1, 128).float().cpu()
    blocks = codec.encode(flat)
    dec = turbo_rotate(codec.decode(blocks), inverse=True).reshape(k.shape)
    return blocks, dec


def group_rel_err(k, khat):
    num = (khat.float() - k.float()).norm(dim=-1)
    den = k.float().norm(dim=-1).clamp_min(1e-6)
    return (num / den)


def main():
    files = sorted(f for f in os.listdir("/tmp/tcq_kv_dump") if f.endswith(".pt"))
    print(f"{len(files)} shards; codec={CODEC}")
    worst = []
    for f in files:
        d = torch.load(f"/tmp/tcq_kv_dump/{f}", weights_only=False)
        k = d["k"].cuda().float()
        # engine k arrives fused (tokens, heads*128) — split heads
        if k.dim() == 2:
            heads = k.shape[1] // 128
            k = k.reshape(k.shape[0], heads, 128)
        L, heads, dim = k.shape
        kd, khat = cuda_roundtrip(k)
        blocks, odec = oracle_roundtrip(k)
        byte_match = torch.equal(kd.reshape(-1, bb).cpu(), blocks)
        err = group_rel_err(k, khat)
        oerr = group_rel_err(
            k.reshape(-1, 128), odec.reshape(-1, 128).cuda())
        tag = f.replace(".pt", "")
        print(f"{tag}: cuda_relerr mean={err.mean():.4f} p99={err.quantile(0.99):.4f} max={err.max():.4f} | oracle mean={oerr.mean():.4f} | bytes_match={byte_match}")
        worst.append((err.max().item(), tag))
    worst.sort(reverse=True)
    print("worst layers:", worst[:5])


if __name__ == "__main__":
    main()
