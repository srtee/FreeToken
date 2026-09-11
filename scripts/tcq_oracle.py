"""Wave-0 KLD oracle harness for the TurboQuant/TCQ KV codecs.

Modes:
  --synthetic   : quantize synthetic post-WHT Gaussian K/V groups with the torch
                  oracle, measure reconstruction MSE vs the buun README table and
                  attention-logit KLD against f16 (self-consistency gate; no GPU
                  model needed).
  --serve HOST  : drive a running FreeToken server over OpenAI-compatible
                  completions with wikitext-2-style prompts, capture reference
                  logits (mode f16 KV) vs quantized KV — requires the server to
                  expose the hooks; placeholder for the Wave-1 gate.
  --assert      : apply the acceptance thresholds from the plan:
                  turbo4 KLD <= 0.001, turbo3_tcq <= 0.002, turbo2_tcq <= 0.007.

KLD metric (mirrors buun's README codec table): for each layer, quantize the
captured K and V, recompute attention logits q·k for a set of queries, and take
the median KL divergence of the softmaxed logits against the f16 reference.
For synthetic mode we generate q ~ N(0, I) post-hoc: q·k over the quantized and
f16 K, softmax over positions, KLD(q_softmax || f16_softmax).

Usage:
  python scripts/tcq_oracle.py --synthetic --n-groups 2048 --seqs 2K 8K 16K
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import torch

sys.path.insert(0, "/home/sherntee/20llms/FreeToken/python")

from freetoken.kernel.turbo_oracle import TurboCodec, turbo_rotate  # noqa: E402

# Plan acceptance thresholds (median attention-logit KLD).
KLD_GATES = {
    "turbo4": 0.001,
    "turbo8": float("inf"),  # near-lossless; no separate gate in the plan
    "turbo3_tcq": 0.002,
    "turbo2_tcq": 0.007,
}
MSE_GATES_SYNTH = {  # on N(0, 1/sqrt(128)) post-rotation data, from buun table
    "turbo4": 0.005,
    "turbo8": 1e-4,
    "turbo3_tcq": 0.02,
    "turbo2_tcq": 0.03,
}

CODECS = ["turbo8", "turbo4", "turbo3_tcq", "turbo2_tcq"]


def attn_logit_kld(
    q: torch.Tensor, k_ref: torch.Tensor, k_new: torch.Tensor, temperature: float = 1.0
) -> float:
    """Median KL(P_ref || P_codec) of softmaxed attention logits over queries.

    q: (H, D), k_ref/k_new: (T, D). Logits per query: (T,) softmaxed.
    """
    ref = torch.softmax(q @ k_ref.T / math.sqrt(k_ref.shape[-1]) / temperature, dim=-1)
    new = torch.softmax(q @ k_new.T / math.sqrt(k_new.shape[-1]) / temperature, dim=-1)
    kld = (ref * (ref.clamp_min(1e-12).log() - new.clamp_min(1e-12).log())).sum(-1)
    return kld.median().item()


def synthetic_mode(n_groups: int, seq_lens: list[int], n_q_per_group: int, assert_gates: bool):
    """Post-rotation synthetic K/V groups -> codec roundtrip -> MSE + KLD."""
    torch.manual_seed(42)
    results: dict[str, dict] = {}
    for seq in seq_lens:
        T = seq
        # One attention head over T positions: K = rotated Gaussians (the codec
        # sees post-FWHT data; in the real stack store_kv rotates before packing).
        k_groups = torch.randn(n_groups, 128) / math.sqrt(128)
        v_groups = torch.randn(n_groups, 128) / math.sqrt(128)
        # queries for the KLD metric
        q = torch.randn(n_q_per_group, 128) / math.sqrt(128)

        row = {"mse_k": {}, "mse_v": {}, "kld": {}, "bpv": {}}
        for name in CODECS:
            codec = TurboCodec(name)
            alpha = getattr(codec, "is_v", False)
            bk = codec.encode(k_groups)
            rk = codec.decode(bk)
            bv = TurboCodec(name, is_v=True).encode(v_groups, alpha=1.02 if "tcq" in name else 1.0)
            rv = codec.decode(bv) if not name.endswith("tcq") else TurboCodec(name, is_v=True).decode(bv)
            mse_k = ((rk - k_groups) ** 2).mean().item()
            mse_v = ((rv - v_groups) ** 2).mean().item()
            kld = attn_logit_kld(q, k_groups, rk)
            bpv = bk.shape[1] * 8 / 128
            row["mse_k"][name] = mse_k
            row["mse_v"][name] = mse_v
            row["kld"][name] = kld
            row["bpv"][name] = bpv
        results[f"{seq}"] = row

    print(json.dumps(results, indent=2))
    ok = True
    for seq, row in results.items():
        for name in CODECS:
            gate = KLD_GATES[name]
            kld = row["kld"][name]
            status = "PASS" if kld <= gate else "FAIL"
            if assert_gates and kld > gate:
                ok = False
            print(f"seq={seq} {name:10s} kld={kld:.5f} gate={gate:.4f} {status}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n-groups", type=int, default=2048)
    ap.add_argument("--seqs", type=str, default="512,2048,8192")
    ap.add_argument("--n-queries", type=int, default=64)
    ap.add_argument("--assert", dest="assert_gates", action="store_true")
    args = ap.parse_args()

    if args.synthetic:
        seqs = [int(s) for s in args.seqs.split(",")]
        ok = synthetic_mode(args.n_groups, seqs, args.n_queries, args.assert_gates)
        sys.exit(0 if ok else 1)
    print("real-serve capture mode lands with Wave 1 (needs the in-tree hooks)")
    sys.exit(2)


if __name__ == "__main__":
    main()