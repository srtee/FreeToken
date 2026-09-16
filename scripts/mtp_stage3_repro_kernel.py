"""Minimal repro: gdn_decode_fla under CUDA graph capture, bs>=2 vs eager.

No model load — synthetic tensors at real model dims (HK=16, HV=32, K=V=128).

Scenarios:
  A. capture-replay with same tensors  -> should be bit-equal to eager
  B. capture, then mutate input tensors in-place, replay -> replay must see
     mutated values (kernel arg pointers fixed at capture)
  C. capture with a DIFFERENT bs's tensors present (simulating graph family
     capture order bs=1 then bs=2) -> cross-family contamination?
"""

import os, sys, torch

sys.path.insert(0, "/home/sherntee/20llms/FreeToken/python")
from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_decode_fla

torch.manual_seed(0)
dev = "cuda:0"
HK, HV, K, V = 16, 32, 128, 128


def make_inputs(bs, seed=0, num_slots=None):
    g = torch.Generator(device=dev).manual_seed(seed)
    ns = num_slots or max(9, bs)
    return dict(
        q=torch.randn(1, bs, HK, K, device=dev, dtype=torch.bfloat16, generator=g),
        k=torch.randn(1, bs, HK, K, device=dev, dtype=torch.bfloat16, generator=g),
        v=torch.randn(1, bs, HV, V, device=dev, dtype=torch.bfloat16, generator=g),
        a=torch.randn(bs, HV, device=dev, dtype=torch.bfloat16, generator=g),
        b=torch.randn(bs, HV, device=dev, dtype=torch.bfloat16, generator=g),
        A_log=(torch.randn(HV, device=dev, dtype=torch.float32, generator=g) * 0.1),
        dt_bias=(torch.randn(HV, device=dev, dtype=torch.float32, generator=g) * 0.1),
        state_source=torch.randn(ns, HV, K, V, device=dev, dtype=torch.bfloat16, generator=g),
        indices=torch.arange(bs, device=dev, dtype=torch.int32),
        cu_seqlens=torch.arange(bs + 1, device=dev, dtype=torch.int32),
        scale=K ** -0.5,
    )


def capture(args, warmup=3):
    static = {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in args.items()}
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            _ = gdn_decode_fla(**static)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = gdn_decode_fla(**static)
    g.replay()
    torch.cuda.synchronize()
    return static, out, g


def report(tag, eager, replay):
    d = (eager.float() - replay.float()).abs()
    print(f"{tag}: maxdiff={d.max().item():.6f} mean={d.mean().item():.6f} "
          f"bit-equal={torch.equal(eager, replay)}")


def scenario(bs, seed):
    print(f"--- bs={bs} seed={seed} ---")
    args = make_inputs(bs, seed)
    # Reference: eager on pristine args, cloned OUT (kernel writes state back in place).
    eager = gdn_decode_fla(**args).clone()
    torch.cuda.synchronize()

    static, out, g = capture(args)

    def replay():
        g.replay()
        torch.cuda.synchronize()
        return out.clone()

    # A: statics untouched since capture. But capture's warmup+recording advanced
    # the state pool rows; eager must be recomputed on the *current* pool state.
    # Compare replay vs eager-on-same-pool-state: capture pool once more via eager.
    eager_on_static = gdn_decode_fla(**static).clone()
    report("A replay vs eager(same pool state)", eager_on_static, replay())

    # B: mutate INPUT tensors only (not pool rows the kernel reads via indices 0..bs-1
    # ... but rows 0..bs-1 ARE indices! mutate q/v only) -> replay must track mutation.
    with torch.no_grad():
        static["q"] += 1.0
        static["v"] += 2.0
    eager_mut = gdn_decode_fla(**static).clone()
    report("B replay after in-place input mutation", eager_mut, replay())

    # C: mutate a NON-indexed pool row? rows bs..ns-1 are not read (indices 0..bs-1).
    # Instead: restore pool to pristine, then check replay reproduces ORIGINAL eager.
    # This is the engine's real pattern: state pool rows rewritten between replays.
    with torch.no_grad():
        static["state_source"].copy_(args["state_source"])
        static["q"].copy_(args["q"])
        static["v"].copy_(args["v"])
    torch.cuda.synchronize()
    report("C replay vs original eager (pool restored)", eager, replay())


def cross_family():
    """Capture bs=1, then bs=2 (engine order), then replay g1 and g2.
    Pools are per-family statics here; looking for capture-order contamination."""
    print("--- cross-family capture (bs=1 then bs=2) ---")
    a1 = make_inputs(1, seed=11)
    e1 = gdn_decode_fla(**a1).clone()
    torch.cuda.synchronize()
    s1, out1, g1 = capture(a1)
    a2 = make_inputs(2, seed=12)
    e2 = gdn_decode_fla(**a2).clone()
    torch.cuda.synchronize()
    s2, out2, g2 = capture(a2)
    torch.cuda.synchronize()
    # restore both pools to capture-time values, then replay each graph
    with torch.no_grad():
        s1["state_source"].copy_(a1["state_source"])
        s2["state_source"].copy_(a2["state_source"])
    torch.cuda.synchronize()
    g1.replay(); torch.cuda.synchronize()
    report("replay g1 (pool restored)", e1, out1.clone())
    g2.replay(); torch.cuda.synchronize()
    report("replay g2 (pool restored)", e2, out2.clone())
    print("done")


if __name__ == "__main__":
    scenario(1, 0)
    scenario(2, 0)
    scenario(2, 1)
    scenario(4, 0)
    cross_family()
    print("done")