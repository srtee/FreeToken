"""MTP wave-2 Stage 2 gate: the draft CUDA-graph family vs eager (offline, in-process).

Gate 1 — bit-equality: 200 consecutive draft steps on the 35B NVFP4
checkpoint. Per step: random bs 1..4, fresh disjoint layer-40 KV bands,
the eager oracle (mtp.draft_step under forward_batch, the exact scheduler
shape) vs the graphed replay (MTPDraftGraphRunner.draft) over the SAME
inputs. Asserts bit-identical argmaxes, bit-identical carry', and an
identical layer-40 KV row write (out_loc/positions semantics preserved).

Gate 2 — losslessness: the 3 ground-truth prompts, spec (graphed draft)
vs plain greedy at bs=1 — token-id sequences must be byte-identical.

Run:  .venv/bin/python scripts/mtp_stage2_gate.py
Env:  CKPT / GATE_TOKENS override; FT_SPEC_DRAFT_EAGER=1 would disable
capture (must NOT be set).
"""
import json
import os
import subprocess
import sys
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch

CKPT = os.environ.get(
    "CKPT",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/"
        "snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5"
    ),
)
N_STEPS = 200
MAX_BS = 4
PROMPTS = [
    "The industrial revolution began in Britain during the late eighteenth century. "
    "Describe the three most important technological innovations of this period "
    "and their effects on urbanization:",
    "Write a Python function that computes the n-th Fibonacci number using "
    "memoization. Include docstring and type hints.",
    "Summarize the causes of the First World War in four paragraphs, covering "
    "alliance systems, militarism, imperialism, and nationalism.",
]
GREEDY = dict(temperature=0.0, top_k=1, top_p=1.0)  # disarm-proof greedy


def build_engine(spec: bool):
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig
    from freetoken.engine.engine import Engine

    cfg = EngineConfig(
        model_path=CKPT,
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        max_running_req=MAX_BS,
        spec_mtp=spec,
        cuda_graph_max_bs=MAX_BS,
        max_seq_len_override=1024,
        num_token_override=2048,
        moe_strategy="offload",
        expert_load="parallel",
        kv_reserve_tokens=2048,
        moe_cache_auto=True,
    )
    return Engine(cfg)


def gate_bit_equality(eng) -> None:
    from freetoken.core import Batch, Req, get_global_ctx

    mtp = eng.model.model.mtp
    pool = eng.kv_cache
    device = eng.device
    dtype = eng.dtype
    hidden = mtp.fc.out_features
    runner = eng.draft_graph_runner
    assert runner is not None, (
        "gate requires the draft family captured (unset FT_SPEC_DRAFT_EAGER)")
    assert eng.graph_runner.max_graph_bs == 0, (
        "trunk graphs must stay disabled under --spec-mtp")
    k_cache = pool.k_cache(mtp.layer.self_attn.layer_id)

    gen = torch.Generator().manual_seed(7)
    # 2 sets x 4 reqs x 8 bands = 64 slots, page 0 = dummy
    def band_base(step: int) -> int:
        return 1 + (step % 8) * 8

    for step in range(N_STEPS):
        bs = int(torch.randint(1, MAX_BS + 1, (1,), generator=gen))
        carries = torch.randn(
            bs, hidden, generator=gen).to(device=device, dtype=dtype)
        tokens = torch.randint(
            0, eng.config.model_config.vocab_size, (bs,), generator=gen,
            dtype=torch.int32).to(device)

        # -- eager oracle: EXACT-bs (the production eager path's shape).
        #    Exact-bs capture makes the graphed replay run the same kernel
        #    geometry, so the comparison is apples-to-apples at every bs.
        reqs = [
            Req(input_ids=torch.zeros(1, dtype=torch.int32), table_idx=i,
                cached_len=0, output_len=1, uid=-(step * 8 + i),
                sampling_params=None, cache_handle=None)
            for i in range(bs)
        ]
        eager = Batch(reqs=reqs, phase="decode")
        eager.padded_reqs = reqs
        eager.input_ids = tokens
        eager.out_loc = (
            torch.arange(bs, dtype=torch.int32) + band_base(step)).to(device)
        eager.positions = torch.full((bs,), step, dtype=torch.int32,
                                     device=device)
        eng.attn_backend.prepare_metadata(eager)
        with torch.cuda.stream(eng.stream):
            with get_global_ctx().forward_batch(eager):
                eager_carry, eager_logits = mtp.draft_step(carries, tokens)
            eager_drafts = eager_logits.argmax(dim=-1).to(torch.int32)
        eager_kv = k_cache[eager.out_loc.long()].clone()

        # -- graphed replay: same inputs, the band's second half so the
        #    two paths write disjoint KV rows
        greqs = [
            Req(input_ids=torch.zeros(1, dtype=torch.int32), table_idx=i,
                cached_len=0, output_len=1, uid=-(step * 8 + i),
                sampling_params=None, cache_handle=None)
            for i in range(bs)
        ]
        gbatch = Batch(reqs=greqs, phase="decode")
        gbatch.padded_reqs = greqs
        gbatch.positions = torch.full((bs,), step, dtype=torch.int32,
                                      device=device)
        gbatch.out_loc = (
            torch.arange(bs, dtype=torch.int32) + band_base(step) + 4
        ).to(device)
        g_drafts, g_carry = runner.draft(carries, tokens, gbatch)
        graph_kv = k_cache[gbatch.out_loc.long()]

        assert torch.equal(eager_drafts, g_drafts), (
            f"step {step} bs={bs}: draft argmax diverged")
        assert torch.equal(eager_carry, g_carry), (
            f"step {step} bs={bs}: carry' diverged")
        assert torch.equal(eager_kv, graph_kv), (
            f"step {step} bs={bs}: layer-40 KV row diverged")

    print(f"GATE 1 PASS: {N_STEPS} steps, eager-vs-graphed draft argmaxes "
          f"bit-identical (bs 1..{MAX_BS}, layer-40 KV rows identical)")


def gate_losslessness() -> None:
    # One Engine per process: CUDA/distributed init is single-shot, so the
    # two modes run in SUBPROCESSES, each on its own rendezvous port (the
    # parent's 2333 listener may outlive it briefly); this parent only
    # collects and diffs.
    outs = {}
    for i, mode in enumerate(("plain", "spec")):
        worker_env = dict(os.environ, FT_DIST_PORT=str(2433 + i))
        out = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__),
                                          "mtp_stage2_gate_llm.py"), mode],
            capture_output=True, text=True, check=False, env=worker_env)
        if out.returncode != 0:
            print(out.stderr[-2000:], file=sys.stderr)
            raise RuntimeError(f"gate-2 worker '{mode}' failed rc={out.returncode}")
        outs[mode] = json.loads(out.stdout.strip().splitlines()[-1])

    for i, (a, b) in enumerate(zip(outs["plain"], outs["spec"])):
        assert a == b, f"prompt {i}: spec output diverged from plain"
    total = sum(len(a) for a in outs["plain"])
    print(f"GATE 2 PASS: spec (graphed draft) vs plain greedy byte-identical "
          f"on {len(PROMPTS)} prompts, {total} tokens at bs=1")


def main() -> None:
    eng = build_engine(spec=True)
    try:
        gate_bit_equality(eng)
    finally:
        eng.draft_graph_runner.destroy_cuda_graphs()
        eng.shutdown()
        # Release the ENGINE'S GPU tensors (weights + KV pool ~14.6GB) so
        # the gate-2 workers can load the model again on this 16G GPU:
        # drop every reference, empty the caching allocator, and drop
        # torch.distributed's global state so the TCPStore listener
        # (tcp://127.0.0.1:2333) is released.
        del eng
        gc.collect()
        torch.cuda.empty_cache()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    gate_losslessness()
    print("STAGE 2 GATE: ALL PASS")


if __name__ == "__main__":
    main()
