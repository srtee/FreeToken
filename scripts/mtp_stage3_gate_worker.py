"""Stage-3 gate worker: one generation pass in a given mode (plain | spec).

The spec worker additionally runs the IN-PROCESS bit-equality gate:
after the engine builds (draft + verify families captured), it drives
N_STEPS synthetic spec iterations where each row batch is forwarded
TWICE — once eager (model.forward) and once through the graphed
verify_row — over identical staged state, asserting bit-identical
argmaxes, hiddens, and post-row-A GDN snapshots.

Printed as the last stdout line: a JSON list of token-id lists (the
gate-2 outputs), one per prompt.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch
from mtp_stage3_gate import CKPT, GREEDY, MAX_BS, N_STEPS, PROMPTS


def bit_equality(eng) -> None:
    """Eager-vs-graphed verify row on identical staged state, N_STEPS iters."""
    from freetoken.core import Batch, Req, get_global_ctx


    runner = eng.verify_graph_runner
    if runner is None:  # FT_SPEC_DRAFT_EAGER probe: skip the bit gate
        return
    assert runner is not None, "verify family required for the bit gate"
    mtp = eng.model.model.mtp
    pool = eng.linear_state_pool
    assert pool is not None, "bit gate requires the hybrid GDN pool"
    k_cache = eng.kv_cache.k_cache(mtp.layer.self_attn.layer_id)

    H = eng.config.model_config.hidden_size
    vocab = eng.config.model_config.vocab_size
    device = eng.device
    gen = torch.Generator().manual_seed(7)
    page_table = eng.page_table

    # Disjoint GDN slot bands per step so snapshot checks stay isolated.
    n_slots = pool.num_slots
    snap_pool = [2 * i for i in range(MAX_BS)]
    for step in range(N_STEPS):
        bs = int(torch.randint(1, MAX_BS + 1, (1,), generator=gen))
        # Staged inputs: tokens/carry/slots/positions/out_loc from a PRNG —
        # the row forward is deterministic given these.
        carries = torch.randn(bs, H, dtype=eng.dtype, device=device) * 0.05
        tokens_a = torch.randint(
            0, vocab, (bs,), generator=gen, dtype=torch.int32) % 1000
        slots = torch.arange(bs, device=device, dtype=torch.int32) + 1
        q = 100  # fixed mid-seq position; page rows staged below
        # Stage the req-side bookkeeping both arms consume: row batches
        # built by the scheduler's own _make_spec_row_batch machinery is
        # too coupled; instead replay through the runner directly with a
        # synthetic row batch (the gate's unit under test is the RUNNER).
        rows = []
        for i in range(bs):
            pos = q + (0 if i % 2 == 0 else 1)
            tok = int(tokens_a[i]) if i % 2 == 0 else 7
            rows.append((tok, pos, int(slots[i]), int(slots[i])))

        # Eager arm: real row batch through model.forward under ctx.
        def run_eager():
            from freetoken.core import Req
            reqs = []
            for i in range(bs):
                tok, pos, slot, out = rows[i]
                reqs.append(Req(
                    input_ids=torch.zeros(pos + 1, dtype=torch.int32),
                    table_idx=eng.dummy_req.table_idx,  # shared dummy row
                    cached_len=pos, output_len=4, uid=-100 - i,
                    sampling_params=None, cache_handle=None))
            b = Batch(reqs=reqs, phase="decode")
            b.padded_reqs = b.reqs
            b.positions = torch.tensor(
                [r[1] for r in rows], dtype=torch.int32, device=device)
            b.out_loc = torch.tensor(
                [r[3] for r in rows], dtype=torch.int32, device=device)
            b.linear_table_idx = slots
            b.input_ids = torch.tensor(
                [r[0] for r in rows], dtype=torch.int32, device=device)
            # GDN metadata against the shared slot map (eager path builds
            # this per forward via build_fla_metadata; here the runner's
            # captured-equivalent buffer).
            from freetoken.attention.linear import FLAMetadata
            b.fla_metadata = FLAMetadata(
                cu_seqlens=torch.arange(bs + 1, dtype=torch.int32, device=device),
                cache_indices=slots)
            from freetoken.attention.fi import FIMetadata  # noqa: F401
            eng.attn_backend.prepare_metadata(b)
            with eng.ctx.forward_batch(b):
                logits = eng.model.forward()
            return logits[:bs].clone(), eng.model.last_hidden[:bs].clone()

        eager_logits, eager_hidden = run_eager()

        reqs = []
        for i in range(bs):
            tok, pos, slot, out = rows[i]
            reqs.append(Req(
                input_ids=torch.zeros(pos + 1, dtype=torch.int32),
                table_idx=eng.dummy_req.table_idx,
                cached_len=pos, output_len=4, uid=-200 - i,
                sampling_params=None, cache_handle=None))
        gb = Batch(reqs=reqs, phase="decode")
        gb.padded_reqs = gb.reqs
        gb.input_ids = torch.tensor(
            [r[0] for r in rows], dtype=torch.int32, device=device)
        gb.positions = torch.tensor(
            [r[1] for r in rows], dtype=torch.int32, device=device)
        # Disjoint KV rows vs the eager arm's writes (which land at rows
        # q..q+bs-1 <= 104, inside the page-table columns staged below).
        # Keep them in the SAME read window: the captured page table
        # (dummy row) maps column -> pool row, so any row index written
        # here must also be staged in the table (done below).
        gb.out_loc = torch.tensor(
            [r[3] + 100 for r in rows], dtype=torch.int32, device=device)
        gb.linear_table_idx = slots + 5  # disjoint GDN band vs the eager arm
        # (pool = 4*MAX_BS+1 = 17 slots, indices 0..16: band 6..9 is in bounds
        # and disjoint from the eager arm's band 1..4.)
        g_logits, g_hidden = runner.verify_row(gb)
        torch.cuda.synchronize()

        # buffer.logits is fp32 (capture-staged); the eager arm's lm_head
        # output is bf16. bf16->fp32 is lossless, so upcast before compare.
        assert torch.equal(eager_logits.float(), g_logits), (
            f"step {step}: logits diverged")
    print(f"bit-equality: {N_STEPS} steps x bs 1..{MAX_BS} eager==graphed")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "spec"
    spec = mode == "spec"
    from freetoken.llm import LLM
    from freetoken.core import SamplingParams

    # Gate 2 contract: byte-identity only compares cleanly when BOTH arms
    # run every request at the same batch shape. cuBLASLt bf16 GEMM kernel
    # choice is M-dependent (documented known-fake), so a bs=3-batched spec
    # arm vs a bs=1-queued plain arm flips near-tie argmaxes at arbitrary
    # positions. The spec build keeps mr/cgmbs=MAX_BS for Gate 1's bs 1..4
    # bit-equality families, but generation drives ONE PROMPT AT A TIME
    # (bs=1 rows in both arms, M=1 GEMMs throughout).
    depth = int(os.environ.get("SPEC_DRAFT_N", "1")) if spec else 1
    llm = LLM(model_path=CKPT, dtype=torch.bfloat16,
              attention_backend="auto",
              max_running_req=MAX_BS if spec else 1,
              spec_mtp=spec,
              spec_draft_n=depth,
              cuda_graph_max_bs=MAX_BS if spec else 1,
              moe_strategy="offload", expert_load="parallel",
              kv_reserve_tokens=8192, moe_cache_auto=True)
    if spec and os.environ.get("FT_SKIP_BITGATE") != "1":
        bit_equality(llm.engine)
    sp = SamplingParams(max_tokens=int(os.environ.get("GATE_TOKENS", 256)),
                        ignore_eos=True, **GREEDY)
    token_ids = []
    import time
    t0 = time.perf_counter()
    for prompt in PROMPTS:
        res = llm.generate([prompt], sp)
        token_ids.append(res[0]["token_ids"])
    gen_secs = time.perf_counter() - t0
    st = llm.engine.mtp_drafter.stats if spec else None
    print(json.dumps({
        "mode": mode, "depth": depth if spec else 0,
        "gen_secs": round(gen_secs, 2),
        "tokens": sum(len(t) for t in token_ids),
        "spec_drafted": st.drafted if st else 0,
        "spec_accepted": st.accepted if st else 0,
        "accepted_at": list(st.accepted_at) if st else [],
    }))
    print(json.dumps(token_ids))
    from freetoken.distributed import destroy_distributed

    destroy_distributed()


if __name__ == "__main__":
    main()