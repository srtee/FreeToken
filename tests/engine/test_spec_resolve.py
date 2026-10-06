"""SpecStep resolve/rollback pure logic (wave-2 Stage 1, fragment F1).

Depth-1 resolve against hand-computed tables, mirroring
tests/engine/test_mtp_loop.py's model: accept iff row-A argmax == draft;
emit/next-input/carry-row per the semantics doc (Q2/Q3). CPU-only.
"""

from __future__ import annotations

import torch

from freetoken.engine.spec_mtp import (
    PerReqStep, batch_resolve, batch_resolve_chain, per_position_accepts,
    resolve_chain, resolve_step)


def test_resolve_accept():
    s = resolve_step(row_a_argmax=1000, row_b_argmax=777, draft_token=1000)
    assert s.accepted
    assert s.emitted == (1000, 777)   # [d, bonus]
    assert s.next_input == 777        # bonus @ q+2
    assert s.carry_row == 1           # row B's hidden


def test_resolve_reject():
    s = resolve_step(row_a_argmax=999, row_b_argmax=5000, draft_token=1000)
    assert not s.accepted
    assert s.emitted == (999,)        # [a] only
    assert s.next_input == 999        # a @ q+1
    assert s.carry_row == 0           # row A's hidden


def test_batch_resolve_matches_loop_model_mixed_pattern():
    # The loop model's fixed TABLE (test_mtp_loop.py): drafter
    # d = 1000 + int(carry). Rows map 1:1 to resolve inputs.
    #   i=0: carry 0.0 -> d=1000; (0,1000) accept; emit [1000, 2000]
    #   i=1: carry 2.0 -> d=1002; (2000,1002) accept; emit [1002, 4000]
    #   i=2: carry 4.0 -> d=1004; (4000,1004) reject; emit [999]
    #   i=3: carry 5.0 -> d=1005; (999,1005) reject; emit [998]
    drafts = torch.tensor([1000, 1002, 1004, 1005], dtype=torch.int32)
    a = torch.tensor([1000, 1002, 999, 998], dtype=torch.int32)
    b = torch.tensor([2000, 4000, 5000, 5000], dtype=torch.int32)
    steps = batch_resolve(a, b, drafts)
    assert [s.accepted for s in steps] == [True, True, False, False]
    assert [s.emitted for s in steps] == [(1000, 2000), (1002, 4000), (999,), (998,)]
    assert [s.next_input for s in steps] == [2000, 4000, 999, 998]
    # carry selection: accept -> row B (index 1), reject -> row A (index 0)
    assert [s.carry_row for s in steps] == [1, 1, 0, 0]


def test_batch_resolve_all_accept_emits_bonus_each():
    n = 5
    drafts = torch.arange(1000, 1000 + n, dtype=torch.int32)
    steps = batch_resolve(drafts.clone(), torch.full((n,), 555), drafts)
    for i, s in enumerate(steps):
        assert s.accepted
        assert s.emitted == (1000 + i, 555)
        assert s.next_input == 555


def test_resolve_step_type_shape():
    # the dataclass contract: exactly the fields the loop needs, no more
    fields = {f for f in PerReqStep.__dataclass_fields__}
    assert fields == {"accepted", "draft_token", "drafts", "emitted",
                      "next_input", "carry_row"}


def test_resolve_chain_depth2_all_accept():
    # drafts [10,11,12]; every row argmax agrees; the final row's argmax
    # (13) is the bonus token
    s = resolve_chain((10, 11, 12, 13), (10, 11, 12))
    assert s.accepted == 3
    assert s.emitted == (10, 11, 12, 13)
    assert s.next_input == 13
    assert s.carry_row == 3           # the last verify row's hidden


def test_resolve_chain_depth2_first_reject():
    # d_0 matches; d_1 does not — the trunk's row-1 argmax (99) is the
    # certain token, exactly one draft rolls back
    s = resolve_chain((10, 99, 55, 13), (10, 11, 12))
    assert s.accepted == 1
    assert s.emitted == (10, 99)
    assert s.next_input == 99
    assert s.carry_row == 1


def test_resolve_chain_depth2_reject_at_zero():
    # d_0 mismatches immediately: emit [a_0] only; rows 1..2's argmaxes
    # are computed by the forward but never consulted
    s = resolve_chain((50, 99, 55, 13), (10, 11, 12))
    assert s.accepted == 0
    assert s.emitted == (50,)
    assert s.next_input == 50
    assert s.carry_row == 0


def test_resolve_chain_depth2_mid_reject_rolls_back_tail():
    s = resolve_chain((10, 11, 55, 13), (10, 11, 99))
    assert s.accepted == 2
    assert s.emitted == (10, 11, 55)
    assert s.next_input == 55
    assert s.carry_row == 2


def test_batch_resolve_chain_depth2_mixed():
    # three requests, accept counts 3 / 1 / 0 — resolves are independent
    rows = torch.tensor([[10, 11, 12, 13],
                         [20, 29, 30, 31],
                         [30, 40, 50, 60]], dtype=torch.int32)
    drafts = torch.tensor([[10, 11, 12],
                           [20, 21, 22],
                           [35, 36, 37]], dtype=torch.int32)
    steps = batch_resolve_chain(rows, drafts)
    assert [s.accepted for s in steps] == [3, 1, 0]
    assert [s.emitted for s in steps] == [(10, 11, 12, 13), (20, 29), (30,)]
    assert [s.next_input for s in steps] == [13, 29, 30]
    assert [s.carry_row for s in steps] == [3, 1, 0]


def test_per_position_accepts_depth2():
    steps = [
        resolve_chain((10, 11, 12, 13), (10, 11, 12)),  # k=3
        resolve_chain((20, 29, 30, 31), (20, 21, 22)),  # k=1
        resolve_chain((30, 40, 50, 60), (35, 36, 37)),  # k=0
    ]
    assert per_position_accepts(steps, depth=3) == [2, 1, 1]
    assert per_position_accepts(steps, depth=1) == [2]