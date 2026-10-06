"""MTP verify-chain logic (mtp-plan 1.3, wave-1 unit gate).

Greedy acceptance: a draft token at position i+1 is accepted iff the
trunk's argmax at the verify row that predicts position i+1 equals it,
AND every earlier draft in the chain was accepted (cumulative).
"""

from __future__ import annotations

import torch

from freetoken.engine.spec_mtp import verify_chain


def test_all_accepted():
    # trunk argmaxes agree with every draft
    target = torch.tensor([[10, 11, 12, 13]])
    drafts = torch.tensor([[10, 11, 12]])  # drafts for positions 1..3
    mask = verify_chain(target, drafts)
    assert mask.tolist() == [[1, 1, 1]]


def test_first_rejected():
    target = torch.tensor([[10, 99, 12, 13]])
    drafts = torch.tensor([[11, 12, 13]])
    mask = verify_chain(target, drafts)
    # row 0 argmax (10) != draft 0 (11) -> nothing accepted
    assert mask.tolist() == [[0, 0, 0]]


def test_mid_chain_rejection_stops_the_rest():
    target = torch.tensor([[10, 11, 99, 13]])
    drafts = torch.tensor([[10, 12, 13]])
    # draft 0 accepted (row0 argmax 10 == 10); row1 argmax 99 != draft 12
    # -> draft 1 rejected; draft 2 rejected even though row2 argmax
    #    matches (the chain is broken)
    mask = verify_chain(target, drafts)
    assert mask.tolist() == [[1, 0, 0]]


def test_batch_independent_rows():
    target = torch.tensor([[10, 11, 12], [20, 99, 22]])
    drafts = torch.tensor([[10, 11], [21, 22]])
    mask = verify_chain(target, drafts)
    assert mask.tolist() == [[1, 1], [0, 0]]
