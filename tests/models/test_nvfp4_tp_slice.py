"""``_tp_slice_pieces`` is a generator: its TP=1 branch must stream, not ``return``.

A ``return pieces`` inside a generator silently discards the whole expert stream
(StopIteration), which starved every TP=1 NVFP4 expert-bank load.
"""

from types import SimpleNamespace

import pytest
import torch

from freetoken.models.nvfp4_banks import _tp_slice_pieces


def _piece(intermediate: int = 64, hidden: int = 32) -> dict[str, torch.Tensor]:
    return {
        "gate": torch.arange(intermediate * hidden).reshape(intermediate, hidden),
        "gate_scale": torch.arange(intermediate * (hidden // 16)).reshape(intermediate, hidden // 16),
        "gate_global": torch.tensor([0.5]),
        "down": torch.arange(hidden * intermediate).reshape(hidden, intermediate),
        "down_scale": torch.arange(hidden * (intermediate // 16)).reshape(hidden, intermediate // 16),
        "down_global": torch.tensor([0.25]),
    }


@pytest.mark.parametrize("rank, size", [(0, 1), (0, 2), (1, 2)])
def test_tp_slice_streams_expert_pieces_at_every_tp(rank, size, monkeypatch):
    import freetoken.distributed.info as info
    from freetoken.distributed import DistributedInfo

    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(rank, size))
    config = SimpleNamespace(moe_intermediate_size=64)
    source = [("layer", 3, 4, _piece())]

    out = list(_tp_slice_pieces(iter(source), config))
    assert len(out) == 1  # the stream must arrive, not StopIteration at the first next()
    layer, e0, e1, piece = out[0]
    assert (layer, e0, e1) == ("layer", 3, 4)

    if size == 1:
        assert torch.equal(piece["gate"], source[0][3]["gate"])
        return

    i_local = 64 // size
    rows = slice(rank * i_local, (rank + 1) * i_local)
    assert torch.equal(piece["gate"], source[0][3]["gate"][rows])
    assert torch.equal(piece["gate_scale"], source[0][3]["gate_scale"][rows])
    assert torch.equal(piece["down"], source[0][3]["down"][:, rows])
    assert torch.equal(piece["down_scale"], source[0][3]["down_scale"][:, rank * 2:(rank + 1) * 2])
    # per-tensor globals pass through unchanged on every rank
    assert torch.equal(piece["gate_global"], source[0][3]["gate_global"])
    assert torch.equal(piece["down_global"], source[0][3]["down_global"])
