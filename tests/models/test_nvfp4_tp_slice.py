"""``_tp_slice_pieces`` contract: the piece stream must survive every TP, and the
sliced pieces must fit the TP-local bank the pack step copies into.

Two historical bugs, both silent at the time:

* a ``return pieces`` inside the generator discarded the whole stream (StopIteration),
  starving every TP=1 NVFP4 expert-bank load;
* the slice rules indexed pieces as 2-D ``[rows, cols]`` while ``per_expert_pieces``
  delivers ``[e1 - e0, rows, cols]`` -- at TP>1 ``t[start:end]`` then hit the size-1
  expert dim (no-op) and ``t[:, cols]`` ate the hidden axis, so a cold TP2 bank pack
  crashed copying full-width gate/up into a rank-local ``[2*i, h/2]`` bank.

Shapes mirror the kernel's NVFP4 row shapes: fp4 packs the K axis /2 for weights and
/GROUP(=16) for scales.
"""

from types import SimpleNamespace

import pytest
import torch

from freetoken.models.nvfp4_banks import _tp_slice_pieces


def _piece(intermediate: int = 64, hidden: int = 32) -> dict[str, torch.Tensor]:
    # per_expert_pieces rank: [e1 - e0, rows, cols]; K axis packed fp4-style
    return {
        "gate": torch.arange(intermediate * (hidden // 2)).reshape(1, intermediate, hidden // 2),
        "up": torch.arange(intermediate * (hidden // 2)).reshape(1, intermediate, hidden // 2),
        "gate_scale": torch.arange(intermediate * (hidden // 16)).reshape(1, intermediate, hidden // 16),
        "up_scale": torch.arange(intermediate * (hidden // 16)).reshape(1, intermediate, hidden // 16),
        "gate_global": torch.tensor([[0.5]]),
        "up_global": torch.tensor([[0.75]]),
        "down": torch.arange(hidden * (intermediate // 2)).reshape(1, hidden, intermediate // 2),
        "down_scale": torch.arange(hidden * (intermediate // 16)).reshape(1, hidden, intermediate // 16),
        "down_global": torch.tensor([[0.25]]),
    }


@pytest.mark.parametrize("rank, size", [(0, 1), (0, 2), (1, 2), (1, 4)])
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

    src = source[0][3]
    if size == 1:
        for name, tensor in src.items():
            assert torch.equal(piece[name], tensor), name
        return

    i_local = 64 // size
    rows = slice(rank * i_local, (rank + 1) * i_local)
    assert torch.equal(piece["gate"], src["gate"][:, rows])
    assert torch.equal(piece["up"], src["up"][:, rows])
    assert torch.equal(piece["gate_scale"], src["gate_scale"][:, rows])
    # down's K axis is last: fp4 weights packed /2, scales /GROUP
    packed = slice(rank * i_local // 2, (rank + 1) * i_local // 2)
    grouped = slice(rank * i_local // 16, (rank + 1) * i_local // 16)
    assert torch.equal(piece["down"], src["down"][..., packed])
    assert torch.equal(piece["down_scale"], src["down_scale"][..., grouped])
    # per-tensor globals pass through unchanged on every rank
    assert torch.equal(piece["gate_global"], src["gate_global"])
    assert torch.equal(piece["down_global"], src["down_global"])


def test_sliced_pieces_fit_the_tp_local_bank(monkeypatch):
    """The pack step must be able to copy the sliced pieces into the bank layout
    (``[2*i_local, hidden/2]`` etc.); full-width pieces from the old axis bug
    crashed right here."""
    import freetoken.distributed.info as info
    from freetoken.distributed import DistributedInfo
    from freetoken.layers.quantization.moe.base import fused_piece, fused_global, global_rows

    rank, size, hidden = 1, 2, 32
    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(rank, size))
    i_local = 32
    config = SimpleNamespace(moe_intermediate_size=64)
    (layer, e0, e1, piece), = _tp_slice_pieces(iter([("l", 0, 1, _piece())]), config)

    out = {
        "gate_up": torch.zeros(1, 2 * i_local, hidden // 2),
        "gate_up_global": torch.zeros(1, 2 * i_local),
        "down": torch.zeros(1, hidden, i_local // 2),
        "down_global": torch.zeros(1, hidden),
    }
    # fused_piece / fused_global expect the delivered [E, rows, cols] rank
    out["gate_up"].copy_(fused_piece(piece, "gate_up"))
    out["gate_up_global"].copy_(fused_global(piece, i_local))
    out["down"].copy_(piece["down"])
    out["down_global"].copy_(global_rows(piece["down_global"], hidden))

    src = _piece()
    # rank 1's slice: gate occupies the top half of the fused stack, up the bottom
    assert torch.equal(out["gate_up"][0, :i_local], src["gate"][0, i_local:])
    assert torch.equal(out["gate_up"][0, i_local:], src["up"][0, i_local:])
    assert torch.equal(out["down"][0], src["down"][0, :, i_local // 2:])
