"""Package-wide runtime hygiene: TP info set once, the global ctx never leaks across tests."""

import pytest


@pytest.fixture(autouse=True)
def _runtime():
    import freetoken.core as core
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    core._GLOBAL_CTX = None
    yield
    core._GLOBAL_CTX = None
    # get_rope is functools.cached with a device-blind key; a CPU-side model build
    # (e.g. checkpoint loader tests) must not leave its CPU rope for GPU tests to hit.
    from freetoken.layers.rotary import get_rope

    get_rope.cache_clear()
