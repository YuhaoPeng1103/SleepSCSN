"""Forward-pass shape checks for the two backbones.

The EEGNet path is a ``Conv2d`` stack fed from 1-D epoch tensors, so a missing
reshape is silently reinterpreted as an unbatched ``(C, H, W)`` input rather
than raising. These tests pin the batch dimension.
"""

from __future__ import annotations

import torch

import pytest

from sleepcp.models import build_model, count_parameters


@pytest.mark.parametrize("name", ["eegnet", "sleepcnn"])
@pytest.mark.parametrize("shape", [(4, 3000), (4, 1, 3000)])
def test_forward_keeps_batch_dimension(name: str, shape: tuple[int, ...]) -> None:
    model = build_model(name)
    out = model(torch.randn(*shape))
    assert out.shape == (4, 5)


def test_backbones_differ_in_size() -> None:
    small = count_parameters(build_model("eegnet"))
    large = count_parameters(build_model("sleepcnn"))
    assert small < large
