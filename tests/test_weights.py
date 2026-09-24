import pytest
import torch

from tiletime.merge import axis_weights


@pytest.mark.parametrize("blend,curve,width", [
    ("feather", "smoothstep", 100), ("feather", "linear", 50), ("feather", "linear", 0), ("hard cut", "linear", 100),
])
def test_axis_weights_sum_to_one_for_neighbour_overlaps(blend, curve, width):
    spans = [(0, 40), (30, 70), (60, 100)]
    w = axis_weights(spans, 100, blend, curve, width)
    assert torch.allclose(w.sum(0), torch.ones(100))
    assert torch.all(w[0, 40:] == 0) and torch.all(w[2, :60] == 0)
    assert w[0, 0] == 1 and w[2, 99] == 1


def test_hard_cut_is_a_step_at_the_midpoint():
    w = axis_weights([(0, 40), (30, 70)], 70, "hard cut")
    assert torch.equal(w[0, :35], torch.ones(35)) and torch.equal(w[0, 35:], torch.zeros(35))


def test_axis_weights_positive_everywhere_with_wide_overlaps():
    spans = [(0, 1280), (610, 1890), (1220, 2500)]  # tiles 0 and 2 also overlap
    w = axis_weights(spans, 2500)
    assert torch.all(w.sum(0) > 0)
