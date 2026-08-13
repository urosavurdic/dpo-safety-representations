import torch

from src.interpretability.lora_subspace_check import subspace_capture_fraction


def test_direction_fully_inside_subspace_captures_all_energy():
    direction = torch.tensor([1.0, 0.0, 0.0])
    lora_B = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # column space = span(e0, e1)
    assert abs(subspace_capture_fraction(direction, lora_B) - 1.0) < 1e-5


def test_direction_orthogonal_to_subspace_captures_no_energy():
    direction = torch.tensor([0.0, 0.0, 1.0])  # orthogonal to span(e0, e1)
    lora_B = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    assert abs(subspace_capture_fraction(direction, lora_B) - 0.0) < 1e-5


def test_partial_overlap_gives_intermediate_fraction():
    direction = torch.tensor([1.0, 0.0, 1.0]) / (2 ** 0.5)  # half in-subspace, half out
    lora_B = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    assert abs(subspace_capture_fraction(direction, lora_B) - 0.5) < 1e-5