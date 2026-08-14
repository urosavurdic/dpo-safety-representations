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

def test_random_direction_baseline_near_expected_for_full_rank_subspace():
    # If lora_B spans the FULL space (identity-like), every random unit
    # vector should be entirely captured -> expected mean ~= 1.0
    from src.interpretability.lora_subspace_check import random_direction_capture_fraction
    full_rank_B = torch.eye(4)
    mean, std = random_direction_capture_fraction(full_rank_B, hidden_dim=4, n_samples=50, seed=0)
    assert abs(mean - 1.0) < 1e-5