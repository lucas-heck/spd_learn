# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for argument checks and code paths not reached by the other tests."""

import pytest
import torch

from torch.testing import assert_close

from spd_learn.functional import (
    exp_map_lem,
    log_euclidean_geodesic,
    log_map_lem,
    matrix_exp,
    matrix_log,
    spd_rpgd_attack,
)
from spd_learn.functional.metrics.affine_invariant import airm_distance
from spd_learn.functional.riemannian_pgd import _project_to_spd_ball
from spd_learn.models import SPDNet
from spd_learn.modules import PositiveDefiniteScalar, SymmetricPositiveDefinite


def _random_spd(*batch, n=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    A = torch.randn(*batch, n, n, dtype=torch.float64, generator=generator)
    return A @ A.mT + n * torch.eye(n, dtype=torch.float64)


# Manifold parametrizations
@pytest.mark.parametrize("cls", [SymmetricPositiveDefinite, PositiveDefiniteScalar])
def test_parametrization_rejects_unknown_mapping(cls):
    with pytest.raises(ValueError, match="mapping must be"):
        cls(mapping="relu")


@pytest.mark.parametrize("mapping", ["exp", "softplus"])
def test_spd_parametrization_round_trip(mapping):
    param = SymmetricPositiveDefinite(mapping=mapping)
    P = _random_spd(2)
    S = param.right_inverse(P)
    assert_close(S, S.mT)
    assert_close(param(S), P)
    assert torch.linalg.eigvalsh(param(-P)).min() > 0


@pytest.mark.parametrize("mapping", ["exp", "softplus"])
def test_positive_scalar_parametrization_round_trip(mapping):
    param = PositiveDefiniteScalar(mapping=mapping)
    s = torch.tensor(2.5, dtype=torch.float64)
    assert_close(param(param.right_inverse(s)), s)
    assert param(torch.tensor(-10.0, dtype=torch.float64)) > 0


# Riemannian PGD attack
class _WrappedModel:
    """Mimics a skorch/Braindecode estimator, which exposes ``module_``."""

    def __init__(self, module):
        self.module_ = module


def test_rpgd_attack_accepts_wrapped_model_and_stays_in_ball():
    torch.manual_seed(0)
    model = SPDNet(input_type="cov", n_chans=3, n_outputs=2)
    x = (torch.eye(3) + 0.1 * torch.ones(3, 3)).unsqueeze(0)
    y = torch.tensor([0])
    eps = 0.05
    x_adv = spd_rpgd_attack(
        _WrappedModel(model),
        x,
        y,
        eps=eps,
        criterion=torch.nn.CrossEntropyLoss(),
        n_iterations=5,
        step_size=1.0,
    )
    assert x_adv.shape == x.shape
    assert airm_distance(x, x_adv).max() <= eps + 1e-4


def test_rpgd_attack_rejects_non_module():
    with pytest.raises(TypeError, match="torch.nn.Module"):
        spd_rpgd_attack(
            object(),
            torch.eye(3)[None],
            torch.tensor([0]),
            eps=0.1,
            criterion=torch.nn.CrossEntropyLoss(),
        )


def test_project_to_spd_ball_single_matrix():
    x0 = _random_spd(seed=1)
    x = _random_spd(seed=2)
    dist = airm_distance(x0, x)
    assert dist.ndim == 0

    inside = _project_to_spd_ball(x0, x, eps=float(dist) + 1.0)
    assert_close(inside, x)

    eps = float(dist) / 2
    projected = _project_to_spd_ball(x0, x, eps=eps)
    assert_close(airm_distance(x0, projected), torch.tensor(eps, dtype=x.dtype))


def test_project_to_spd_ball_batch_projects_only_outside_points():
    x0 = _random_spd(2, seed=3)
    x = _random_spd(2, seed=4)
    dist = airm_distance(x0, x)
    eps = float(dist.min() + dist.max()) / 2  # one point inside, one outside
    projected = _project_to_spd_ball(x0, x, eps=eps)
    inside = dist <= eps
    assert_close(projected[inside], x[inside])
    assert_close(
        airm_distance(x0[~inside], projected[~inside]),
        torch.full((int((~inside).sum()),), eps, dtype=x.dtype),
    )


# Log-Euclidean metric
def test_log_euclidean_exp_and_log_maps_are_inverse():
    P = _random_spd(2, seed=5)
    Q = _random_spd(2, seed=6)
    V = log_map_lem(P, Q)
    assert_close(V, matrix_log.apply(Q) - matrix_log.apply(P))
    assert_close(exp_map_lem(P, V), Q)


def test_log_euclidean_geodesic_scalar_tensor_time_on_batch():
    A = _random_spd(4, seed=7)
    B = _random_spd(4, seed=8)
    t = torch.tensor(0.25, dtype=torch.float64)
    G = log_euclidean_geodesic(A, B, t)
    expected = matrix_exp.apply(0.75 * matrix_log.apply(A) + 0.25 * matrix_log.apply(B))
    assert_close(G, expected)
