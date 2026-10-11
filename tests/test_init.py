# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the initialization functions in :mod:`spd_learn.init`."""

import logging

import pytest
import torch

import spd_learn.init as spd_init


def _assert_orthonormal_columns(W):
    eye = torch.eye(W.shape[-1], dtype=W.dtype).expand(*W.shape[:-2], -1, -1)
    torch.testing.assert_close(W.mT @ W, eye, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("shape", [(6, 3), (2, 6, 3)])
def test_stiefel_is_orthonormal_and_reproducible(shape):
    W1 = spd_init.stiefel_(torch.empty(shape), seed=7)
    W2 = spd_init.stiefel_(torch.empty(shape), seed=7)
    _assert_orthonormal_columns(W1)
    torch.testing.assert_close(W1, W2)


def test_stiefel_without_seed_warns_and_uses_seed_zero(caplog):
    with caplog.at_level(logging.WARNING, logger=spd_init.logger.name):
        W = spd_init.stiefel_(torch.empty(5, 2))
    assert "No seed provided" in caplog.text
    torch.testing.assert_close(W, spd_init.stiefel_(torch.empty(5, 2), seed=0))


@pytest.mark.parametrize("shape", [(6, 3), (2, 6, 3)])
def test_stiefel_falls_back_to_qr(monkeypatch, caplog, shape):
    def failing_polar_factor(_):
        raise torch.linalg.LinAlgError("forced failure")

    monkeypatch.setattr(spd_init, "orthogonal_polar_factor", failing_polar_factor)
    with caplog.at_level(logging.WARNING, logger=spd_init.logger.name):
        W = spd_init.stiefel_(torch.empty(shape), seed=1)
    assert "Falling back to QR" in caplog.text
    assert W.shape == shape
    _assert_orthonormal_columns(W)


@pytest.mark.parametrize("shape", [(4, 4), (3, 2, 5, 5)])
def test_spd_identity(shape):
    tensor = torch.randn(shape)
    out = spd_init.spd_identity_(tensor)
    assert out is tensor
    torch.testing.assert_close(out, torch.eye(shape[-1]).expand(shape))


def test_spd_identity_rejects_non_square():
    with pytest.raises(ValueError, match="square"):
        spd_init.spd_identity_(torch.empty(3, 4))
