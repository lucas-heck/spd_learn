# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
from .batchnorm import BatchReNorm, SPDBatchNormMean, SPDBatchNormMeanVar
from .bilinear import BiMap, BiMapIncreaseDim
from .covariance import CovLayer
from .dropout import SPDDropout
from .gbw import GBWAttention, GBWGeometry, GBWPatchCovariance  # noqa: F401
from .liebn import SPDBatchNormLie
from .manifold import PositiveDefiniteScalar, SymmetricPositiveDefinite
from .modeig import ExpEig, LogEig, ReEig
from .regularize import Shrinkage, TraceNorm
from .residual import LogEuclideanResidual
from .utils import PatchEmbeddingLayer, Vec, Vech
from .wavelet import WaveletConv


# GBW blocks are explicit public imports above, outside the flat module
# registry: they have covariance-token/geometry interfaces, not its shared
# generic matrix-layer constructor and forward contract.
__all__ = [
    # covariance
    "CovLayer",
    # manifold parametrization
    "SymmetricPositiveDefinite",
    "PositiveDefiniteScalar",
    # modeig
    "LogEig",
    "ReEig",
    "ExpEig",
    # bilinear
    "BiMap",
    "BiMapIncreaseDim",
    # batchnorm
    "SPDBatchNormMean",
    "BatchReNorm",
    "SPDBatchNormMeanVar",
    "SPDBatchNormLie",
    # dropout
    "SPDDropout",
    # residual
    "LogEuclideanResidual",
    # utils
    "PatchEmbeddingLayer",
    "Vec",
    "Vech",
    # regularization
    "TraceNorm",
    "Shrinkage",
    # wavelet
    "WaveletConv",
]
