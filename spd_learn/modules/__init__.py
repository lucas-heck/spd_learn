# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
from .batchnorm import BatchReNorm, SPDBatchNormMean, SPDBatchNormMeanVar
from .bilinear import BiMap, BiMapIncreaseDim
from .covariance import CovLayer
from .dropout import SPDDropout
from .liebn import SPDBatchNormLie
from .manifold import PositiveDefiniteScalar, SymmetricPositiveDefinite
from .modeig import ExpEig, LogEig, ReEig
from .regularize import Shrinkage, TraceNorm
from .residual import LogEuclideanResidual
from .spdtransnet import (  # noqa: F401 (sequence-native registry below)
    LearnedSinusoidalPosition,
    SPDInterEpochEncoder,
    SPDIntraEpochEncoder,
    SPDLogTokenizer,
    SPDTemporalEncoder,
    SPDTransformerEncoderLayer,
    StructurePreservingMultiheadAttention,
)
from .utils import PatchEmbeddingLayer, Vec, Vech
from .wavelet import WaveletConv


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

# Sequence modules operate on explicit token axes, not generic covariance inputs.
__sequence_modules__ = [
    "SPDLogTokenizer",
    "StructurePreservingMultiheadAttention",
    "SPDTransformerEncoderLayer",
    "LearnedSinusoidalPosition",
    "SPDTemporalEncoder",
    "SPDIntraEpochEncoder",
    "SPDInterEpochEncoder",
]
