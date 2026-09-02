"""Minimal pairwise-loss primitives used by the stochastic screen-SGD method."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, ndtr

Array = NDArray[np.float64]
Problem = Literal["rank", "auc"]


def project_l2_ball(x: Array, radius: float) -> Array:
    if radius <= 0:
        raise ValueError("radius must be positive")
    norm = float(np.linalg.norm(x))
    return x.copy() if norm <= radius else x * (radius / norm)


def project_l2_ball_metric(x: Array, radius: float, metric_inverse: Array) -> Array:
    """Project onto ``||u||_2 <= radius`` in the ``metric_inverse`` norm.

    ``metric_inverse`` is the SPD matrix multiplying the gradient in the
    preconditioned update.  The projection solves

        min_u (u-x)' metric_inverse^{-1} (u-x)  subject to ||u||_2 <= radius.

    If ``x`` lies outside the ball, the KKT solution is
    ``u=(I+lambda*metric_inverse)^{-1}x`` with the unique ``lambda>0``
    making ``||u||_2=radius``.  The eigendecomposition is inexpensive because
    this routine is used only after screening, in dimension at most K.
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
    x = np.asarray(x, dtype=float)
    if float(np.linalg.norm(x)) <= radius:
        return x.copy()
    P = (np.asarray(metric_inverse, dtype=float) +
         np.asarray(metric_inverse, dtype=float).T) / 2.0
    evals, Q = np.linalg.eigh(P)
    if not np.all(np.isfinite(evals)) or float(evals[0]) <= 0.0:
        raise np.linalg.LinAlgError("metric_inverse must be positive definite")
    z = Q.T @ x

    def norm_at(lam: float) -> float:
        return float(np.linalg.norm(z / (1.0 + lam * evals)))

    lo, hi = 0.0, 1.0
    while norm_at(hi) > radius:
        hi *= 2.0
        if hi > 1e16:
            raise FloatingPointError("metric projection failed to bracket multiplier")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if norm_at(mid) > radius:
            lo = mid
        else:
            hi = mid
    u = Q @ (z / (1.0 + hi * evals))
    # Remove the last few ulps of possible outward error from the scalar solve.
    nu = float(np.linalg.norm(u))
    if nu > radius:
        u *= radius / nu
    return u


def rank_rho(residual: Array, bandwidth: float) -> Array:
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    z = residual / bandwidth
    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    phi0 = 1.0 / np.sqrt(2.0 * np.pi)
    return residual * (ndtr(z) - 0.5) + bandwidth * (phi - phi0)


def scalar_loss_score_curvature(
    problem: Problem,
    w: Array,
    margin: Array,
    bandwidth: float = 1.0,
) -> Tuple[Array, Array, Array]:
    """Return loss, first derivative, and second derivative in the margin."""
    if problem == "rank":
        residual = w - margin
        z = residual / bandwidth
        loss = rank_rho(residual, bandwidth)
        score = -(ndtr(z) - 0.5)
        curvature = np.exp(-0.5 * z * z) / (
            np.sqrt(2.0 * np.pi) * bandwidth
        )
        return loss, score, curvature
    if problem == "auc":
        loss = np.logaddexp(0.0, -margin)
        score = -expit(-margin)
        prob = expit(margin)
        return loss, score, prob * (1.0 - prob)
    raise ValueError(f"unknown problem {problem}")


def endpoint_aggregated_gradient(
    X: Array,
    pair_i: NDArray[np.int64],
    pair_j: NDArray[np.int64],
    scalar_scores: Array,
) -> Array:
    """Compute mean_b score_b (X_i_b-X_j_b) without forming differences."""
    if pair_i.shape != pair_j.shape or pair_i.shape[0] != scalar_scores.shape[0]:
        raise ValueError("pair arrays and scores must align")
    if scalar_scores.size == 0:
        raise ValueError("at least one pair is required")
    endpoints = np.zeros(X.shape[0], dtype=float)
    np.add.at(endpoints, pair_i, scalar_scores)
    np.add.at(endpoints, pair_j, -scalar_scores)
    return X.T @ (endpoints / scalar_scores.size)


@dataclass(frozen=True)
class PairData:
    """Small explicit-pair evaluator used only by derivative/unit tests."""

    D: Array
    w: Array
    problem: Problem
    bandwidth: float = 1.0

    def __post_init__(self) -> None:
        if self.D.ndim != 2:
            raise ValueError("D must be a matrix")
        if self.w.ndim != 1 or self.w.shape[0] != self.D.shape[0]:
            raise ValueError("w must have one value per pair")
        if self.D.shape[0] == 0:
            raise ValueError("at least one pair is required")

    @property
    def n_pairs(self) -> int:
        return int(self.D.shape[0])

    def value_grad_hess(
        self, theta: Array, need_hess: bool = True
    ) -> Tuple[float, Array, Optional[Array]]:
        margin = self.D @ np.asarray(theta, dtype=float)
        loss, score, curvature = scalar_loss_score_curvature(
            self.problem, self.w, margin, self.bandwidth
        )
        value = float(np.mean(loss))
        grad = self.D.T @ score / self.n_pairs
        hess = None
        if need_hess:
            hess = (self.D.T * curvature) @ self.D / self.n_pairs
        return value, grad, hess
