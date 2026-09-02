"""Full-data balanced-matching HTP and pair-SGD inference.

Discovery uses a fixed incomplete-pair objective formed from logarithmically
many maximal matchings.  The score at zero is the first HTP proposal; later
full-p residual gradients allow re-entry after joint restricted refits.  The
resulting fixed-dimensional candidate is cleaned and finally refit by warm-point
preconditioned pair-SGD, with a sampled H\'ajek sandwich.  No routine
enumerates the complete pair population.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray

from .pairwise_method import (scalar_loss_score_curvature, project_l2_ball,
                              project_l2_ball_metric)

Array = NDArray[np.float64]
Problem = Literal["rank", "auc"]
PAIR_ID_EXACT_TRACKING_LIMIT = 25 * 100_000


def _child_stream_audit(
    parent_role: str,
    child_names: tuple[str, ...],
    seeds: NDArray[np.int64],
    *,
    consumed: Optional[tuple[bool, ...]] = None,
) -> dict:
    """Return a JSON-safe registry for child generators minted by one parent."""
    if len(child_names) != int(np.asarray(seeds).size):
        raise ValueError("child stream names and seeds must have equal length")
    if consumed is None:
        consumed = tuple(True for _ in child_names)
    if len(consumed) != len(child_names):
        raise ValueError("child stream consumption flags must match child names")
    records = [
        {
            "stream_name": f"{parent_role}.{name}",
            "seed": int(seed),
            "consumed": bool(was_consumed),
        }
        for name, seed, was_consumed in zip(child_names, np.asarray(seeds), consumed)
    ]
    seed_values = [record["seed"] for record in records]
    names = [record["stream_name"] for record in records]
    return {
        "stream_role": str(parent_role),
        "spawned_child_streams": records,
        "spawned_child_stream_count": len(records),
        "spawned_child_stream_names_unique": len(set(names)) == len(names),
        "spawned_child_seeds_unique": len(set(seed_values)) == len(seed_values),
    }


class _PairIdentityTracker:
    """Count every pair evaluation and retain IDs only under a fixed memory cap."""

    def __init__(
        self, n: int, expected: int, scope: str,
        *, mirror: Optional["_PairIdentityTracker"] = None,
    ) -> None:
        self.n = int(n)
        self.expected = int(expected)
        self.scope = str(scope)
        self.used = 0
        self.mirror = mirror
        self._reason: Optional[str] = None
        if self.expected <= PAIR_ID_EXACT_TRACKING_LIMIT:
            self._ids: Optional[NDArray[np.int64]] = np.empty(self.expected, dtype=np.int64)
        else:
            self._ids = None
            self._reason = (
                f"expected evaluations {self.expected} exceed exact-ID cap "
                f"{PAIR_ID_EXACT_TRACKING_LIMIT}"
            )

    def record(self, ii: NDArray[np.int64], jj: NDArray[np.int64]) -> None:
        ii = np.asarray(ii, dtype=np.int64)
        jj = np.asarray(jj, dtype=np.int64)
        if ii.shape != jj.shape:
            raise ValueError("pair identity arrays must be aligned")
        if self.mirror is not None:
            self.mirror.record(ii, jj)
        count = int(ii.size)
        if self._ids is not None:
            stop = self.used + count
            if stop > self._ids.size:
                self._ids = None
                self._reason = "observed evaluations exceeded the declared exact-ID capacity"
            else:
                lo = np.minimum(ii, jj)
                hi = np.maximum(ii, jj)
                self._ids[self.used:stop] = lo * self.n + hi
        self.used += count

    def summary(self) -> dict:
        distinct: Optional[int] = None
        digest: Optional[str] = None
        exact = bool(self._ids is not None and self.used == self.expected)
        reason = self._reason
        if exact:
            unique_ids = np.unique(self._ids[:self.used])
            distinct = int(unique_ids.size)
            canonical = np.asarray(unique_ids, dtype="<i8")
            digest = hashlib.sha256(canonical.tobytes()).hexdigest()
        elif reason is None:
            reason = f"observed {self.used} evaluations but expected {self.expected}"
        return {
            "pair_identity_scope": self.scope,
            "pair_identity_definition": "unordered min(i,j)*N+max(i,j) on base-unit indices",
            "pair_evaluations_total": int(self.used),
            "pair_evaluations_expected": int(self.expected),
            "pair_evaluations_match_expected": bool(self.used == self.expected),
            # Full ID lists are intentionally omitted; exact count plus the hash
            # commits to the sorted set without inflating every experiment row.
            "pair_ids_distinct_exact": None,
            "pair_ids_distinct_count_is_exact": exact,
            "pair_ids_distinct_count_exact": distinct,
            "pair_ids_distinct_sha256": digest,
            "distinct_unavailable_reason": None if exact else reason,
            "pair_id_exact_tracking_limit": int(PAIR_ID_EXACT_TRACKING_LIMIT),
            "complete": bool(self.used == self.expected and exact),
        }


def _pair_accounting(
    identity: dict,
    *,
    objective_calls: int,
    gradient_calls: int,
    hessian_calls: int,
    stagewise: dict,
) -> dict:
    return {
        **identity,
        "objective_call_count": int(objective_calls),
        "gradient_call_count": int(gradient_calls),
        "hessian_call_count": int(hessian_calls),
        "stagewise": stagewise,
    }


def _new_final_pair_union_tracker(n: int, d: int, scope: str) -> _PairIdentityTracker:
    """Allocate exact-union tracking for the default final optimizer+sandwich path."""
    steps = optimizer_steps(int(n))
    burn_steps = warm_burnin_steps(int(n))
    batch = optimizer_batch_size(int(n), int(d))
    optimizer_pairs = burn_steps * batch + pilot_pairs(int(n)) + steps * batch
    hessian_pairs = int(math.ceil(int(n) * math.log(max(int(n), 3))))
    partners = int(math.ceil(math.log(max(int(n), 3)) ** 2))
    sandwich_pairs = hessian_pairs + int(n) * partners
    return _PairIdentityTracker(
        int(n), int(optimizer_pairs + sandwich_pairs), str(scope)
    )


def _final_pair_accounting(
    tracker: _PairIdentityTracker, optimizer: dict, sandwich: dict
) -> dict:
    optimizer_accounting = optimizer.get("pair_accounting", {})
    sandwich_accounting = sandwich.get("pair_accounting", {})
    return _pair_accounting(
        tracker.summary(),
        objective_calls=int(optimizer_accounting.get("objective_call_count", 0))
                        + int(sandwich_accounting.get("objective_call_count", 0)),
        gradient_calls=int(optimizer_accounting.get("gradient_call_count", 0))
                       + int(sandwich_accounting.get("gradient_call_count", 0)),
        hessian_calls=int(optimizer_accounting.get("hessian_call_count", 0))
                      + int(sandwich_accounting.get("hessian_call_count", 0)),
        stagewise={
            "optimizer": optimizer_accounting,
            "sandwich": sandwich_accounting,
        },
    )


def _empty_final_pair_accounting(scope: str) -> dict:
    identity = {
        "pair_identity_scope": str(scope),
        "pair_identity_definition": "unordered min(i,j)*N+max(i,j) on base-unit indices",
        "pair_evaluations_total": 0,
        "pair_evaluations_expected": 0,
        "pair_evaluations_match_expected": True,
        "pair_ids_distinct_exact": [],
        "pair_ids_distinct_count_is_exact": True,
        "pair_ids_distinct_count_exact": 0,
        "pair_ids_distinct_sha256": hashlib.sha256(b"").hexdigest(),
        "distinct_unavailable_reason": None,
        "pair_id_exact_tracking_limit": int(PAIR_ID_EXACT_TRACKING_LIMIT),
        "complete": True,
    }
    return _pair_accounting(
        identity, objective_calls=0, gradient_calls=0, hessian_calls=0,
        stagewise={"optimizer": None, "sandwich": None},
    )


def stratified_half_split(
    y: Array, problem: Problem, rng: np.random.Generator
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Split independent base units; in AUC split each class separately."""
    n = y.shape[0]
    if problem == "rank":
        perm = rng.permutation(n)
        cut = n // 2
        return np.sort(perm[:cut]), np.sort(perm[cut:])
    by_class: dict[int, NDArray[np.int64]] = {}
    for cls in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == cls))
        if idx.size < 2:
            raise ValueError("each class needs at least two observations")
        by_class[cls] = np.asarray(idx, dtype=np.int64)

    # Pin the first arm to floor(N/2).  Splitting both classes at their own
    # floors would be wrong when both class counts are odd: the two arms would
    # then differ by two base units.  Giving the second class the complementary
    # cut assigns the two odd remainders to opposite arms while preserving a
    # within-class half split and at least one observation of each class per arm.
    first_target = n // 2
    cut0 = int(by_class[0].size // 2)
    cut1 = int(first_target - cut0)
    cuts = {0: cut0, 1: cut1}
    first: list[int] = []
    second: list[int] = []
    for cls in (0, 1):
        idx = by_class[cls]
        cut = cuts[cls]
        if not 1 <= cut < idx.size:
            raise RuntimeError("balanced stratified split produced an empty class arm")
        first.extend(idx[:cut].tolist())
        second.extend(idx[cut:].tolist())
    if abs(len(first) - len(second)) > 1 or len(first) + len(second) != n:
        raise RuntimeError("balanced stratified split violated the half-sample size rule")
    return np.asarray(sorted(first), dtype=np.int64), np.asarray(sorted(second), dtype=np.int64)



def complete_score_at_zero(
    X: Array,
    y: Array,
    problem: Problem,
    bandwidth: float = 1.0,
    block_rows: int = 256,
) -> Array:
    """Exact complete-U score at ``theta=0`` for the difference-form examples.

    The routine is a diagnostic/reference implementation used by Gate 1.6.  It
    never creates a pair-by-feature matrix.  Rank endpoint weights are formed in
    row blocks; the AUC score reduces to the difference of class means.
    """
    n, p = X.shape
    if n < 2:
        raise ValueError("at least two base units are required")
    if problem == "auc":
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        if pos.size == 0 or neg.size == 0:
            raise ValueError("both AUC classes are required")
        return -0.5 * (X[pos].mean(axis=0) - X[neg].mean(axis=0))
    if problem != "rank":
        raise ValueError(f"unknown problem {problem}")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    pairs = n * (n - 1) / 2.0
    endpoints = np.empty(n, dtype=float)
    for start in range(0, n, block_rows):
        stop = min(n, start + block_rows)
        rows = np.arange(start, stop)
        residual = y[rows, None] - y[None, :]
        _, scalar, _ = scalar_loss_score_curvature(
            "rank", residual, np.zeros_like(residual), bandwidth
        )
        scalar[np.arange(stop - start), rows] = 0.0
        endpoints[start:stop] = scalar.sum(axis=1) / pairs
    return X.T @ endpoints

def incomplete_score_screen(
    X: Array,
    y: Array,
    problem: Problem,
    K: int,
    rng: np.random.Generator,
    bandwidth: float = 1.0,
    n_pairs: Optional[int] = None,
) -> tuple[NDArray[np.int64], Array, dict]:
    """Loss-aligned randomized incomplete-U top-K score screen.

    For the difference-form rank/AUC examples the sampled pair contributions
    are accumulated at their endpoints, so the p-dimensional operation is one
    matrix-vector product rather than a pair-by-p materialization.
    """
    n, p = X.shape
    if not 1 <= K < p:
        raise ValueError("K must lie in [1,p-1]")
    if n_pairs is None:
        n_pairs = int(math.ceil(n * math.log(max(2 * p * n, 3))))
    pair_tracker = _PairIdentityTracker(n, n_pairs, "incomplete_score_screen")
    ii, jj = _sample_pairs(y, problem, n_pairs, rng, pair_tracker=pair_tracker)
    if problem == "rank":
        w = y[ii] - y[jj]
        _, scalar, _ = scalar_loss_score_curvature(
            problem, w, np.zeros(n_pairs), bandwidth
        )
    else:
        scalar = np.full(n_pairs, -0.5, dtype=float)
    endpoints = np.zeros(n, dtype=float)
    np.add.at(endpoints, ii, scalar)
    np.add.at(endpoints, jj, -scalar)
    score = X.T @ (endpoints / n_pairs)
    order = np.lexsort((np.arange(p), -np.abs(score)))
    candidate = np.sort(order[:K]).astype(np.int64)
    pair_identity = pair_tracker.summary()
    return candidate, score, {
        "screen_pair_draws": int(n_pairs),
        "candidate_size": int(K),
        "score_inf": float(np.max(np.abs(score))),
        "complete_pair_enumeration": False,
        "endpoint_aggregation": True,
        "pair_instrumentation": pair_identity,
        "pair_accounting": _pair_accounting(
            pair_identity, objective_calls=0, gradient_calls=1,
            hessian_calls=0,
            stagewise={"screen_gradient": {"pair_evaluations_total": int(n_pairs),
                                             "gradient_call_count": 1}},
        ),
        "objective_call_count": 0,
        "gradient_call_count": 1,
        "hessian_call_count": 0,
    }


def _sample_pairs(
    y: Array, problem: Problem, batch_size: int, rng: np.random.Generator,
    *, pair_tracker: Optional[_PairIdentityTracker] = None,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    n = y.shape[0]
    if problem == "rank":
        ii = rng.integers(0, n, size=batch_size, dtype=np.int64)
        raw = rng.integers(0, n - 1, size=batch_size, dtype=np.int64)
        jj = raw + (raw >= ii)
        if pair_tracker is not None:
            pair_tracker.record(ii, jj)
        return ii, jj
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("both AUC classes are required")
    ii = pos[rng.integers(0, pos.size, size=batch_size)]
    jj = neg[rng.integers(0, neg.size, size=batch_size)]
    if pair_tracker is not None:
        pair_tracker.record(ii, jj)
    return ii, jj


def pair_minibatch_gradient(
    Xs: Array,
    y: Array,
    theta: Array,
    problem: Problem,
    batch_size: int,
    rng: np.random.Generator,
    bandwidth: float = 1.0,
    *,
    pair_tracker: Optional[_PairIdentityTracker] = None,
) -> Array:
    ii, jj = _sample_pairs(y, problem, batch_size, rng, pair_tracker=pair_tracker)
    d = Xs[ii] - Xs[jj]
    if problem == "rank":
        w = y[ii] - y[jj]
    else:
        w = np.zeros(batch_size)
    margin = d @ theta
    _, scalar, _ = scalar_loss_score_curvature(problem, w, margin, bandwidth)
    return d.T @ scalar / batch_size


@dataclass
class SGDFit:
    theta: Array
    diagnostics: dict


def tail_averaged_pair_sgd(
    Xs: Array,
    y: Array,
    problem: Problem,
    rng: np.random.Generator,
    *,
    bandwidth: float = 1.0,
    radius: float = 20.0,
    n_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    eta0: float = 1.0,
    exponent: float = 2.0 / 3.0,
    x0: Optional[Array] = None,
    pair_tracker: Optional[_PairIdentityTracker] = None,
) -> SGDFit:
    """Projected pair-SGD with a deterministic t^{-2/3} gain and tail averaging."""
    n, d = Xs.shape
    if d == 0:
        return SGDFit(np.zeros(0), {"steps": 0, "boundary_rate": 0.0, "tail_boundary_hits": 0,
                                      "tail_boundary_rate": 0.0, "boundary_margin": float(radius),
                                      "finite": True, "selection_empty": True,
                                      "pair_evaluations_total": 0,
                                      "pair_accounting": {
                                          "pair_identity_scope": "tail_averaged_pair_sgd",
                                          "pair_evaluations_total": 0,
                                          "pair_ids_distinct_exact": [],
                                          "pair_ids_distinct_count_exact": 0,
                                          "pair_ids_distinct_sha256": hashlib.sha256(b"").hexdigest(),
                                          "distinct_unavailable_reason": None,
                                          "objective_call_count": 0,
                                          "gradient_call_count": 0,
                                          "hessian_call_count": 0,
                                          "stagewise": {},"complete": True,
                                      }})
    if n_steps is None:
        n_steps = int(math.ceil(n * math.log(max(n, 3))))
    if batch_size is None:
        batch_size = int(math.ceil(d + math.log(max(n * n_steps, 3))))
    if n_steps < 4:
        raise ValueError("n_steps must be at least four")
    if not (0.5 < exponent < 1.0):
        raise ValueError("exponent must lie in (1/2,1)")
    own_pair_tracker = pair_tracker is None
    if pair_tracker is None:
        pair_tracker = _PairIdentityTracker(
            n, int(n_steps * batch_size), "tail_averaged_pair_sgd"
        )
    theta = np.zeros(d) if x0 is None else project_l2_ball(np.asarray(x0, dtype=float), radius)
    start_avg = n_steps // 2
    avg = np.zeros(d)
    avg_count = 0
    boundary_hits = 0
    tail_boundary_hits = 0
    grad_sq = 0.0
    for t in range(1, n_steps + 1):
        grad = pair_minibatch_gradient(
            Xs, y, theta, problem, batch_size, rng, bandwidth,
            pair_tracker=pair_tracker,
        )
        grad_sq += float(grad @ grad)
        eta = eta0 / (t ** exponent)
        raw = theta - eta * grad
        theta = project_l2_ball(raw, radius)
        hit = bool(np.linalg.norm(raw) > radius)
        if hit:
            boundary_hits += 1
        if t > start_avg:
            if hit:
                tail_boundary_hits += 1
            avg += theta
            avg_count += 1
    theta_bar = avg / avg_count
    pair_identity = (
        pair_tracker.summary() if own_pair_tracker else {
            "pair_identity_scope": "shared_parent_tracker",
            "pair_evaluations_total": int(n_steps * batch_size),
            "pair_ids_distinct_exact": None,
            "pair_ids_distinct_count_exact": None,
            "distinct_unavailable_reason": "identity is summarized by the parent optimizer",
            "complete": True,
        }
    )
    return SGDFit(
        theta_bar,
        {
            "steps": int(n_steps),
            "batch_size": int(batch_size),
            "pair_draws": int(n_steps * batch_size),
            "pair_evaluations_total": int(n_steps * batch_size),
            "pair_instrumentation": pair_identity,
            "pair_accounting": _pair_accounting(
                pair_identity, objective_calls=0, gradient_calls=int(n_steps),
                hessian_calls=0,
                stagewise={"gradient": {
                    "pair_evaluations_total": int(n_steps * batch_size),
                    "gradient_call_count": int(n_steps),
                }},
            ),
            "objective_call_count": 0,
            "gradient_call_count": int(n_steps),
            "hessian_call_count": 0,
            "eta0": float(eta0),
            "exponent": float(exponent),
            "tail_average_count": int(avg_count),
            "boundary_rate": float(boundary_hits / n_steps),
            "tail_boundary_hits": int(tail_boundary_hits),
            "tail_boundary_rate": float(tail_boundary_hits / max(avg_count, 1)),
            "mean_gradient_sq": float(grad_sq / n_steps),
            "norm": float(np.linalg.norm(theta_bar)),
            "boundary_margin": float(radius - np.linalg.norm(theta_bar)),
            "finite": bool(np.all(np.isfinite(theta_bar))),
            "radius": float(radius),
        },
    )


def _sampled_hessian(
    Xs: Array,
    y: Array,
    theta: Array,
    problem: Problem,
    rng: np.random.Generator,
    bandwidth: float,
    n_pairs: int,
    chunk_size: int = 8192,
    pair_tracker: Optional[_PairIdentityTracker] = None,
) -> Array:
    d = Xs.shape[1]
    H = np.zeros((d, d))
    done = 0
    while done < n_pairs:
        b = min(chunk_size, n_pairs - done)
        ii, jj = _sample_pairs(
            y, problem, b, rng, pair_tracker=pair_tracker
        )
        D = Xs[ii] - Xs[jj]
        w = y[ii] - y[jj] if problem == "rank" else np.zeros(b)
        _, _, curvature = scalar_loss_score_curvature(
            problem, w, D @ theta, bandwidth
        )
        H += (D.T * curvature) @ D
        done += b
    return H / n_pairs



# ---------------------------------------------------------------------------
# Theorem-scheduled warm-point preconditioned pair-SGD
# ---------------------------------------------------------------------------
# The low-dimensional optimizer has no empirically tuned batch or ridge
# constant.  For a d-dimensional restricted problem on a fold of size n,
#
#   T_w(n) = n,
#   B_P(n) = ceil(n log(n vee 3)),
#   b(n,d) = ceil{d + log[n T(n)]},
#   r_n = n^{-1/2},
#   T(n) = ceil(n log(n vee 3)).
#
# The dimension-plus-log-horizon batch is the scale in a uniform concentration
# bound for a d-vector minibatch gradient over T(n) iterations.  The relative
# spectral floor r_n vanishes, stabilizing finite-sample inversion while the
# preconditioner converges to the inverse empirical Hessian.  No ordering
# between r_n and the pilot Monte Carlo scale is claimed or consumed: the
# perturbation argument uses only ||A_w - A||_op = o_P(1) and rho = O(r_n).
# Neither sequence is selected from recovery, coverage, or development results.

METHOD_VERSION = "v26.1f-warm-pilot-quadratic-growth-repair"


def optimizer_steps(n: int) -> int:
    return int(math.ceil(n * math.log(max(n, 3))))


def warm_burnin_steps(n: int) -> int:
    return max(4, int(n))


def pilot_pairs(n: int) -> int:
    return optimizer_steps(n)


def optimizer_batch_size(n: int, d: int) -> int:
    """Deterministic dimension-plus-log-horizon minibatch schedule."""
    if n < 2 or d < 1:
        raise ValueError("n>=2 and d>=1 are required")
    return int(math.ceil(d + math.log(max(n * optimizer_steps(n), 3))))


def relative_spectral_floor(n: int) -> float:
    """Vanishing relative spectral floor."""
    if n < 1:
        raise ValueError("n must be positive")
    return float(n ** -0.5)


def frozen_preconditioner(A: Array, n: int) -> tuple[Array, dict]:
    """Return ``(A+r_n lambda_max(A) I)^{-1}``, ``r_n=n^{-1/2}``."""
    A = (np.asarray(A, dtype=float) + np.asarray(A, dtype=float).T) / 2.0
    ev = np.linalg.eigvalsh(A)
    lam_min = float(ev[0])
    lam_max = float(ev[-1])
    if not np.isfinite(lam_max) or lam_max <= 0.0:
        raise np.linalg.LinAlgError("pilot Hessian has no positive finite eigenvalue")
    r_n = relative_spectral_floor(n)
    rho = r_n * lam_max
    chol = np.linalg.cholesky(A + rho * np.eye(A.shape[0]))
    eye = np.eye(A.shape[0])
    P = np.linalg.solve(chol.T, np.linalg.solve(chol, eye))
    kappa = float(lam_max / lam_min) if lam_min > 0 else float("inf")
    retained = float(lam_min / (lam_min + rho)) if lam_min > 0 else 0.0
    return P, {
        "pilot_lambda_min": lam_min,
        "pilot_lambda_max": lam_max,
        "pilot_condition_number": kappa,
        "relative_spectral_floor": r_n,
        "spectral_floor_absolute": float(rho),
        "preconditioned_condition_number": float((lam_max + rho) / (lam_min + rho))
        if lam_min + rho > 0 else float("inf"),
        "weakest_direction_drift_retained": retained,
        "spectral_condition_number_cap": float(1.0 + math.sqrt(n)),
    }


def warm_preconditioned_pair_sgd(
    Xs: Array,
    y: Array,
    problem: Problem,
    rng: np.random.Generator,
    *,
    bandwidth: float = 1.0,
    radius: float = 20.0,
    n_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    eta0: float = 1.0,
    exponent: float = 2.0 / 3.0,
    x0: Optional[Array] = None,
    stream_role: str = "pair_sgd",
    pair_union_tracker: Optional[_PairIdentityTracker] = None,
) -> SGDFit:
    """Burn in, estimate one warm-point Hessian, freeze it, and tail-average.

    The incoming generator is used only to mint three named child seeds.
    Burn-in, pilot, and main-phase pairs are therefore generated by distinct
    child generators, matching the stagewise filtration used in the proof.
    Main updates use projection in the ``P^{-1}`` metric; after the
    transformation ``z=P^{-1/2} theta`` this is an ordinary Euclidean projected
    stochastic approximation with symmetric positive-definite drift.
    """
    n, d = Xs.shape
    stage_seeds = rng.integers(0, np.iinfo(np.int64).max, size=3, dtype=np.int64)
    burn_rng = np.random.default_rng(int(stage_seeds[0]))
    pilot_rng = np.random.default_rng(int(stage_seeds[1]))
    main_rng = np.random.default_rng(int(stage_seeds[2]))
    stream_audit = _child_stream_audit(
        stream_role, ("burnin", "pilot_hessian", "main"), stage_seeds
    )
    if d == 0:
        return SGDFit(np.zeros(0), {"steps": 0, "boundary_rate": 0.0, "tail_boundary_hits": 0,
                                      "tail_boundary_rate": 0.0, "boundary_margin": float(radius),
                                      "finite": True, "selection_empty": True,
                                      "pair_evaluations_total": 0,
                                      "pair_instrumentation": {
                                          "pair_identity_scope": stream_role,
                                          "pair_evaluations_total": 0,
                                          "pair_ids_distinct_exact": [],
                                          "pair_ids_distinct_count_exact": 0,
                                          "pair_ids_distinct_sha256": hashlib.sha256(b"").hexdigest(),
                                          "distinct_unavailable_reason": None,
                                          "complete": True,
                                      },
                                      "pair_accounting": {
                                          "pair_identity_scope": stream_role,
                                          "pair_evaluations_total": 0,
                                          "pair_ids_distinct_exact": [],
                                          "pair_ids_distinct_count_exact": 0,
                                          "pair_ids_distinct_sha256": hashlib.sha256(b"").hexdigest(),
                                          "distinct_unavailable_reason": None,
                                          "objective_call_count": 0,
                                          "gradient_call_count": 0,
                                          "hessian_call_count": 0,
                                          "stagewise": {},
                                          "complete": True,
                                      },
                                      "objective_call_count": 0,
                                      "gradient_call_count": 0,
                                      "hessian_call_count": 0,
                                      **stream_audit})
    if n_steps is None:
        n_steps = optimizer_steps(n)
    if batch_size is None:
        batch_size = optimizer_batch_size(n, d)
    if n_steps < 4:
        raise ValueError("n_steps must be at least four")
    if not (0.5 < exponent < 1.0):
        raise ValueError("exponent must lie in (1/2,1)")
    burn_steps = warm_burnin_steps(n)
    n_pilot = pilot_pairs(n)
    pair_tracker = _PairIdentityTracker(
        n,
        int(burn_steps * batch_size + n_pilot + n_steps * batch_size),
        stream_role,
        mirror=pair_union_tracker,
    )

    burn = tail_averaged_pair_sgd(
        Xs, y, problem, burn_rng, bandwidth=bandwidth, radius=radius,
        n_steps=burn_steps, batch_size=batch_size, eta0=eta0,
        exponent=exponent, x0=x0, pair_tracker=pair_tracker,
    )
    theta_w = np.asarray(burn.theta, dtype=float)

    A_w = _sampled_hessian(
        Xs, y, theta_w, problem, pilot_rng, bandwidth, n_pilot,
        pair_tracker=pair_tracker,
    )
    P, pilot_info = frozen_preconditioner(A_w, n)
    pilot_info["pilot_pairs"] = int(n_pilot)
    pilot_info["burnin_steps"] = int(burn_steps)

    theta = project_l2_ball(theta_w, radius)
    start_avg = n_steps // 2
    avg = np.zeros(d)
    avg_count = 0
    boundary_hits = 0
    tail_boundary_hits = 0
    grad_sq = 0.0
    for t in range(1, n_steps + 1):
        grad = pair_minibatch_gradient(
            Xs, y, theta, problem, batch_size, main_rng, bandwidth,
            pair_tracker=pair_tracker,
        )
        grad_sq += float(grad @ grad)
        raw = theta - (eta0 / (t ** exponent)) * (P @ grad)
        hit = bool(np.linalg.norm(raw) > radius)
        if hit:
            boundary_hits += 1
        theta = project_l2_ball_metric(raw, radius, P)
        if t > start_avg:
            if hit:
                tail_boundary_hits += 1
            avg += theta
            avg_count += 1
    theta_bar = avg / max(avg_count, 1)
    pair_identity = pair_tracker.summary()
    diagnostics = {
        "optimizer": "rate_calibrated_warm_point_preconditioner",
        "stage_streams": ["burnin", "pilot_hessian", "main"],
        "stage_streams_independently_seeded": True,
        **stream_audit,
        "steps": int(n_steps),
        "batch_size": int(batch_size),
        "batch_rule": "ceil(d + log(n*T(n)))",
        "pair_draws": int(n_steps * batch_size + burn_steps * batch_size + n_pilot),
        "pair_evaluations_total": int(n_steps * batch_size + burn_steps * batch_size + n_pilot),
        "stage_pair_evaluations": {
            "burnin_gradient": int(burn_steps * batch_size),
            "pilot_hessian": int(n_pilot),
            "main_gradient": int(n_steps * batch_size),
        },
        "pair_instrumentation": pair_identity,
        "pair_accounting": _pair_accounting(
            pair_identity,
            objective_calls=0,
            gradient_calls=int(burn_steps + n_steps),
            hessian_calls=int(math.ceil(n_pilot / 8192)),
            stagewise={
                "burnin_gradient": {
                    "pair_evaluations_total": int(burn_steps * batch_size),
                    "gradient_call_count": int(burn_steps),
                },
                "pilot_hessian": {
                    "pair_evaluations_total": int(n_pilot),
                    "hessian_call_count": int(math.ceil(n_pilot / 8192)),
                },
                "main_gradient": {
                    "pair_evaluations_total": int(n_steps * batch_size),
                    "gradient_call_count": int(n_steps),
                },
            },
        ),
        "objective_call_count": 0,
        "gradient_call_count": int(burn_steps + n_steps),
        "hessian_call_count": int(math.ceil(n_pilot / 8192)),
        "eta0": float(eta0),
        "exponent": float(exponent),
        "tail_average_count": int(avg_count),
        "boundary_rate": float(boundary_hits / n_steps),
        "tail_boundary_hits": int(tail_boundary_hits),
        "tail_boundary_rate": float(tail_boundary_hits / max(avg_count, 1)),
        "projection_metric": "P^{-1}",
        "mean_gradient_sq": float(grad_sq / n_steps),
        "norm": float(np.linalg.norm(theta_bar)),
        "boundary_margin": float(radius - np.linalg.norm(theta_bar)),
        "finite": bool(np.all(np.isfinite(theta_bar))),
        "radius": float(radius),
        "burnin_boundary_rate": float(burn.diagnostics.get("boundary_rate", 0.0)),
        "warm_point_norm": float(np.linalg.norm(theta_w)),
        "warm_point_move": float(np.linalg.norm(theta_bar - theta_w)),
        **pilot_info,
    }
    return SGDFit(theta_bar, diagnostics)


def _sampled_placements_one_sample(
    Xs: Array,
    y: Array,
    theta: Array,
    problem: Problem,
    rng: np.random.Generator,
    bandwidth: float,
    partners: int,
    pair_tracker: Optional[_PairIdentityTracker] = None,
) -> Array:
    n, d = Xs.shape
    base = np.repeat(np.arange(n, dtype=np.int64), partners)
    raw = rng.integers(0, n - 1, size=base.size, dtype=np.int64)
    mate = raw + (raw >= base)
    if pair_tracker is not None:
        pair_tracker.record(base, mate)
    D = Xs[base] - Xs[mate]
    w = y[base] - y[mate]
    _, scalar, _ = scalar_loss_score_curvature(problem, w, D @ theta, bandwidth)
    G = D * scalar[:, None]
    return G.reshape(n, partners, d).mean(axis=1)


def _sampled_placements_two_sample(
    Xs: Array,
    y: Array,
    theta: Array,
    rng: np.random.Generator,
    bandwidth: float,
    partners: int,
    pair_tracker: Optional[_PairIdentityTracker] = None,
) -> tuple[Array, Array, int, int]:
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    np_, nn = pos.size, neg.size
    # Positive placements h_+(z+)=E_- g(z+,Z-).
    pbase = np.repeat(pos, partners)
    pmate = neg[rng.integers(0, nn, size=pbase.size)]
    if pair_tracker is not None:
        pair_tracker.record(pbase, pmate)
    Dp = Xs[pbase] - Xs[pmate]
    _, sp, _ = scalar_loss_score_curvature(
        "auc", np.zeros(pbase.size), Dp @ theta, bandwidth
    )
    hp = (Dp * sp[:, None]).reshape(np_, partners, Xs.shape[1]).mean(axis=1)
    # Negative placements h_-(z-)=E_+ g(Z+,z-).
    nbase = np.repeat(neg, partners)
    nmate = pos[rng.integers(0, np_, size=nbase.size)]
    if pair_tracker is not None:
        pair_tracker.record(nmate, nbase)
    Dn = Xs[nmate] - Xs[nbase]
    _, sn, _ = scalar_loss_score_curvature(
        "auc", np.zeros(nbase.size), Dn @ theta, bandwidth
    )
    hm = (Dn * sn[:, None]).reshape(nn, partners, Xs.shape[1]).mean(axis=1)
    return hp, hm, np_, nn


def sampled_hajek_sandwich(
    Xs: Array,
    y: Array,
    theta: Array,
    problem: Problem,
    rng: np.random.Generator,
    *,
    bandwidth: float = 1.0,
    hessian_pairs: Optional[int] = None,
    partners: Optional[int] = None,
    stream_role: str = "sandwich",
    pair_union_tracker: Optional[_PairIdentityTracker] = None,
) -> tuple[Array, Array, Array, dict]:
    """Sampled Hessian and sampled placement-value sandwich on fixed dimension."""
    n, d = Xs.shape
    if d == 0:
        empty_identity = {
            "pair_identity_scope": stream_role,
            "pair_evaluations_total": 0,
            "pair_ids_distinct_exact": [],
            "pair_ids_distinct_count_exact": 0,
            "pair_ids_distinct_sha256": hashlib.sha256(b"").hexdigest(),
            "distinct_unavailable_reason": None,
            "complete": True,
        }
        return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0)), {
            "pair_evaluations_total": 0,
            "pair_instrumentation": empty_identity,
            "pair_accounting": _pair_accounting(
                empty_identity, objective_calls=0, gradient_calls=0,
                hessian_calls=0, stagewise={},
            ),
        }
    if hessian_pairs is None:
        hessian_pairs = int(math.ceil(n * math.log(max(n, 3))))
    if partners is None:
        partners = int(math.ceil(math.log(max(n, 3)) ** 2))
    pair_tracker = _PairIdentityTracker(
        n, int(hessian_pairs + n * partners), stream_role,
        mirror=pair_union_tracker,
    )
    stage_seeds = rng.integers(0, np.iinfo(np.int64).max, size=2, dtype=np.int64)
    hessian_rng = np.random.default_rng(int(stage_seeds[0]))
    placement_rng = np.random.default_rng(int(stage_seeds[1]))
    stream_audit = _child_stream_audit(
        stream_role, ("hessian", "placements"), stage_seeds
    )
    H = _sampled_hessian(
        Xs, y, theta, problem, hessian_rng, bandwidth, hessian_pairs,
        pair_tracker=pair_tracker,
    )
    H = (H + H.T) / 2.0
    if problem == "rank":
        placements = _sampled_placements_one_sample(
            Xs, y, theta, problem, placement_rng, bandwidth, partners,
            pair_tracker=pair_tracker,
        )
        centered = placements - placements.mean(axis=0)
        Omega = 4.0 * (centered.T @ centered) / n
    else:
        hp, hm, np_, nn = _sampled_placements_two_sample(
            Xs, y, theta, placement_rng, bandwidth, partners,
            pair_tracker=pair_tracker,
        )
        hp -= hp.mean(axis=0)
        hm -= hm.mean(axis=0)
        N = np_ + nn
        Omega = (N / np_) * (hp.T @ hp / np_) + (N / nn) * (hm.T @ hm / nn)
    Omega = (Omega + Omega.T) / 2.0
    eigH = np.linalg.eigvalsh(H)
    if eigH.min() <= 1e-10:
        raise np.linalg.LinAlgError(
            f"sampled Hessian is not positive definite: min eigenvalue {eigH.min():.3e}"
        )
    invH = np.linalg.inv(H)
    V = invH @ Omega @ invH
    V = (V + V.T) / 2.0
    eigV = np.linalg.eigvalsh(V)
    pair_identity = pair_tracker.summary()
    hessian_calls = int(math.ceil(hessian_pairs / 8192))
    placement_calls = 1 if problem == "rank" else 2
    return H, Omega, V, {
        "stage_streams": ["sandwich_hessian", "sandwich_placements"],
        "stage_streams_independently_seeded": True,
        **stream_audit,
        "pair_evaluations_total": int(hessian_pairs + n * partners),
        "pair_draws": int(hessian_pairs + n * partners),
        "stage_pair_evaluations": {
            "sandwich_hessian": int(hessian_pairs),
            "sandwich_placements": int(n * partners),
        },
        "pair_instrumentation": pair_identity,
        "pair_accounting": _pair_accounting(
            pair_identity, objective_calls=0, gradient_calls=placement_calls,
            hessian_calls=hessian_calls,
            stagewise={
                "sandwich_hessian": {
                    "pair_evaluations_total": int(hessian_pairs),
                    "hessian_call_count": hessian_calls,
                },
                "sandwich_placements": {
                    "pair_evaluations_total": int(n * partners),
                    "gradient_call_count": placement_calls,
                },
            },
        ),
        "objective_call_count": 0,
        "gradient_call_count": placement_calls,
        "hessian_call_count": hessian_calls,
        "hessian_pairs": int(hessian_pairs),
        "partners_per_unit": int(partners),
        "hessian_min_eigenvalue": float(eigH.min()),
        "hessian_max_eigenvalue": float(eigH.max()),
        "sandwich_min_eigenvalue": float(eigV.min()),
        "sandwich_max_eigenvalue": float(eigV.max()),
    }


def _top_s_local(theta: Array, s: int) -> NDArray[np.int64]:
    return np.sort(np.lexsort((np.arange(theta.size), -np.abs(theta)))[:s])


@dataclass
class StochasticScreenCleanResult:
    candidate: NDArray[np.int64]
    selected: NDArray[np.int64]
    clean_theta: Array
    final_theta: Array
    final_V: Array
    diagnostics: dict


def fit_stochastic_screen_clean(
    X: Array,
    y: Array,
    problem: Problem,
    K: int,
    rng: np.random.Generator,
    *,
    bandwidth: float = 1.0,
    known_s: Optional[int] = None,
    radius: float = 20.0,
    batch_size: Optional[int] = None,
    eta0: float = 1.0,
    exponent: float = 2.0 / 3.0,
    clean_steps: Optional[int] = None,
    final_steps: Optional[int] = None,
    split_indices: Optional[tuple[NDArray[np.int64], NDArray[np.int64]]] = None,
) -> StochasticScreenCleanResult:
    """Fit the scalable incomplete-U screen / pair-SGD / sampled-sandwich method."""
    if split_indices is None:
        idx1, idx2 = stratified_half_split(y, problem, rng)
    else:
        idx1 = np.asarray(split_indices[0], dtype=np.int64)
        idx2 = np.asarray(split_indices[1], dtype=np.int64)
        if idx1.size + idx2.size != y.size or np.intersect1d(idx1, idx2).size:
            raise ValueError("split_indices must be a disjoint partition of the base units")
        if np.unique(np.concatenate([idx1, idx2])).size != y.size:
            raise ValueError("split_indices must cover every base unit exactly once")
    candidate, screen_score, screen_info = incomplete_score_screen(
        X[idx1], y[idx1], problem, K, rng, bandwidth
    )
    clean_fit = warm_preconditioned_pair_sgd(
        np.ascontiguousarray(X[idx2][:, candidate]),
        y[idx2], problem, rng,
        bandwidth=bandwidth, radius=radius, n_steps=clean_steps,
        batch_size=batch_size, eta0=eta0, exponent=exponent,
        stream_role="legacy_split_clean",
    )
    Hc, Oc, Vc, clean_sand = sampled_hajek_sandwich(
        np.ascontiguousarray(X[idx2][:, candidate]), y[idx2], clean_fit.theta,
        problem, rng, bandwidth=bandwidth, stream_role="legacy_split_clean_sandwich",
    )
    clean_diag = np.diag(Vc)
    if np.any(~np.isfinite(clean_diag)) or np.any(clean_diag <= 0.0):
        raise FloatingPointError("clean sampled sandwich has invalid diagonal")
    if known_s is not None:
        keep_local = _top_s_local(clean_fit.theta, known_s)
        selected = np.sort(candidate[keep_local])
        threshold = {"route": "known_s", "known_s": int(known_s)}
    else:
        critical = float(math.sqrt(2.0 * math.log(K * idx2.size)))
        se = np.sqrt(clean_diag / idx2.size)
        keep = np.abs(clean_fit.theta) > critical * se
        selected = candidate[keep]
        threshold = {
            "route": "unknown_s",
            "critical_value": critical,
            "selected_size": int(selected.size),
        }
    final_theta = np.zeros(X.shape[1])
    if selected.size == 0:
        final_V = np.zeros((0, 0))
        final_fit_info = {"steps": 0, "selection_empty": True}
        final_sand = {}
    else:
        local_x0 = np.zeros(selected.size)
        lookup = {int(j): r for r, j in enumerate(candidate)}
        for r, j in enumerate(selected):
            local_x0[r] = clean_fit.theta[lookup[int(j)]]
        final_fit = warm_preconditioned_pair_sgd(
            np.ascontiguousarray(X[:, selected]), y, problem, rng,
            bandwidth=bandwidth, radius=radius, n_steps=final_steps,
            batch_size=batch_size, eta0=eta0, exponent=exponent, x0=local_x0,
            stream_role="legacy_split_final",
        )
        _, _, final_V, final_sand = sampled_hajek_sandwich(
            np.ascontiguousarray(X[:, selected]), y, final_fit.theta,
            problem, rng, bandwidth=bandwidth, stream_role="legacy_split_final_sandwich",
        )
        final_theta[selected] = final_fit.theta
        final_fit_info = final_fit.diagnostics
    diagnostics = {
        "screen_fold_size": int(idx1.size),
        "clean_fold_size": int(idx2.size),
        "screen_index_sum": int(idx1.sum()),
        "clean_index_sum": int(idx2.sum()),
        "split_indices_sha256_input": {"screen_sum": int(idx1.sum()), "clean_sum": int(idx2.sum())},
        "screen": screen_info,
        "clean_sgd": clean_fit.diagnostics,
        "clean_sandwich": clean_sand,
        "threshold": threshold,
        "final_sgd": final_fit_info,
        "final_sandwich": final_sand,
        "candidate_indices": candidate.tolist(),
        "selected_indices": selected.tolist(),
        "selection_empty": bool(selected.size == 0),
    }
    clean_full = np.zeros(X.shape[1])
    clean_full[candidate] = clean_fit.theta
    return StochasticScreenCleanResult(
        candidate, selected, clean_full, final_theta, final_V, diagnostics
    )

# ---------------------------------------------------------------------------
# v26 full-data HTP discovery / same-data clean and inference
# ---------------------------------------------------------------------------
from .htp_discovery import HTPCertificationError, fully_corrective_pair_htp


def ambient_pruning_critical(n: int, p: int) -> float:
    """Slowly-diverging theorem schedule for adaptive same-data pruning."""
    if n < 2 or p < 2:
        raise ValueError("n,p must be at least two")
    loglog = math.log(math.log(max(float(n), math.e ** math.e)))
    return float(math.sqrt(2.0 * math.log(max(float(p), math.e)) * loglog))


@dataclass
class FullDataHTPRoutesResult:
    candidate: NDArray[np.int64]
    clean_theta: Array
    clean_V: Array
    known_selected: NDArray[np.int64]
    unknown_selected: NDArray[np.int64]
    known_final_theta: Array
    known_final_V: Array
    unknown_final_theta: Array
    unknown_final_V: Array
    diagnostics: dict


def _final_fit_on_selected(
    X: Array, y: Array, problem: Problem,
    selected: NDArray[np.int64], candidate: NDArray[np.int64],
    clean_local: Array, rng: np.random.Generator, *, bandwidth: float,
    radius: float, eta0: float, exponent: float,
    stream_role: str,
) -> tuple[Array, Array, dict, dict]:
    p = X.shape[1]
    selected = np.asarray(selected, dtype=np.int64)
    full = np.zeros(p)
    if selected.size == 0:
        empty_accounting = _empty_final_pair_accounting(f"{stream_role}.optimizer_and_sandwich")
        return full, np.zeros((0, 0)), {
            "steps": 0, "selection_empty": True,
            "final_pair_accounting": empty_accounting,
        }, {"final_pair_accounting": empty_accounting}
    lookup = {int(j): i for i, j in enumerate(candidate)}
    x0 = np.asarray([clean_local[lookup[int(j)]] for j in selected])
    Xs = np.ascontiguousarray(X[:, selected])
    pair_union_tracker = _new_final_pair_union_tracker(
        X.shape[0], selected.size, f"{stream_role}.optimizer_and_sandwich"
    )
    fit = warm_preconditioned_pair_sgd(
        Xs, y, problem, rng, bandwidth=bandwidth, radius=radius,
        eta0=eta0, exponent=exponent, x0=x0,
        stream_role=f"{stream_role}.optimizer",
        pair_union_tracker=pair_union_tracker,
    )
    _, _, V, sand = sampled_hajek_sandwich(
        Xs, y, fit.theta, problem, rng, bandwidth=bandwidth,
        stream_role=f"{stream_role}.sandwich",
        pair_union_tracker=pair_union_tracker,
    )
    final_accounting = _final_pair_accounting(
        pair_union_tracker, fit.diagnostics, sand
    )
    fit.diagnostics["final_pair_accounting"] = final_accounting
    sand["final_pair_accounting"] = final_accounting
    full[selected] = fit.theta
    return full, V, fit.diagnostics, sand


def fit_full_data_htp_routes(
    X: Array, y: Array, problem: Problem, K: int, known_s: int,
    rng: np.random.Generator, *, radius_buffer: float, bandwidth: float = 1.0,
    radius: float = 20.0, eta0: float = 1.0,
    exponent: float = 2.0 / 3.0,
) -> FullDataHTPRoutesResult:
    """Run full-data HTP discovery and both support-selection routes once."""
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    n, p = X.shape
    if not 1 <= known_s <= K < p:
        raise ValueError("require 1 <= known_s <= K < p")
    seeds = rng.integers(0, np.iinfo(np.int64).max, size=4, dtype=np.int64)
    disc_rng = np.random.default_rng(int(seeds[0]))
    clean_rng = np.random.default_rng(int(seeds[1]))
    known_rng = np.random.default_rng(int(seeds[2]))
    unknown_rng = np.random.default_rng(int(seeds[3]))

    discovery = fully_corrective_pair_htp(
        X, y, problem, K, disc_rng, bandwidth=bandwidth, radius=radius,
        radius_buffer=radius_buffer,
    )
    disc_diag = discovery.diagnostics
    disc_tol = float(disc_diag.get("numerical_gradient_tolerance", 1.0 / n))
    disc_radius = float(disc_diag.get("radius", math.nan))
    disc_min_margin = float(disc_diag.get("minimum_refit_boundary_margin", math.nan))
    if not disc_diag.get("all_refits_success", False):
        failing = next(
            (h for h in disc_diag.get("history", [])
             if not (h["expanded_refit_success"] and h["corrective_refit_success"])),
            None,
        )
        if failing is None:
            stage, support_size, outer_iteration = "restricted_refit", -1, -1
            residual, margin = math.nan, math.nan
        elif not failing["expanded_refit_success"]:
            stage = "expanded_refit"
            support_size = int(failing["union_size"])
            outer_iteration = int(failing["outer"])
            residual = float(failing["expanded_gradient_inf"])
            margin = float(failing["expanded_boundary_margin"])
        else:
            stage = "corrective_refit"
            support_size = int(failing["candidate_size"])
            outer_iteration = int(failing["outer"])
            residual = float(failing["corrective_gradient_inf"])
            margin = float(failing["corrective_boundary_margin"])
        raise HTPCertificationError(
            "HTP restricted refit failed numerical certification",
            stage=stage, support_size=support_size,
            outer_iteration=outer_iteration, residual=residual,
            tolerance=disc_tol, boundary_margin=margin,
            boundary_distance=disc_radius - margin,
        )
    active_grad_inf = float(disc_diag.get("active_gradient_inf", math.inf))
    if active_grad_inf > (1.0 / n):
        raise HTPCertificationError(
            "HTP terminal active score exceeds certified tolerance",
            stage="terminal_active_gradient",
            support_size=int(disc_diag.get("candidate_size", -1)),
            outer_iteration=int(disc_diag.get("outer_iterations", -1)),
            residual=active_grad_inf, tolerance=1.0 / n,
            boundary_margin=disc_min_margin,
            boundary_distance=disc_radius - disc_min_margin,
        )
    candidate = discovery.candidate
    if candidate.size != K:
        raise HTPCertificationError(
            f"HTP candidate has size {candidate.size}, expected {K}",
            stage="candidate_size",
            support_size=int(candidate.size),
            outer_iteration=int(disc_diag.get("outer_iterations", -1)),
            residual=active_grad_inf, tolerance=1.0 / n,
            boundary_margin=disc_min_margin,
            boundary_distance=disc_radius - disc_min_margin,
        )
    Xc = np.ascontiguousarray(X[:, candidate])
    clean = warm_preconditioned_pair_sgd(
        Xc, y, problem, clean_rng, bandwidth=bandwidth, radius=radius,
        eta0=eta0, exponent=exponent, x0=discovery.theta[candidate],
        stream_role="full_data.clean.optimizer",
    )
    _, _, Vc, clean_sand = sampled_hajek_sandwich(
        Xc, y, clean.theta, problem, clean_rng, bandwidth=bandwidth,
        stream_role="full_data.clean.sandwich",
    )
    diag = np.diag(Vc)
    if np.any(~np.isfinite(diag)) or np.any(diag <= 0.0):
        raise FloatingPointError("candidate sampled sandwich has invalid diagonal")

    known_selected = np.sort(candidate[_top_s_local(clean.theta, int(known_s))]).astype(np.int64)
    critical = ambient_pruning_critical(n, p)
    se = np.sqrt(diag / n)
    unknown_selected = np.sort(candidate[np.abs(clean.theta) > critical * se]).astype(np.int64)

    known_theta, known_V, known_fit, known_sand = _final_fit_on_selected(
        X, y, problem, known_selected, candidate, clean.theta, known_rng,
        bandwidth=bandwidth, radius=radius, eta0=eta0, exponent=exponent,
        stream_role="full_data.known_final",
    )
    if np.array_equal(unknown_selected, known_selected):
        unknown_theta, unknown_V = known_theta.copy(), known_V.copy()
        unknown_fit, unknown_sand, shared_final = dict(known_fit), dict(known_sand), True
    else:
        unknown_theta, unknown_V, unknown_fit, unknown_sand = _final_fit_on_selected(
            X, y, problem, unknown_selected, candidate, clean.theta, unknown_rng,
            bandwidth=bandwidth, radius=radius, eta0=eta0, exponent=exponent,
            stream_role="full_data.unknown_final",
        )
        shared_final = False

    clean_full = np.zeros(p); clean_full[candidate] = clean.theta
    return FullDataHTPRoutesResult(
        candidate, clean_full, Vc, known_selected, unknown_selected,
        known_theta, known_V, unknown_theta, unknown_V,
        {
            "version": METHOD_VERSION, "observation_split": False,
            "discovery": discovery.diagnostics,
            "clean_sgd": clean.diagnostics, "clean_sandwich": clean_sand,
            "unknown_pruning": {
                "ambient_dimension": int(p),
                "critical_value": float(critical),
                "critical_rule": "sqrt(2 log(p) log log(n))",
                "slow_divergence_factor": float(math.sqrt(math.log(math.log(max(float(n), math.e ** math.e))))),
            },
            "known_selected": known_selected.tolist(),
            "unknown_selected": unknown_selected.tolist(),
            "known_final_sgd": known_fit, "known_final_sandwich": known_sand,
            "unknown_final_sgd": unknown_fit, "unknown_final_sandwich": unknown_sand,
            "shared_final_fit": bool(shared_final),
            **_child_stream_audit(
                "full_data",
                ("discovery", "clean", "known_final", "unknown_final"),
                seeds,
                consumed=(
                    True,
                    True,
                    bool(known_selected.size),
                    bool((not shared_final) and unknown_selected.size),
                ),
            ),
        },
    )
