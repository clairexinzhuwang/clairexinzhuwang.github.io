"""Balanced-matching fully corrective HTP for sparse pairwise M-estimation.

Discovery uses R_N=ceil(log(p vee e)) independent maximal matchings. The
score at zero is the first HTP proposal; later full-p residual gradients allow
marginally weak variables to enter after joint refitting. All corrective
optimization is restricted to at most 3K coordinates and is certified by the
actual restricted score, not by a solver status flag alone.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from .pairwise_method import (scalar_loss_score_curvature, endpoint_aggregated_gradient,
                              project_l2_ball)

Array = NDArray[np.float64]
Problem = Literal["rank", "auc"]


def _fixed_pool_identity(
    n: int, pair_i: NDArray[np.int64], pair_j: NDArray[np.int64], scope: str
) -> dict:
    lo = np.minimum(np.asarray(pair_i, dtype=np.int64), np.asarray(pair_j, dtype=np.int64))
    hi = np.maximum(np.asarray(pair_i, dtype=np.int64), np.asarray(pair_j, dtype=np.int64))
    unique_ids = np.unique(lo * int(n) + hi)
    canonical = np.asarray(unique_ids, dtype="<i8")
    return {
        "pair_identity_scope": str(scope),
        "pair_identity_definition": "unordered min(i,j)*N+max(i,j) on base-unit indices",
        "pair_ids_distinct_exact": None,
        # ``unique_ids`` is materialized above, so this count and its canonical
        # digest are exact producer facts rather than an inferred certificate.
        "pair_ids_distinct_count_is_exact": True,
        "pair_ids_distinct_count_exact": int(unique_ids.size),
        "pair_ids_distinct_sha256": hashlib.sha256(canonical.tobytes()).hexdigest(),
        "distinct_unavailable_reason": None,
    }


def matching_rounds(p: int) -> int:
    if p < 2:
        raise ValueError("p must be at least two")
    return int(math.ceil(math.log(max(float(p), math.e))))


def balanced_pair_pool(
    y: Array,
    problem: Problem,
    rounds: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.int64], NDArray[np.int64], dict]:
    """Concatenate independent maximal matchings without truncating a round."""
    y = np.asarray(y)
    n = int(y.size)
    if n < 2 or rounds < 1:
        raise ValueError("n>=2 and rounds>=1 are required")
    out_i: list[NDArray[np.int64]] = []
    out_j: list[NDArray[np.int64]] = []
    per_round: list[int] = []
    if problem == "auc":
        pos0 = np.flatnonzero(y == 1)
        neg0 = np.flatnonzero(y == 0)
        if pos0.size == 0 or neg0.size == 0:
            raise ValueError("both AUC classes are required")
    for _ in range(int(rounds)):
        if problem == "rank":
            perm = rng.permutation(n)
            m = n // 2
            ii = np.asarray(perm[: 2 * m : 2], dtype=np.int64)
            jj = np.asarray(perm[1 : 2 * m : 2], dtype=np.int64)
        elif problem == "auc":
            pos = rng.permutation(pos0)
            neg = rng.permutation(neg0)
            m = min(pos.size, neg.size)
            ii = np.asarray(pos[:m], dtype=np.int64)
            jj = np.asarray(neg[:m], dtype=np.int64)
        else:
            raise ValueError(problem)
        if ii.size == 0:
            raise RuntimeError("a maximal matching produced no pair")
        out_i.append(ii); out_j.append(jj); per_round.append(int(ii.size))
    pair_i = np.concatenate(out_i)
    pair_j = np.concatenate(out_j)
    identity = _fixed_pool_identity(n, pair_i, pair_j, "htp_balanced_matching_pool")
    return pair_i, pair_j, {
        "matching_rounds": int(rounds),
        "matching_round_rule": "ceil(log(p vee e))",
        "pair_budget": int(pair_i.size),
        "pairs_per_round_min": int(min(per_round)),
        "pairs_per_round_max": int(max(per_round)),
        "balanced_maximal_matchings": True,
        "within_round_base_reuse": False,
        "complete_pair_enumeration": False,
        "pair_pool_identity": identity,
    }


def _top_abs(values: Array, k: int) -> NDArray[np.int64]:
    k = min(max(int(k), 0), int(values.size))
    if k == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.lexsort((np.arange(values.size), -np.abs(values)))
    return np.sort(order[:k].astype(np.int64))



def polish_iteration_cap(n: int) -> int:
    """Deterministic projected-gradient budget ceil(log^2(n vee 3))."""
    if n < 2:
        raise ValueError("n must be at least two")
    return int(math.ceil(math.log(max(float(n), 3.0)) ** 2))


def projected_gradient_polish(
    fg,
    start: Array,
    radius: float,
    nu: float,
    cap: int,
    *,
    step_init: float = 1.0,
) -> tuple[Array, float, float, dict]:
    """Certify constrained stationarity by a projected-gradient mapping.

    L-BFGS is only an initializer.  The returned residual is
    ``||theta-Proj(theta-grad f(theta))||_inf`` at the fixed reference step one.
    The best residual seen is returned so the polish can never worsen its input.
    """
    theta = project_l2_ball(np.asarray(start, dtype=float), float(radius))
    value, grad = fg(theta)
    value = float(value); grad = np.asarray(grad, dtype=float)
    tau = float(step_init)
    cap = max(1, int(cap))

    def residual_ref(th: Array, g: Array) -> float:
        gm = th - project_l2_ball(th - g, radius)
        return float(np.max(np.abs(gm))) if gm.size else 0.0

    def residual_step(th: Array, g: Array, step: float) -> float:
        gm = (th - project_l2_ball(th - step * g, radius)) / step
        return float(np.max(np.abs(gm))) if gm.size else 0.0

    finite = bool(np.isfinite(value) and np.all(np.isfinite(theta)) and np.all(np.isfinite(grad)))
    residual = residual_ref(theta, grad) if finite else float('inf')
    initial = float(residual)
    best_theta = theta.copy(); best_grad = grad.copy(); best_value = value
    best_residual = residual; best_index = 0
    iterations = 0; backtracks = 0; inner_exhausted = False

    while finite and iterations < cap and residual > nu:
        accepted = False
        for _ in range(cap):
            trial = project_l2_ball(theta - tau * grad, radius)
            step_vec = trial - theta
            trial_value, trial_grad = fg(trial)
            trial_value = float(trial_value); trial_grad = np.asarray(trial_grad, dtype=float)
            model = value + float(grad @ step_vec) + float(step_vec @ step_vec) / (2.0 * tau)
            if (np.isfinite(trial_value) and np.all(np.isfinite(trial_grad))
                    and trial_value <= model + 1e-14):
                theta, value, grad = trial, trial_value, trial_grad
                accepted = True
                break
            tau *= 0.5
            backtracks += 1
        if not accepted:
            inner_exhausted = True
            break
        iterations += 1
        finite = bool(np.isfinite(value) and np.all(np.isfinite(theta)) and np.all(np.isfinite(grad)))
        residual = residual_ref(theta, grad) if finite else float('inf')
        if residual < best_residual:
            best_theta = theta.copy(); best_grad = grad.copy(); best_value = value
            best_residual = residual; best_index = iterations

    theta = best_theta; grad = best_grad; value = float(best_value); residual = float(best_residual)
    diagnostics = {
        "polish_iterations": int(iterations),
        "polish_iterations_to_best": int(best_index),
        "polish_cap": int(cap),
        "polish_backtracks": int(backtracks),
        "polish_final_step": float(tau),
        "polish_reference_step": 1.0,
        "polish_residual_initial": float(initial),
        "polish_residual_final": float(residual),
        "polish_residual_at_accepted_step": float(residual_step(theta, grad, tau)),
        "polish_cap_binding": bool(finite and residual > nu),
        "polish_inner_budget_exhausted": bool(inner_exhausted),
        "polish_nonfinite": bool(not finite),
    }
    return theta, value, residual, diagnostics


@dataclass
class RestrictedFit:
    support: NDArray[np.int64]
    coef: Array
    objective: float
    gradient_inf: float
    success: bool
    iterations: int
    message: str
    norm: float
    boundary_margin: float
    diagnostics: dict


class IncompletePairRisk:
    def __init__(self, X: Array, y: Array, problem: Problem,
                 pair_i: NDArray[np.int64], pair_j: NDArray[np.int64],
                 bandwidth: float) -> None:
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y)
        self.problem = problem
        self.pair_i = np.asarray(pair_i, dtype=np.int64)
        self.pair_j = np.asarray(pair_j, dtype=np.int64)
        self.bandwidth = float(bandwidth)
        if self.pair_i.shape != self.pair_j.shape or self.pair_i.size == 0:
            raise ValueError("pair arrays must be aligned and nonempty")
        self.w = (self.y[self.pair_i] - self.y[self.pair_j]
                  if problem == "rank" else np.zeros(self.pair_i.size))
        self.base_loss_call_count = 1
        self.full_gradient_call_count = 0
        self.joint_objective_gradient_call_count = 0
        self.objective_callback_call_count = 0
        self.gradient_callback_call_count = 0
        zero = np.zeros(self.pair_i.size)
        self.base_loss = scalar_loss_score_curvature(
            problem, self.w, zero, self.bandwidth
        )[0]

    @property
    def n_pairs(self) -> int:
        return int(self.pair_i.size)

    def margin(self, support: NDArray[np.int64], coef: Array) -> Array:
        support = np.asarray(support, dtype=np.int64)
        if support.size == 0:
            return np.zeros(self.n_pairs)
        Xs = self.X[:, support]  # restrict before endpoint indexing
        return (Xs[self.pair_i] - Xs[self.pair_j]) @ np.asarray(coef, dtype=float)

    def full_gradient(self, support: NDArray[np.int64], coef: Array) -> Array:
        self.full_gradient_call_count += 1
        margin = self.margin(support, coef)
        _, score, _ = scalar_loss_score_curvature(
            self.problem, self.w, margin, self.bandwidth
        )
        return endpoint_aggregated_gradient(self.X, self.pair_i, self.pair_j, score)

    def restricted_fit(self, support: NDArray[np.int64], x0: Optional[Array],
                       gradient_tolerance: float, radius: float,
                       radius_buffer: float) -> RestrictedFit:
        support = np.asarray(sorted(set(map(int, support))), dtype=np.int64)
        if support.size == 0:
            return RestrictedFit(support, np.zeros(0), 0.0, float("inf"), False, 0,
                                 "empty support is not a fitted model", 0.0, float(radius), {
                                     "pair_evaluations_total": 0,
                                     "objective_call_count": 0,
                                     "gradient_call_count": 0,
                                     "hessian_call_count": 0,
                                     "joint_objective_gradient_call_count": 0,
                                 })
        Xs = self.X[:, support]
        D = Xs[self.pair_i] - Xs[self.pair_j]
        start = (np.zeros(support.size) if x0 is None or
                 np.asarray(x0).shape != (support.size,) else
                 np.asarray(x0, dtype=float).copy())

        start_joint = self.joint_objective_gradient_call_count
        start_objective_callbacks = self.objective_callback_call_count
        start_gradient_callbacks = self.gradient_callback_call_count

        def fg(theta: Array) -> tuple[float, Array]:
            self.joint_objective_gradient_call_count += 1
            margin = D @ theta
            loss, score, _ = scalar_loss_score_curvature(
                self.problem, self.w, margin, self.bandwidth
            )
            return float(np.mean(loss - self.base_loss)), D.T @ score / self.n_pairs

        def objective(z: Array) -> float:
            self.objective_callback_call_count += 1
            return fg(z)[0]

        def gradient(z: Array) -> Array:
            self.gradient_callback_call_count += 1
            return fg(z)[1]

        res = minimize(objective, start, jac=gradient,
                       method="L-BFGS-B",
                       options={"maxiter": 500, "gtol": float(gradient_tolerance),
                                "ftol": 1e-14, "maxls": 50})
        # L-BFGS is an initializer only.  A projected-gradient polish supplies
        # the constrained first-order certificate used by the theorem and gates.
        coef, value, grad_inf, polish = projected_gradient_polish(
            fg, np.asarray(res.x, dtype=float), radius, gradient_tolerance,
            polish_iteration_cap(self.X.shape[0]),
        )
        _, raw_grad = fg(coef)
        raw_grad_inf = float(np.max(np.abs(raw_grad)))
        norm = float(np.linalg.norm(coef))
        boundary_margin = float(radius - norm)
        finite = bool(np.isfinite(value) and np.all(np.isfinite(coef)) and np.isfinite(grad_inf))
        score_ok = bool(grad_inf <= gradient_tolerance)
        interior = bool(boundary_margin >= 0.5 * radius_buffer)
        success = bool(finite and score_ok and interior)
        message = str(res.message)
        if not score_ok:
            message = f"projected-gradient residual {grad_inf:.3e} exceeds tolerance {gradient_tolerance:.3e}; {message}"
        if not interior:
            message = (f"restricted fit boundary margin {boundary_margin:.6g} is below "
                       f"required {0.5*radius_buffer:.6g}; {message}")
        # Attach implementation diagnostics without trusting the scipy status flag.
        message += (f"; raw_grad_inf={raw_grad_inf:.3e}; polish_iters={polish['polish_iterations']}; "
                    f"polish_backtracks={polish['polish_backtracks']}")
        joint_calls = int(self.joint_objective_gradient_call_count - start_joint)
        objective_callbacks = int(self.objective_callback_call_count - start_objective_callbacks)
        gradient_callbacks = int(self.gradient_callback_call_count - start_gradient_callbacks)
        fit_accounting = {
            "pair_evaluations_total": int(self.n_pairs * joint_calls),
            "objective_call_count": joint_calls,
            "gradient_call_count": joint_calls,
            "hessian_call_count": 0,
            "joint_objective_gradient_call_count": joint_calls,
            "objective_callback_call_count": objective_callbacks,
            "gradient_callback_call_count": gradient_callbacks,
        }
        return RestrictedFit(support, coef, value, grad_inf, success,
                             int(getattr(res, "nit", -1)) + int(polish["polish_iterations"]),
                             message, norm, boundary_margin, fit_accounting)


@dataclass
class HTPDiscoveryResult:
    candidate: NDArray[np.int64]
    theta: Array
    diagnostics: dict


class HTPCertificationError(RuntimeError):
    """Certification failure with machine-readable context for arm-local records."""

    def __init__(self, message: str, *, stage: str, support_size: int,
                 outer_iteration: int, residual: float, tolerance: float,
                 boundary_margin: float, boundary_distance: float) -> None:
        super().__init__(message)
        self.stage = str(stage)
        self.support_size = int(support_size)
        self.outer_iteration = int(outer_iteration)
        self.residual = float(residual)
        self.tolerance = float(tolerance)
        self.boundary_margin = float(boundary_margin)
        self.boundary_distance = float(boundary_distance)

    def context(self) -> dict:
        return {
            "stage": self.stage,
            "support_size": self.support_size,
            "outer_iteration": self.outer_iteration,
            "residual": self.residual,
            "tolerance": self.tolerance,
            "boundary_margin": self.boundary_margin,
            "boundary_distance": self.boundary_distance,
        }


def fully_corrective_pair_htp(
    X: Array, y: Array, problem: Problem, K: int, rng: np.random.Generator,
    *, radius_buffer: float, bandwidth: float = 1.0, radius: float = 20.0,
    rounds: Optional[int] = None, proposal_multiple: int = 2,
    max_outer: Optional[int] = None,
) -> HTPDiscoveryResult:
    """Score-initialized global-gradient fully corrective pairwise HTP."""
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    n, p = X.shape
    if not 1 <= K < p:
        raise ValueError("K must lie in [1,p-1]")
    if proposal_multiple < 1:
        raise ValueError("proposal_multiple must be positive")
    rounds = matching_rounds(p) if rounds is None else int(rounds)
    max_outer = int(K) if max_outer is None else int(max_outer)
    pair_i, pair_j, pool_info = balanced_pair_pool(y, problem, rounds, rng)
    risk = IncompletePairRisk(X, y, problem, pair_i, pair_j, bandwidth)
    active = np.zeros(0, dtype=np.int64); theta_active = np.zeros(0)
    full_theta = np.zeros(p); history: list[dict] = []
    stable_and_stationary = False; all_refits_success = True
    numerical_tol = 1.0 / n

    for outer in range(1, max_outer + 1):
        grad = risk.full_gradient(active, theta_active)
        proposal = _top_abs(grad, min(proposal_multiple * K, p))
        union = np.union1d(active, proposal).astype(np.int64)
        old_map = {int(j): float(v) for j, v in zip(active, theta_active)}
        x0_union = np.asarray([old_map.get(int(j), 0.0) for j in union])
        expanded = risk.restricted_fit(union, x0_union, numerical_tol, radius, radius_buffer)
        all_refits_success = all_refits_success and expanded.success
        keep_local = _top_abs(expanded.coef, K)
        keep = np.sort(expanded.support[keep_local]).astype(np.int64)
        exp_map = {int(j): float(v) for j, v in zip(expanded.support, expanded.coef)}
        x0_keep = np.asarray([exp_map[int(j)] for j in keep])
        corrective = risk.restricted_fit(keep, x0_keep, numerical_tol, radius, radius_buffer)
        all_refits_success = all_refits_success and corrective.success
        new_active, new_coef = corrective.support, corrective.coef
        support_stable = np.array_equal(new_active, active)
        stationary = corrective.gradient_inf <= numerical_tol
        stable_and_stationary = bool(support_stable and stationary)
        history.append({
            "outer": outer, "proposal_size": int(proposal.size),
            "union_size": int(union.size), "candidate_size": int(new_active.size),
            "support_stable": bool(support_stable), "stationary": bool(stationary),
            "expanded_refit_success": bool(expanded.success),
            "corrective_refit_success": bool(corrective.success),
            "expanded_gradient_inf": float(expanded.gradient_inf),
            "corrective_gradient_inf": float(corrective.gradient_inf),
            "expanded_iterations": int(expanded.iterations),
            "corrective_iterations": int(corrective.iterations),
            "expanded_pair_accounting": expanded.diagnostics,
            "corrective_pair_accounting": corrective.diagnostics,
            "objective": float(corrective.objective), "coef_norm": float(corrective.norm),
            "expanded_boundary_margin": float(expanded.boundary_margin),
            "corrective_boundary_margin": float(corrective.boundary_margin),
        })
        active, theta_active = new_active, new_coef
        full_theta[:] = 0.0; full_theta[active] = theta_active
        if stable_and_stationary:
            break

    final_grad = risk.full_gradient(active, theta_active)
    active_grad_inf = float(np.max(np.abs(final_grad[active]))) if active.size else 0.0
    cap_reached = bool(not stable_and_stationary)
    terminal_support_changed = bool(history and not history[-1]["support_stable"])
    if stable_and_stationary and all_refits_success:
        termination_reason = "stable_certified"
    elif (cap_reached and all_refits_success
          and active_grad_inf <= numerical_tol and int(active.size) == int(K)):
        termination_reason = "cap_certified"
    else:
        termination_reason = "uncertified_failure"
    pool_identity = dict(pool_info["pair_pool_identity"])
    pair_evaluation_calls = int(
        risk.base_loss_call_count
        + risk.full_gradient_call_count
        + risk.joint_objective_gradient_call_count
    )
    pair_accounting = {
        **pool_identity,
        "pair_evaluations_total": int(risk.n_pairs * pair_evaluation_calls),
        "objective_call_count": int(
            risk.base_loss_call_count + risk.joint_objective_gradient_call_count
        ),
        "gradient_call_count": int(
            risk.full_gradient_call_count + risk.joint_objective_gradient_call_count
        ),
        "hessian_call_count": 0,
        "stagewise": {
            "base_loss": {
                "pair_evaluations_total": int(risk.n_pairs * risk.base_loss_call_count),
                "objective_call_count": int(risk.base_loss_call_count),
            },
            "full_gradient": {
                "pair_evaluations_total": int(risk.n_pairs * risk.full_gradient_call_count),
                "gradient_call_count": int(risk.full_gradient_call_count),
            },
            "restricted_joint_objective_gradient": {
                "pair_evaluations_total": int(
                    risk.n_pairs * risk.joint_objective_gradient_call_count
                ),
                "objective_call_count": int(risk.joint_objective_gradient_call_count),
                "gradient_call_count": int(risk.joint_objective_gradient_call_count),
                "joint_call_count": int(risk.joint_objective_gradient_call_count),
                "objective_callback_call_count": int(risk.objective_callback_call_count),
                "gradient_callback_call_count": int(risk.gradient_callback_call_count),
                "direct_joint_call_count": int(
                    risk.joint_objective_gradient_call_count
                    - risk.objective_callback_call_count
                    - risk.gradient_callback_call_count
                ),
            },
        },
        "complete": True,
    }
    return HTPDiscoveryResult(active, full_theta, {
        **pool_info, "proposal_multiple": int(proposal_multiple),
        "max_outer": int(max_outer), "outer_iterations": int(len(history)),
        "stable_and_stationary": bool(stable_and_stationary),
        "all_refits_success": bool(all_refits_success),
        "termination_reason": str(termination_reason),
        "cap_reached": bool(cap_reached),
        "terminal_support_changed": bool(terminal_support_changed),
        "numerical_gradient_tolerance": float(numerical_tol),
        "radius": float(radius), "radius_buffer": float(radius_buffer),
        "minimum_refit_boundary_margin": float(min(
            [min(h["expanded_boundary_margin"],h["corrective_boundary_margin"]) for h in history]
            or [float(radius)])),
        "active_gradient_inf": active_grad_inf,
        "candidate_size": int(active.size), "history": history,
        "pair_evaluations_total": int(pair_accounting["pair_evaluations_total"]),
        "objective_call_count": int(pair_accounting["objective_call_count"]),
        "gradient_call_count": int(pair_accounting["gradient_call_count"]),
        "hessian_call_count": 0,
        "pair_accounting": pair_accounting,
    })
