"""Memory-bounded exact complete-U computations for difference-form pair features.

These routines are exact for rank and positive-negative AUC objectives with
D_ab = X_a-X_b. They avoid materializing pair-by-feature matrices and are used
by Gate 3 and the formal experiment driver.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, ndtr

from .pairwise_method import project_l2_ball


def rank_rho_fast(residual: np.ndarray, bandwidth: float) -> np.ndarray:
    z = residual / bandwidth
    phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    phi0 = 1.0 / math.sqrt(2.0 * math.pi)
    return residual * (ndtr(z) - 0.5) + bandwidth * (phi - phi0)


class StructuredCompleteU:
    """Exact complete-U evaluator for difference-form rank/AUC objectives.

    Pairwise scalar matrices are processed in row blocks. Gradients use endpoint
    aggregation. Hessians and placements exploit the expansion of
    (x_i-x_j)(x_i-x_j)^T, reducing the dominant cost from O(P d^2) to
    O(P d + N d^2) for d selected coordinates.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, problem: str, bandwidth: float = 1.0, block_rows: int = 128):
        self.X = np.ascontiguousarray(X, dtype=float)
        self.y = np.asarray(y)
        self.problem = problem
        self.bandwidth = float(bandwidth)
        self.block_rows = int(block_rows)
        if problem == "auc":
            self.pos = np.flatnonzero(self.y == 1)
            self.neg = np.flatnonzero(self.y == 0)
            if self.pos.size == 0 or self.neg.size == 0:
                raise ValueError("both AUC classes required")
            self.Xp = np.ascontiguousarray(self.X[self.pos])
            self.Xn = np.ascontiguousarray(self.X[self.neg])
            self.n_pairs = int(self.pos.size * self.neg.size)
        elif problem == "rank":
            self.n_pairs = int(self.X.shape[0] * (self.X.shape[0] - 1) // 2)
        else:
            raise ValueError(problem)

    @property
    def d(self) -> int:
        return self.X.shape[1]

    def value_grad(self, beta: np.ndarray) -> tuple[float, np.ndarray]:
        beta = np.asarray(beta, dtype=float)
        if self.problem == "rank":
            n = self.X.shape[0]
            denom = float(n * (n - 1))
            residual = self.y - self.X @ beta
            c = np.zeros(n)
            total_loss = 0.0
            for start in range(0, n, self.block_rows):
                stop = min(n, start + self.block_rows)
                idx = np.arange(start, stop)
                zraw = residual[idx, None] - residual[None, :]
                loss = rank_rho_fast(zraw, self.bandwidth)
                score = -(ndtr(zraw / self.bandwidth) - 0.5)
                score[np.arange(stop - start), idx] = 0.0
                loss[np.arange(stop - start), idx] = 0.0
                total_loss += float(loss.sum())
                c[idx] += score.sum(axis=1)
                c -= score.sum(axis=0)
            grad = self.X.T @ c / denom
            return total_loss / denom, grad

        sp = self.Xp @ beta
        sn = self.Xn @ beta
        np_, nn = self.Xp.shape[0], self.Xn.shape[0]
        denom = float(np_ * nn)
        cp = np.zeros(np_)
        cn = np.zeros(nn)
        total_loss = 0.0
        for start in range(0, np_, self.block_rows):
            stop = min(np_, start + self.block_rows)
            margin = sp[start:stop, None] - sn[None, :]
            score = -expit(-margin)
            total_loss += float(np.logaddexp(0.0, -margin).sum())
            cp[start:stop] += score.sum(axis=1)
            cn -= score.sum(axis=0)
        grad = (self.Xp.T @ cp + self.Xn.T @ cn) / denom
        return total_loss / denom, grad

    def hessian_placements(self, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        beta = np.asarray(beta, dtype=float)
        d = self.d
        H = np.zeros((d, d))
        if self.problem == "rank":
            n = self.X.shape[0]
            denom = float(n * (n - 1))
            residual = self.y - self.X @ beta
            placements = np.zeros((n, d))
            col_w_total = np.zeros(n)
            for start in range(0, n, self.block_rows):
                stop = min(n, start + self.block_rows)
                idx = np.arange(start, stop)
                XI = self.X[start:stop]
                zraw = residual[idx, None] - residual[None, :]
                z = zraw / self.bandwidth
                score = -(ndtr(z) - 0.5)
                curvature = np.exp(-0.5 * z * z) / (math.sqrt(2.0 * math.pi) * self.bandwidth)
                score[np.arange(stop - start), idx] = 0.0
                curvature[np.arange(stop - start), idx] = 0.0
                row_score = score.sum(axis=1)
                placements[start:stop] = (row_score[:, None] * XI - score @ self.X) / (n - 1)

                row_w = curvature.sum(axis=1)
                col_w_total += curvature.sum(axis=0)
                cross = XI.T @ (curvature @ self.X)
                H += XI.T @ (row_w[:, None] * XI)
                H -= cross + cross.T
            H += self.X.T @ (col_w_total[:, None] * self.X)
            H /= denom
            centered = placements - placements.mean(axis=0)
            Omega = 4.0 * (centered.T @ centered) / n
            return H, Omega, placements

        np_, nn = self.Xp.shape[0], self.Xn.shape[0]
        denom = float(np_ * nn)
        sp = self.Xp @ beta
        sn = self.Xn @ beta
        hp = np.zeros((np_, d))
        hm_sum = np.zeros((nn, d))
        col_w_total = np.zeros(nn)
        for start in range(0, np_, self.block_rows):
            stop = min(np_, start + self.block_rows)
            XP = self.Xp[start:stop]
            margin = sp[start:stop, None] - sn[None, :]
            score = -expit(-margin)
            sprob = expit(margin)
            curvature = sprob * (1.0 - sprob)
            row_score = score.sum(axis=1)
            col_score = score.sum(axis=0)
            hp[start:stop] = (row_score[:, None] * XP - score @ self.Xn) / nn
            hm_sum += score.T @ XP - col_score[:, None] * self.Xn

            row_w = curvature.sum(axis=1)
            col_w_total += curvature.sum(axis=0)
            cross = XP.T @ (curvature @ self.Xn)
            H += XP.T @ (row_w[:, None] * XP)
            H -= cross + cross.T
        H += self.Xn.T @ (col_w_total[:, None] * self.Xn)
        H /= denom
        hm = hm_sum / np_
        hp_c = hp - hp.mean(axis=0)
        hm_c = hm - hm.mean(axis=0)
        covp = hp_c.T @ hp_c / np_
        covm = hm_c.T @ hm_c / nn
        N = np_ + nn
        Omega = (N / np_) * covp + (N / nn) * covm
        return H, Omega, np.vstack([hp, hm])


def restricted_refit_structured(
    X: np.ndarray,
    y: np.ndarray,
    selected: np.ndarray,
    problem: str,
    radius: float,
    bandwidth: float,
    tol: float,
    x0_full: Optional[np.ndarray] = None,
    block_rows: int = 128,
) -> tuple[np.ndarray, dict, StructuredCompleteU]:
    selected = np.asarray(sorted(set(map(int, selected))), dtype=int)
    Xs = np.ascontiguousarray(X[:, selected])
    evaluator = StructuredCompleteU(Xs, y, problem, bandwidth, block_rows)
    if x0_full is None:
        x0 = np.zeros(selected.size)
    else:
        x0 = np.asarray(x0_full)[selected].copy()
        if np.linalg.norm(x0) > radius:
            x0 = project_l2_ball(x0, radius)

    calls = 0
    def fg(beta: np.ndarray):
        nonlocal calls
        calls += 1
        return evaluator.value_grad(beta)

    result = minimize(
        fg,
        x0,
        jac=True,
        method="L-BFGS-B",
        options={"ftol": 1e-15, "gtol": max(tol * 0.1, 1e-12), "maxiter": 1000, "maxls": 50},
    )
    beta = np.asarray(result.x, dtype=float)
    unconstrained_norm = float(np.linalg.norm(beta))
    used_fallback = False
    if unconstrained_norm > radius * (1.0 + 1e-9):
        used_fallback = True
        cons = ({"type": "ineq", "fun": lambda b: radius * radius - float(b @ b), "jac": lambda b: -2.0 * b},)
        result = minimize(
            fg,
            project_l2_ball(beta, radius),
            jac=True,
            method="SLSQP",
            constraints=cons,
            options={"ftol": min(tol, 1e-10), "maxiter": 2000, "disp": False},
        )
        beta = np.asarray(result.x, dtype=float)
    value, grad = evaluator.value_grad(beta)
    beta_norm = float(np.linalg.norm(beta))
    # KKT residual for the l2-ball constrained problem. Raw score infinity norm
    # is meaningful only for an interior solution; on the boundary the normal
    # cone can balance a nonzero gradient.
    projected = project_l2_ball(beta - grad, radius)
    kkt_map = beta - projected
    theta = np.zeros(X.shape[1])
    theta[selected] = beta
    interior_margin = float(radius - beta_norm)
    boundary_active = bool(interior_margin <= 1e-6 * max(1.0, radius))
    score_inf = float(np.max(np.abs(grad)))
    kkt_inf = float(np.max(np.abs(kkt_map)))
    info = {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", -1)),
        "function_calls": int(calls),
        "objective": float(value),
        "score_inf": score_inf,
        "kkt_inf": kkt_inf,
        "numerical_residual_inf": kkt_inf if boundary_active else score_inf,
        "numerical_residual_pass": bool((kkt_inf if boundary_active else score_inf) <= tol),
        "boundary_active": boundary_active,
        "requested_tolerance": float(tol),
        "norm": beta_norm,
        "radius": float(radius),
        "interior_margin": interior_margin,
        "used_constrained_fallback": used_fallback,
        "pair_count": evaluator.n_pairs,
        "dimension": int(selected.size),
    }
    return theta, info, evaluator


def sandwich_structured(evaluator: StructuredCompleteU, theta_selected: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    H, Omega, _ = evaluator.hessian_placements(theta_selected)
    H = (H + H.T) / 2.0
    Omega = (Omega + Omega.T) / 2.0
    eigH = np.linalg.eigvalsh(H)
    invH = np.linalg.inv(H)
    V = invH @ Omega @ invH
    V = (V + V.T) / 2.0
    eigV = np.linalg.eigvalsh(V)
    return H, Omega, V, {
        "hessian_min_eigenvalue": float(eigH.min()),
        "hessian_max_eigenvalue": float(eigH.max()),
        "sandwich_min_eigenvalue": float(eigV.min()),
        "sandwich_max_eigenvalue": float(eigV.max()),
    }


