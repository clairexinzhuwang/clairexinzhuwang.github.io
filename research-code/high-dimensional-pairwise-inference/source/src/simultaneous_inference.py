"""Dependence-aware simultaneous intervals on a recovered support.

The selected support is random.  Validity is obtained through exact-recovery
oracleization: on {S_hat=S}, the selected-support Gaussian maximum equals the
oracle maximum on S.  The implementation approximates its conditional quantile
by Gaussian multiplier draws from the estimated correlation matrix.  The finite-draw critical value is an exact Monte Carlo order statistic under
the fitted Gaussian reference law.  Bonferroni is retained only as a transparent
comparator; it is not used to truncate the Gaussian-max critical value.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

Array = NDArray[np.float64]
INFERENCE_METHOD = "selected_support_gaussian_max_monte_carlo_rank"
ALPHA_FLOOR_DRAWS = 19  # = ceil(1/0.05) - 1; smallest M with k = ceil((M+1)*0.95) <= M
JSON_POSITIVE_INFINITY = "+infinity"


def off_c_n_interval_metrics(
    *, gaussian_seed: int, calibration_stream_role: str = "unavailable"
) -> dict[str, Any]:
    """Total JSON-safe inference record for an outcome outside reportability set C_N.

    The legacy nullable interval fields are retained for compatibility with the
    frozen consumer schema.  ``off_c_n_totality`` is the authoritative totalized
    convention: the standardized statistic is +infinity, its critical value is
    zero, and the coverage event is false.
    """
    return {
        "inference_method": INFERENCE_METHOD,
        "calibration_mode": "off_c_n_totality",
        "calibration_stream_role": str(calibration_stream_role),
        "calibration_reportable": False,
        "selected_joint_coverage": False,
        "active_joint_coverage": False,
        "mean_interval_width": None,
        "mean_raw_gaussian_max_interval_width": None,
        "mean_bonferroni_interval_width": None,
        "max_abs_bias_selected": None,
        "l2_bias_selected": None,
        "raw_critical_value": None,
        "critical_value": None,
        "single_coordinate_critical_value": None,
        "bonferroni_critical_value": None,
        "lower_safeguard_active": None,
        "raw_exceeds_bonferroni": None,
        "monte_carlo_order_index": 0,
        "monte_carlo_reference_level": None,
        "raw_critical_to_bonferroni_ratio": None,
        "critical_to_bonferroni_ratio": None,
        "gaussian_draws": 0,
        "gaussian_seed": int(gaussian_seed),
        "selected_dimension": 0,
        "correlation_min_eigenvalue_raw": None,
        "base_sample_size": None,
        "alpha": None,
        "coordinate_indices": [],
        "coordinate_estimates": [],
        "coordinate_targets": [],
        "covariance_matrix": [],
        "variance_diagonal": [],
        "standard_errors": [],
        "interval_lower": [],
        "interval_upper": [],
        "coordinate_covered": [],
        "coordinate_active": [],
        "standardized_abs_errors": [],
        "off_c_n_totality": {
            "applies": True,
            "standardized_statistic": JSON_POSITIVE_INFINITY,
            "critical_value": 0.0,
            "coverage": False,
            "infinity_encoding": JSON_POSITIVE_INFINITY,
        },
    }


def gaussian_draw_count(n: int) -> int:
    """Deterministic DKW-based auxiliary-draw schedule.

    With epsilon_n = 1/(2 log n) and delta_n=n^{-2}, the DKW inequality is
    bounded by delta_n once M >= 2 log^2(n) log(2 n^2).  The exact ceiling is
    an implementation sequence, not a coverage-tuned constant.
    """
    x = float(max(int(n), 3))
    base = int(math.ceil(2.0 * math.log(x) ** 2 * math.log(2.0 * x * x)))
    # The order statistic k = ceil((M+1)(1-alpha)) must exist, i.e. k <= M, which needs
    # M + 1 >= 1/alpha. Enforcing the floor here makes the schedule total rather than
    # leaving a small-n hole for the caller to assume away. At alpha = 0.05 the floor is
    # 19, so this changes nothing for n >= 5 and nothing anywhere on the formal grid.
    return max(base, ALPHA_FLOOR_DRAWS)


def _correlation_from_covariance(v: Array) -> tuple[Array, float]:
    v = np.asarray(v, dtype=float)
    if v.ndim != 2 or v.shape[0] != v.shape[1]:
        raise ValueError("V must be square")
    v = (v + v.T) / 2.0
    diag = np.diag(v)
    if np.any(~np.isfinite(diag)) or np.any(diag <= 0.0):
        raise FloatingPointError("simultaneous inference requires positive finite variance diagonal")
    sd = np.sqrt(diag)
    r = v / np.outer(sd, sd)
    r = (r + r.T) / 2.0
    # Sampling noise can create tiny negative eigenvalues.  Project only the
    # numerical negative part away, then renormalize to a correlation matrix.
    eig, q = np.linalg.eigh(r)
    raw_min = float(eig.min())
    eig = np.maximum(eig, 0.0)
    r = (q * eig) @ q.T
    d = np.sqrt(np.maximum(np.diag(r), 1e-15))
    r = r / np.outer(d, d)
    r = (r + r.T) / 2.0
    np.fill_diagonal(r, 1.0)
    return r, raw_min


def gaussian_max_critical(
    v: Array,
    alpha: float,
    n: int,
    seed: int,
) -> dict[str, Any]:
    """Approximate the conditional Gaussian-max critical value.

    The covariance estimate is converted to a correlation matrix. Draws are
    generated from that estimated Gaussian law and the empirical (1-alpha)
    quantile of max |Z_j| is retained as ``raw_critical_value``. The operational
    interval uses the exact Monte Carlo order-statistic rule. A lower projection
    to the exact one-coordinate critical value is conservative. Bonferroni is
    recorded only as a comparator and never truncates the operational value.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    d = int(np.asarray(v).shape[0])
    if d < 1:
        raise ValueError("selected dimension must be positive")
    single = float(norm.ppf(1.0 - alpha / 2.0))
    bonf = float(norm.ppf(1.0 - alpha / (2.0 * d)))
    if d == 1:
        return {
            "calibration_mode": "exact_one_coordinate_normal",
            "raw_critical_value": single,
            "critical_value": single,
            "single_coordinate_critical_value": single,
            "bonferroni_critical_value": bonf,
            "gaussian_draws": 0,
            "gaussian_seed": int(seed),
            "selected_dimension": 1,
            "correlation_min_eigenvalue_raw": 1.0,
            "lower_safeguard_active": False,
            "raw_exceeds_bonferroni": False,
            "monte_carlo_order_index": 0,
            "monte_carlo_reference_level": 1.0 - alpha,
            "raw_critical_to_bonferroni_ratio": 1.0,
            "critical_to_bonferroni_ratio": 1.0,
        }
    r, raw_min = _correlation_from_covariance(v)
    m = gaussian_draw_count(n)
    rng = np.random.default_rng(int(seed))
    eig, q = np.linalg.eigh(r)
    root = q * np.sqrt(np.maximum(eig, 0.0))
    z = rng.standard_normal((m, d)) @ root.T
    maxima = np.max(np.abs(z), axis=1)
    order_index = int(math.ceil((m + 1) * (1.0 - alpha)))
    if order_index > m:
        raise ValueError("Gaussian draw count is too small for the requested Monte Carlo level")
    raw_crit = float(np.partition(maxima, order_index - 1)[order_index - 1])
    lower_active = bool(raw_crit < single)
    crit = float(max(raw_crit, single))
    return {
        "calibration_mode": "gaussian_max_monte_carlo_order_statistic",
        "raw_critical_value": raw_crit,
        "critical_value": crit,
        "single_coordinate_critical_value": single,
        "bonferroni_critical_value": bonf,
        "gaussian_draws": int(m),
        "gaussian_seed": int(seed),
        "selected_dimension": d,
        "correlation_min_eigenvalue_raw": raw_min,
        "lower_safeguard_active": lower_active,
        "raw_exceeds_bonferroni": bool(raw_crit > bonf),
        "monte_carlo_order_index": order_index,
        "monte_carlo_reference_level": float(order_index / (m + 1)),
        "raw_critical_to_bonferroni_ratio": float(raw_crit / bonf),
        "critical_to_bonferroni_ratio": float(crit / bonf),
    }


def selected_support_interval_metrics(
    theta_hat: Array,
    v: Array,
    selected: NDArray[np.int64],
    target: Array,
    n: int,
    alpha: float,
    *,
    gaussian_seed: int,
    calibration_stream_role: str = "selected_support_gaussian_max",
) -> dict[str, Any]:
    selected = np.asarray(selected, dtype=np.int64)
    if selected.size == 0:
        payload = off_c_n_interval_metrics(
            gaussian_seed=int(gaussian_seed),
            calibration_stream_role=calibration_stream_role,
        )
        payload["base_sample_size"] = int(n)
        payload["alpha"] = float(alpha)
        return payload
    v = np.asarray(v, dtype=float)
    if v.shape != (selected.size, selected.size):
        raise ValueError("sandwich dimension does not match selected support")
    diag = np.diag(v)
    if np.any(~np.isfinite(diag)) or np.any(diag <= 0.0):
        raise FloatingPointError("invalid interval variance diagonal")
    cal = gaussian_max_critical(v, alpha, n, gaussian_seed)
    se = np.sqrt(diag / n)
    half = float(cal["critical_value"]) * se
    raw_half = float(cal["raw_critical_value"]) * se
    bonf_half = float(cal["bonferroni_critical_value"]) * se
    est = np.asarray(theta_hat, dtype=float)[selected]
    truth = np.asarray(target, dtype=float)[selected]
    covered = np.abs(est - truth) <= half
    active = np.isin(selected, np.flatnonzero(np.asarray(target) != 0.0))
    ratios = np.divide(
        np.abs(est - truth), half,
        out=np.full_like(half, np.inf), where=half > 0,
    )
    return {
        "inference_method": INFERENCE_METHOD,
        "calibration_stream_role": str(calibration_stream_role),
        "calibration_reportable": True,
        "selected_joint_coverage": bool(np.all(covered)),
        # A nonempty selection containing no truly-active coordinate is a total recovery
        # failure, and the paper counts every selection failure as simultaneous noncoverage,
        # so this is False rather than vacuously True. It also matches the verifier's
        # recomputation; the previous disagreement would have made the verifier flag a
        # legitimate record as corrupt, and the verifier cannot be changed after the formal
        # seeds open. Unreachable on the development grid, but not provably so on 288,000
        # formal records.
        "active_joint_coverage": bool(np.any(active) and np.all(covered[active])),
        "mean_interval_width": float(np.mean(2.0 * half)),
        "mean_raw_gaussian_max_interval_width": float(np.mean(2.0 * raw_half)),
        "mean_bonferroni_interval_width": float(np.mean(2.0 * bonf_half)),
        "max_abs_bias_selected": float(np.max(np.abs(est - truth))),
        "l2_bias_selected": float(np.linalg.norm(est - truth)),
        "base_sample_size": int(n),
        "alpha": float(alpha),
        "coordinate_indices": selected.tolist(),
        "coordinate_estimates": est.astype(float).tolist(),
        "coordinate_targets": truth.astype(float).tolist(),
        "covariance_matrix": v.astype(float).tolist(),
        "variance_diagonal": diag.astype(float).tolist(),
        "standard_errors": se.astype(float).tolist(),
        "interval_lower": (est - half).astype(float).tolist(),
        "interval_upper": (est + half).astype(float).tolist(),
        "coordinate_covered": covered.astype(bool).tolist(),
        "coordinate_active": active.astype(bool).tolist(),
        "standardized_abs_errors": ratios.astype(float).tolist(),
        "off_c_n_totality": {
            "applies": False,
            "standardized_statistic": None,
            "critical_value": None,
            "coverage": None,
            "infinity_encoding": JSON_POSITIVE_INFINITY,
        },
        **cal,
    }
