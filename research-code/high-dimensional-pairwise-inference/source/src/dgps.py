"""Data-generating mechanisms and population targets for the formal grid."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from functools import lru_cache

import numpy as np
from scipy import optimize, signal, stats
from scipy.special import expit


DESIGNS = ("identity", "toeplitz", "tridiag")
LAW_NAMES = {
    1: "normal",
    2: "t4_scaled",
    3: "gamma_asymmetric",
    4: "t3",
    5: "cauchy",
    6: "contaminated_normal",
}
LAW_LABELS = {
    1: "N(0,1)",
    2: "t4/sqrt2",
    3: "(Gamma(4,1)-2)/2",
    4: "t3",
    5: "Cauchy",
    6: ".95N+.05N(0,100^2)",
}
AUC_CONDITIONAL_GAUSSIAN_DGP = "auc-conditional-gaussian-domain-separated-v1"
AUC_DGP_STREAM_ROLES = (
    "dgp.auc.support",
    "dgp.auc.latent_index",
    "dgp.auc.error",
    "dgp.auc.covariate_residual",
)
# Inherited identity-design signal norms used by the earlier calibrated grid.
AUC_CALIB_NU = {
    1: 5.357101324446773,
    2: 7.316502949807255,
    3: 5.750380844747082,
    4: 8.430183889272627,
    5: 30.17439092426487,
    6: 9.713383688148989,
}
Q75 = float(stats.norm.ppf(0.75))
_TRAPEZOID = getattr(np, "trapezoid", np.trapz)


def _gamma_median_mad() -> tuple[float, float]:
    med = float((stats.gamma.ppf(0.5, a=4.0) - 2.0) / 2.0)
    cdf = lambda x: stats.gamma.cdf(2.0 * x + 2.0, a=4.0)
    mad = optimize.brentq(lambda d: cdf(med + d) - cdf(med - d) - 0.5, 1e-10, 20.0)
    return med, float(mad / Q75)


def _mixture_scale() -> float:
    mad = optimize.brentq(
        lambda d: 0.95 * (2.0 * stats.norm.cdf(d) - 1.0)
        + 0.05 * (2.0 * stats.norm.cdf(d / 100.0) - 1.0) - 0.5,
        1e-10,
        100.0,
    )
    return float(mad / Q75)


_GAMMA_MED, _GAMMA_SCALE = _gamma_median_mad()
_AUC_STANDARDIZATION = {
    1: (0.0, 1.0),
    2: (0.0, float(stats.t.ppf(0.75, df=4) / (math.sqrt(2.0) * Q75))),
    3: (_GAMMA_MED, _GAMMA_SCALE),
    4: (0.0, float(stats.t.ppf(0.75, df=3) / Q75)),
    5: (0.0, float(1.0 / Q75)),
    6: (0.0, _mixture_scale()),
}


def draw_raw_error(law: int, n: int, rng: np.random.Generator) -> np.ndarray:
    if law == 1:
        return rng.standard_normal(n)
    if law == 2:
        return rng.standard_t(4, n) / math.sqrt(2.0)
    if law == 3:
        return (rng.gamma(4.0, 1.0, n) - 2.0) / 2.0
    if law == 4:
        return rng.standard_t(3, n)
    if law == 5:
        return rng.standard_cauchy(n)
    if law == 6:
        clean = rng.random(n) < 0.95
        out = rng.standard_normal(n)
        out[~clean] = 100.0 * rng.standard_normal(np.count_nonzero(~clean))
        return out
    raise ValueError(f"unknown law {law}")


def draw_auc_error(law: int, n: int, rng: np.random.Generator) -> np.ndarray:
    med, scale = _AUC_STANDARDIZATION[law]
    return (draw_raw_error(law, n, rng) - med) / scale


def raw_error_cdf(law: int, x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if law == 1:
        return stats.norm.cdf(x)
    if law == 2:
        return stats.t.cdf(x * math.sqrt(2.0), df=4)
    if law == 3:
        return stats.gamma.cdf(2.0 * x + 2.0, a=4.0)
    if law == 4:
        return stats.t.cdf(x, df=3)
    if law == 5:
        return stats.cauchy.cdf(x)
    if law == 6:
        return 0.95 * stats.norm.cdf(x) + 0.05 * stats.norm.cdf(x / 100.0)
    raise ValueError(f"unknown law {law}")


def auc_class_probability(law: int, q: np.ndarray | float) -> np.ndarray:
    """P(Y=1|Q=q) for Y=1{q+epsilon_std>0}."""
    med, scale = _AUC_STANDARDIZATION[law]
    return 1.0 - raw_error_cdf(law, med - scale * np.asarray(q, dtype=float))


def make_covariates(
    rng: np.random.Generator, n: int, p: int, design: str
) -> np.ndarray:
    z = rng.standard_normal((n, p))
    if design == "identity":
        return z
    if design == "toeplitz":
        rho = 0.5
        x = np.empty_like(z)
        x[:, 0] = z[:, 0]
        scale = math.sqrt(1.0 - rho * rho)
        for j in range(1, p):
            x[:, j] = rho * x[:, j - 1] + scale * z[:, j]
        return x
    if design == "tridiag":
        # Bidiagonal Cholesky of diag(1), offdiag(0.48).
        x = np.empty_like(z)
        diag_prev = 1.0
        x[:, 0] = z[:, 0]
        for j in range(1, p):
            sub = 0.48 / diag_prev
            diag = math.sqrt(1.0 - sub * sub)
            x[:, j] = sub * z[:, j - 1] + diag * z[:, j]
            diag_prev = diag
        return x
    raise ValueError(f"unknown design {design}")


def covariance_submatrix(indices: np.ndarray, design: str) -> np.ndarray:
    idx = np.asarray(indices, dtype=int)
    d = np.abs(idx[:, None] - idx[None, :])
    if design == "identity":
        return np.eye(idx.size)
    if design == "toeplitz":
        return 0.5 ** d
    if design == "tridiag":
        out = np.eye(idx.size)
        out[d == 1] = 0.48
        return out
    raise ValueError(design)


def covariance_times_sparse(
    p: int, support: np.ndarray, values: np.ndarray, design: str
) -> np.ndarray:
    support = np.asarray(support, dtype=int)
    values = np.asarray(values, dtype=float)
    out = np.zeros(p)
    if design == "identity":
        out[support] = values
        return out
    if design == "toeplitz":
        grid = np.arange(p)[:, None]
        out = (0.5 ** np.abs(grid - support[None, :])) @ values
        return out
    if design == "tridiag":
        out[support] += values
        left = support - 1
        right = support + 1
        mask = left >= 0
        np.add.at(out, left[mask], 0.48 * values[mask])
        mask = right < p
        np.add.at(out, right[mask], 0.48 * values[mask])
        return out
    raise ValueError(design)


def random_support(rng: np.random.Generator, p: int, s: int) -> np.ndarray:
    return np.sort(rng.choice(p, size=s, replace=False))


def auc_dgp_child_seeds(seed: int) -> dict[str, int]:
    """Derive the exact domain-separated AUC child streams from one data seed."""
    values = np.random.default_rng(int(seed)).integers(
        0, np.iinfo(np.int64).max,
        size=len(AUC_DGP_STREAM_ROLES), dtype=np.int64,
    )
    result = {
        role: int(value) for role, value in zip(AUC_DGP_STREAM_ROLES, values)
    }
    if len(set(result.values())) != len(result):
        raise RuntimeError("AUC DGP child streams collided")
    return result


def replay_auc_latent_identity(
    seed: int, n: int, p: int, s: int, design: str, law: int,
    signal_scale: float = 1.0,
) -> dict:
    """Replay exact AUC support/latent/error/labels without an n-by-p matrix."""
    signal_scale = float(signal_scale)
    if not math.isfinite(signal_scale) or signal_scale <= 0.0:
        raise ValueError("signal_scale must be positive and finite")
    streams = auc_dgp_child_seeds(int(seed))
    support = random_support(
        np.random.default_rng(streams["dgp.auc.support"]), int(p), int(s),
    )
    active = signal_scale * math.sqrt(AUC_CALIB_NU[int(law)] / int(s))
    active_values = np.full(int(s), active, dtype=float)
    sigma_s = covariance_submatrix(support, str(design))
    nu = float(active_values @ sigma_s @ active_values)
    q = (
        math.sqrt(nu)
        * np.random.default_rng(streams["dgp.auc.latent_index"]).standard_normal(int(n))
    )
    error = draw_auc_error(
        int(law), int(n), np.random.default_rng(streams["dgp.auc.error"]),
    )
    labels = np.asarray(q + error > 0.0, dtype=np.uint8)
    q_le = np.asarray(q, dtype="<f8")
    positives = int(np.count_nonzero(labels))
    return {
        "algorithm": AUC_CONDITIONAL_GAUSSIAN_DGP,
        "stream_roles": list(AUC_DGP_STREAM_ROLES),
        "child_streams": [
            {"stream_name": role, "seed": streams[role], "consumed": True}
            for role in AUC_DGP_STREAM_ROLES
        ],
        "support": support,
        "active": active,
        "nu": nu,
        "latent_index": q,
        "latent_index_dtype": "<f8",
        "latent_index_order": "C",
        "latent_index_length": int(n),
        "latent_index_sha256": hashlib.sha256(q_le.tobytes(order="C")).hexdigest(),
        "labels": labels,
        "binary_labels_dtype": "uint8",
        "binary_labels_order": "C",
        "binary_labels_length": int(n),
        "binary_labels_sha256": hashlib.sha256(labels.tobytes(order="C")).hexdigest(),
        "n_positive": positives,
        "n_negative": int(labels.size - positives),
    }


@lru_cache(maxsize=4096)
def _auc_target_fft(
    law: int,
    nu_rounded: float,
    grid_points: int,
    tail_sd: float,
) -> tuple[float, float, float]:
    """Compute the pairwise-logistic population target by one-dimensional FFT.

    If Q is N(0,nu), the positive and negative class densities are proportional
    to phi_nu(q)m(q) and phi_nu(q){1-m(q)}.  Their convolution gives the density
    of Delta=Q^+-Q^-, reducing the score equation to one numerical integral.
    The grid and tail cutoff are deterministic and are validated at a second
    resolution by :func:`auc_target_scale`.
    """
    if law not in LAW_NAMES:
        raise ValueError(f"unknown law {law}")
    nu = float(nu_rounded)
    if not (nu > 0.0 and math.isfinite(nu)):
        raise ValueError("nu must be positive and finite")
    if grid_points < 4096 or grid_points % 2:
        raise ValueError("grid_points must be an even integer at least 4096")
    if tail_sd < 6.0:
        raise ValueError("tail_sd must be at least 6")

    sd = math.sqrt(nu)
    q = np.linspace(-tail_sd * sd, tail_sd * sd, grid_points)
    dq = float(q[1] - q[0])
    base = np.exp(-0.5 * q * q / nu) / math.sqrt(2.0 * math.pi * nu)
    pplus = auc_class_probability(law, q)
    fplus = base * pplus
    fminus = base * (1.0 - pplus)
    pi = float(_TRAPEZOID(fplus, q))
    if not 0.0 < pi < 1.0:
        raise RuntimeError("invalid population class probability")
    fplus /= pi
    fminus /= 1.0 - pi

    # f_Delta(delta)=int f_+(q)f_-(q-delta)dq.  Tiny negative FFT
    # roundoff is removed before the density is renormalized.
    fdelta = signal.fftconvolve(fplus, fminus[::-1], mode="full") * dq
    fdelta = np.maximum(fdelta, 0.0)
    delta = np.linspace(-2.0 * tail_sd * sd, 2.0 * tail_sd * sd, fdelta.size)
    normalizer = float(_TRAPEZOID(fdelta, delta))
    if not normalizer > 0.0:
        raise RuntimeError("failed to normalize the Delta density")
    fdelta /= normalizer

    def equation(t: float) -> float:
        return float(_TRAPEZOID(delta * expit(-t * delta) * fdelta, delta))

    lo, hi = 0.0, 1.0
    while equation(hi) > 0.0 and hi < 1024.0:
        hi *= 2.0
    if hi >= 1024.0:
        raise RuntimeError("failed to bracket the AUC target scale")
    tstar = float(optimize.brentq(equation, lo, hi, xtol=1e-12, rtol=1e-12))
    return tstar, pi, abs(equation(tstar))


@lru_cache(maxsize=4096)
def auc_target_scale(
    law: int,
    nu_rounded: float,
) -> tuple[float, float, float, float, float]:
    """Return a deterministic AUC surrogate target and numerical diagnostics.

    Returns ``(t_star, prevalence, root_residual, t_resolution_sensitivity,
    prevalence_resolution_sensitivity)``.  The sensitivity values compare the
    primary FFT calculation with both a half-resolution grid and a shorter
    Gaussian-tail cutoff.  They are numerical convergence diagnostics, not
    probabilistic standard errors.
    """
    primary = _auc_target_fft(law, nu_rounded, 32768, 8.0)
    coarse = _auc_target_fft(law, nu_rounded, 16384, 8.0)
    shorter = _auc_target_fft(law, nu_rounded, 32768, 7.0)
    t_sensitivity = max(abs(primary[0] - coarse[0]), abs(primary[0] - shorter[0]))
    pi_sensitivity = max(abs(primary[1] - coarse[1]), abs(primary[1] - shorter[1]))
    return primary[0], primary[1], primary[2], t_sensitivity, pi_sensitivity


@dataclass(frozen=True)
class GeneratedData:
    X: np.ndarray
    y: np.ndarray
    theta_target: np.ndarray
    generating_direction: np.ndarray
    support: np.ndarray
    metadata: dict
    dgp_certificate: dict | None = None


def generate_rank(
    seed: int, n: int, p: int, s: int, design: str, law: int,
    beta: float = math.sqrt(3.0),
) -> GeneratedData:
    rng = np.random.default_rng(seed)
    support = random_support(rng, p, s)
    theta = np.zeros(p)
    theta[support] = beta
    X = make_covariates(rng, n, p, design)
    y = X @ theta + draw_raw_error(law, n, rng)
    return GeneratedData(
        X=X, y=y, theta_target=theta.copy(), generating_direction=theta.copy(), support=support,
        metadata={"law": law, "law_label": LAW_LABELS[law], "design": design, "beta": beta},
    )


def generate_auc(
    seed: int, n: int, p: int, s: int, design: str, law: int,
    signal_scale: float = 1.0,
) -> GeneratedData:
    """Generate the calibrated AUC instance.

    ``signal_scale`` multiplies the frozen latent-index coefficient magnitude.
    The default preserves the frozen Gaussian joint distribution, target,
    cells, and thresholds.  The domain-separated conditional construction is
    a new deterministic seed mapping, intentionally introduced before any
    admissible current-source evidence so its labels can be replayed in O(n).
    Nondefault values are development-only boundary-panel inputs.
    """
    signal_scale = float(signal_scale)
    if not math.isfinite(signal_scale) or signal_scale <= 0.0:
        raise ValueError("signal_scale must be positive and finite")
    identity = replay_auc_latent_identity(
        seed, n, p, s, design, law, signal_scale=signal_scale,
    )
    support = np.asarray(identity["support"], dtype=np.int64)
    bar_theta = np.zeros(p)
    bar_theta[support] = float(identity["active"])
    residual_seed = next(
        item["seed"] for item in identity["child_streams"]
        if item["stream_name"] == "dgp.auc.covariate_residual"
    )
    Z = make_covariates(
        np.random.default_rng(int(residual_seed)), n, p, design,
    )
    q = np.asarray(identity["latent_index"], dtype=float)
    sigma_b = covariance_times_sparse(p, support, bar_theta[support], design)
    nu = float(identity["nu"])
    X = Z + (q - Z @ bar_theta)[:, None] * (sigma_b / nu)[None, :]
    projection_residual = float(np.max(np.abs(X @ bar_theta - q)))
    projection_tolerance = float(1e-10 * max(1.0, np.max(np.abs(q))))
    if not math.isfinite(projection_residual) or projection_residual > projection_tolerance:
        raise FloatingPointError(
            "conditional-Gaussian AUC covariates violate their latent-index projection"
        )
    y = np.asarray(identity["labels"], dtype=int)
    sigma_s = covariance_submatrix(support, design)
    nu = float(bar_theta[support] @ sigma_s @ bar_theta[support])
    tstar, pi, residual, t_sensitivity, pi_sensitivity = auc_target_scale(
        law, round(nu, 12)
    )
    theta_target = tstar * bar_theta
    metadata = {
        "law": law, "law_label": LAW_LABELS[law], "design": design,
        "nu": nu,
        "t_star": tstar,
        "target_root_residual": residual,
        "target_t_resolution_sensitivity": t_sensitivity,
        "target_prevalence_resolution_sensitivity": pi_sensitivity,
        "target_numerical_method": "FFT convolution; 32768 grid points; 8 Gaussian SD cutoff",
        "population_prevalence": pi,
        "realized_prevalence": float(np.mean(y)),
    }
    # Preserve the default metadata byte-for-byte.  The development-only scale
    # is recorded only when the caller deliberately leaves the formal path.
    if signal_scale != 1.0:
        metadata["signal_scale"] = signal_scale
    return GeneratedData(
        X=X, y=y, theta_target=theta_target, generating_direction=bar_theta,
        support=support,
        metadata=metadata,
        dgp_certificate={
            key: value for key, value in identity.items()
            if key not in {"support", "active", "latent_index", "labels"}
        } | {
            "support": support.tolist(),
            "latent_projection_max_abs_residual": projection_residual,
            "latent_projection_tolerance": projection_tolerance,
        },
    )


def population_screen_geometry(data: GeneratedData, problem: str) -> dict:
    p = data.theta_target.size
    if problem == "rank":
        vector = covariance_times_sparse(
            p, data.support, data.theta_target[data.support], data.metadata["design"]
        )
    elif problem == "auc":
        vector = covariance_times_sparse(
            p, data.support, data.generating_direction[data.support], data.metadata["design"]
        )
    else:
        raise ValueError(problem)
    return {"direction": vector}
