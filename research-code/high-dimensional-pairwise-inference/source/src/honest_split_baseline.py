"""Honest 50/50 split comparator for the pairwise HTP pipeline.

Selection (discovery, cleaning, and pruning) uses only the selection half.
The final restricted estimator and sandwich use only the independent inference
half.  This module is a comparator, not the proposed full-data method.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from .stochastic_screen_sgd import (
    Array, Problem, _child_stream_audit, _empty_final_pair_accounting,
    _final_pair_accounting, _new_final_pair_union_tracker,
    fit_full_data_htp_routes, sampled_hajek_sandwich,
    stratified_half_split, warm_preconditioned_pair_sgd,
)


@dataclass
class HonestSplitRoutesResult:
    selection_indices: NDArray[np.int64]
    inference_indices: NDArray[np.int64]
    candidate: NDArray[np.int64]
    known_selected: NDArray[np.int64]
    unknown_selected: NDArray[np.int64]
    known_final_theta: Array
    known_final_V: Array
    unknown_final_theta: Array
    unknown_final_V: Array
    diagnostics: dict


def _inference_fit(
    X: Array,
    y: Array,
    problem: Problem,
    selected: NDArray[np.int64],
    x0_full: Array,
    rng: np.random.Generator,
    *,
    bandwidth: float,
    radius: float,
    eta0: float,
    exponent: float,
    stream_role: str,
) -> tuple[Array, Array, dict, dict]:
    p = X.shape[1]
    full = np.zeros(p)
    selected = np.asarray(selected, dtype=np.int64)
    if selected.size == 0:
        empty_accounting = _empty_final_pair_accounting(
            f"{stream_role}.optimizer_and_sandwich"
        )
        return full, np.zeros((0, 0)), {
            "selection_empty": True, "finite": False,
            "final_pair_accounting": empty_accounting,
        }, {"final_pair_accounting": empty_accounting}
    Xs = np.ascontiguousarray(X[:, selected])
    pair_union_tracker = _new_final_pair_union_tracker(
        X.shape[0], selected.size, f"{stream_role}.optimizer_and_sandwich"
    )
    fit = warm_preconditioned_pair_sgd(
        Xs, y, problem, rng, bandwidth=bandwidth, radius=radius,
        eta0=eta0, exponent=exponent, x0=np.asarray(x0_full[selected], dtype=float),
        stream_role=f"{stream_role}.optimizer",
        pair_union_tracker=pair_union_tracker,
    )
    _, _, V, sand = sampled_hajek_sandwich(
        Xs, y, fit.theta, problem, rng, bandwidth=bandwidth,
        stream_role=f"{stream_role}.sandwich",
        pair_union_tracker=pair_union_tracker,
    )
    final_accounting = _final_pair_accounting(pair_union_tracker, fit.diagnostics, sand)
    fit.diagnostics["final_pair_accounting"] = final_accounting
    sand["final_pair_accounting"] = final_accounting
    full[selected] = fit.theta
    return full, V, fit.diagnostics, sand


def fit_honest_split_htp_routes(
    X: Array,
    y: Array,
    problem: Problem,
    K: int,
    known_s: int,
    rng: np.random.Generator,
    *,
    radius_buffer: float,
    bandwidth: float = 1.0,
    radius: float = 20.0,
    eta0: float = 1.0,
    exponent: float = 2.0 / 3.0,
) -> HonestSplitRoutesResult:
    """Fit the same selector on one half and infer on the other half."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    seeds = rng.integers(0, np.iinfo(np.int64).max, size=4, dtype=np.int64)
    split_rng = np.random.default_rng(int(seeds[0]))
    selection_rng = np.random.default_rng(int(seeds[1]))
    known_rng = np.random.default_rng(int(seeds[2]))
    unknown_rng = np.random.default_rng(int(seeds[3]))
    idx_sel, idx_inf = stratified_half_split(y, problem, split_rng)

    selection = fit_full_data_htp_routes(
        np.ascontiguousarray(X[idx_sel]), y[idx_sel], problem, K, known_s,
        selection_rng, bandwidth=bandwidth, radius=radius,
        radius_buffer=radius_buffer, eta0=eta0, exponent=exponent,
    )
    known_theta, known_V, known_fit, known_sand = _inference_fit(
        np.ascontiguousarray(X[idx_inf]), y[idx_inf], problem,
        selection.known_selected, selection.clean_theta, known_rng,
        bandwidth=bandwidth, radius=radius, eta0=eta0, exponent=exponent,
        stream_role="honest_split.known_inference_final",
    )
    if np.array_equal(selection.unknown_selected, selection.known_selected):
        unknown_theta = known_theta.copy()
        unknown_V = known_V.copy()
        unknown_fit = dict(known_fit)
        unknown_sand = dict(known_sand)
        shared_final = True
    else:
        unknown_theta, unknown_V, unknown_fit, unknown_sand = _inference_fit(
            np.ascontiguousarray(X[idx_inf]), y[idx_inf], problem,
            selection.unknown_selected, selection.clean_theta, unknown_rng,
            bandwidth=bandwidth, radius=radius, eta0=eta0, exponent=exponent,
            stream_role="honest_split.unknown_inference_final",
        )
        shared_final = False

    return HonestSplitRoutesResult(
        idx_sel, idx_inf, selection.candidate,
        selection.known_selected, selection.unknown_selected,
        known_theta, known_V, unknown_theta, unknown_V,
        {
            "observation_split": True,
            "selection_base_units": int(idx_sel.size),
            "inference_base_units": int(idx_inf.size),
            "selection_indices": idx_sel.tolist(),
            "inference_indices": idx_inf.tolist(),
            "selection_pipeline": selection.diagnostics,
            "known_final_sgd": known_fit,
            "known_final_sandwich": known_sand,
            "unknown_final_sgd": unknown_fit,
            "unknown_final_sandwich": unknown_sand,
            "shared_final_fit": bool(shared_final),
            **_child_stream_audit(
                "honest_split",
                ("split", "selection_pipeline", "known_inference_final", "unknown_inference_final"),
                seeds,
                consumed=(
                    True,
                    True,
                    bool(selection.known_selected.size),
                    bool((not shared_final) and selection.unknown_selected.size),
                ),
            ),
        },
    )
