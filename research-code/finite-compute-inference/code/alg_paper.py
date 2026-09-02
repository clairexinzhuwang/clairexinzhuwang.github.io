"""Kernel-agnostic implementation of the manuscript's Algorithm 1.

FROZEN 2026-08-16.  This module IS the algorithm; the paper's pseudocode is
transcribed from it, and every experiment script calls `run_algorithm1`.

    (1) pilot     theta_pilot = argmin over N_0 := B*t_0 tuples of
                  mean loss + ||theta||^2 / (2 N_0)         (damped Newton + Armijo)
    (2) gain      m_0 fresh tuples -> A_0 at the pilot -> D_0 = S_n(A_0), HELD FIXED
    (3) recursion t = t_0+1..T: theta <- theta - D_0 gbar_t / t   (original clock)
    (4) post-run  m fresh tuples -> A_T, D_T = S_n(A_T), V_T
    (5) data term anchor estimator zeta_k: q_k iid anchors (with replacement),
                  two independent GROUPS per anchor, each the mean of
                  k_per_stream = P conditional tuples containing that anchor;
                  the two groups are centred separately, symmetrized, divided
                  by q_k - 1, and PSD-projected
    (6) Sigma_T = D_T ( sum_k d_k^2 zeta_k / n_k
                        + {t_0 + f_{n,B} (T - t_0)} V_T / (B T^2) ) D_T^T

Sampling.  The pilot tuples, the m_0 initial-Hessian tuples and the m post-run
tuples are ALWAYS drawn with replacement, whatever `scheme` says; `scheme`
governs the recursion minibatches only.  The pilot's N_0 tuples are therefore
i.i.d. and contribute t_0/(B T^2), while each of the T - t_0 minibatches
contributes f_{n,B}/(B T^2), giving the coefficient

    {t_0 + f_{n,B} (T - t_0)} / (B T^2),

which collapses to 1/(BT) only under sampling with replacement, where
f_{n,B} = 1.  The older manuscript factor f (T - t_0)/(B T^2), which drops the
pilot's own sampling variance, is returned as `Sigma_main`/`alg_main` for the
archived audit only: it is NOT the data-only covariance.  The data-only
covariance is `Sigma_dat`.

A kernel is any object with
    n_ref, samples: list of (n_k, d_k)
    draw(k, rng)                 -> tuple batch object for k uniform tuples (scheme WR)
    draw_wor(k, rng)             -> same, without replacement inside the batch
    stats(batch, theta)          -> (mean_loss, mean_score(p), mean_hessian(p,p), scores(k,p))
    anchor_pair(kidx, anchor, rng, theta) -> (Y, Z): the means of two independent groups of
                                    k_per_stream conditional tuples containing `anchor` in sample kidx
                                    (already scaled by gamma_n where the kernel needs it)
    N_tuples                     -> population size N (for f_{n,B} under WOR)
"""
import time

import numpy as np


def stabilized_inverse(M, eps):
    """S_n(M): floor eigenvalues at eps*lambda_max and invert.  S_n(0) = I."""
    w, Q = np.linalg.eigh((M + M.T) / 2)
    if w[-1] <= 0:
        return np.eye(M.shape[0])
    return (Q * (1.0 / np.maximum(w, eps * w[-1]))) @ Q.T


def psd_project(M):
    w, Q = np.linalg.eigh((M + M.T) / 2)
    return (Q * np.maximum(w, 0.0)) @ Q.T, float(w[0]), float(np.linalg.norm(np.minimum(w, 0.0)))


CHUNK = 250_000          # tuples per block; caps peak memory per worker


def stats_chunked(kernel, batch, theta, want_scores=False):
    """kernel.stats over `batch` in blocks of CHUNK tuples, so a 10^7-tuple pilot
    never materialises its full feature/score arrays.  Returns the same
    (loss, score, hessian, scores) tuple; `scores` is None unless requested."""
    n = kernel.batch_len(batch)
    if n <= CHUNK:
        return kernel.stats(batch, theta)
    p = len(theta)
    L = 0.0; g = np.zeros(p); H = np.zeros((p, p)); S = [] if want_scores else None
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n); w = (b - a) / n
        Lb, gb, Hb, Sb = kernel.stats(kernel.batch_slice(batch, a, b), theta)
        L += w * Lb; g += w * gb; H += w * Hb
        if want_scores: S.append(Sb)
    return L, g, H, (np.vstack(S) if want_scores else None)


def pilot_newton(kernel, batch, theta0, N0, max_iter=100, tol_c=1e-3):
    """argmin_theta { mean loss over `batch` + ||theta||^2/(2 N0) } by damped Newton
    with Armijo backtracking.  Returns theta and a diagnostics dict."""
    lam = 1.0 / N0
    th = np.asarray(theta0, float).copy()

    def f_g_H(t):
        L, g, H, _ = stats_chunked(kernel, batch, t)
        return L + lam * (t @ t) / 2, g + lam * t, H + lam * np.eye(len(t))

    f, g, H = f_g_H(th)
    n_bt = 0; conv = False; it = 0
    for it in range(1, max_iter + 1):
        d = np.linalg.solve(H, g)
        a = 1.0
        while True:
            th_new = th - a * d
            f_new, g_new, H_new = f_g_H(th_new)
            if f_new <= f - 1e-4 * a * (g @ d) or a < 1e-10:
                break
            a *= 0.5; n_bt += 1
        th, f, g, H = th_new, f_new, g_new, H_new
        if np.linalg.norm(a * d) < 1e-10 or np.linalg.norm(g) < 1e-12:
            conv = True
            break
    gnorm = float(np.linalg.norm(g))
    return th, {"pilot_iters": it, "pilot_backtracks": n_bt, "pilot_converged": bool(conv),
                "pilot_gradnorm": gnorm, "pilot_tol": tol_c / np.sqrt(N0),
                "pilot_ok": bool(gnorm <= tol_c / np.sqrt(N0))}


def anchor_zeta(kernel, kidx, theta, q, rng, k_per_stream=1):
    """Anchor estimator for zeta_k: q iid uniform anchors with replacement, two
    independent conditional tuples per anchor, separate centring, sym, /(q-1),
    PSD projection.  Returns (zeta_hat, diagnostics)."""
    n_k, _ = kernel.samples[kidx]
    anchors = rng.integers(0, n_k, size=q)
    # chunked so the (q, 2k, p) partner arrays stay small at n = 1e6
    blk = max(1, 20000 // max(1, k_per_stream // 4))
    Ys = []; Zs = []
    for s0 in range(0, q, blk):
        Yb, Zb = kernel.anchor_streams(kidx, anchors[s0:s0 + blk], rng, theta, k_per_stream)
        Ys.append(Yb); Zs.append(Zb)
    Y = np.vstack(Ys); Z = np.vstack(Zs)
    Yc = Y - Y.mean(0); Zc = Z - Z.mean(0)
    C = (Yc.T @ Zc) / (q - 1)
    C = (C + C.T) / 2
    Zeta, min_eig, moved = psd_project(C)
    return Zeta, {"zeta_min_eig_pre": min_eig, "zeta_proj_moved": moved,
                  "zeta_projected": bool(min_eig < 0)}


#: the phases whose wall-clock `run_algorithm1` reports
PHASES = ("pilot", "hessian0", "recursion", "terminal", "anchors", "total")

#: the components that get their own generator when `streams` is used
STREAM_NAMES = ("pilot", "hessian0", "recursion", "terminal", "anchors")


def component_streams(seed):
    """One independent generator per sampling component, from one seed.

    Two runs given the same seed draw the same pilot tuples, the same initial
    Hessian tuples, the same recursion tuple stream, the same terminal tuples
    and the same anchors and partners, whatever the minibatch size B is,
    because no component's state depends on how another component was
    consumed.  That is what makes a comparison across B a paired one.
    """
    kids = np.random.SeedSequence(seed).spawn(len(STREAM_NAMES))
    return {nm: np.random.default_rng(k) for nm, k in zip(STREAM_NAMES, kids)}


def run_algorithm1(kernel, theta0, B, T, t0, m0, m, q, scheme="WR", eps=None,
                   rng=None, want_zeta=True, k_per_stream=1, trace=False,
                   streams=None, stream_len=None):
    """One run of Algorithm 1.  Returns dict with theta_T, Sigma_T (both variance
    factors), the pieces, and all diagnostics.

    `streams` is an optional dict of per-component generators, as returned by
    `component_streams`.  It changes which tuples are drawn, never what is done
    with them: the estimator, the arithmetic and the order of operations are
    identical either way.  Left at None the function draws everything from the
    single `rng`, exactly as the archived results were produced.

    Under `streams` and scheme "WR" the recursion's B(T - t_0) tuples are drawn
    as one ordered stream and cut into consecutive blocks of B, so runs that
    differ only in B consume the same tuples in the same order and differ only
    in how they are grouped.  Under "WOR" the minibatch is drawn per step,
    because drawing without replacement is a property of the batch and a batch
    of size B is not a sub-block of a longer one.
    """
    t_phase = {}
    _clock = time.perf_counter
    _t_run = _clock()
    rng = np.random.default_rng() if rng is None else rng
    rs = streams if streams is not None else {nm: rng for nm in STREAM_NAMES}
    p = len(theta0)
    eps = 1.0 / kernel.n_ref if eps is None else eps
    # The pilot tuples, the m0 Hessian tuples and the m post-run tuples are drawn
    # independently and uniformly (i.i.d., i.e. with replacement) as in the
    # manuscript; only the recursion's minibatches follow `scheme`.
    draw = kernel.draw
    draw_mb = kernel.draw if scheme == "WR" else kernel.draw_wor
    assert 1 <= t0 < T, (t0, T)
    N0 = B * t0

    # (1) pilot on the first t0 minibatches' worth of tuples, always with
    #     replacement whatever `scheme` says
    _t = _clock()
    pilot_batch = draw(N0, rs["pilot"])
    theta, diag = pilot_newton(kernel, pilot_batch, theta0, N0)
    diag["N0"] = N0; diag["t0"] = t0
    t_phase["pilot"] = _clock() - _t

    # (2) fixed gain from m0 fresh tuples at the pilot, with replacement
    _t = _clock()
    _, _, A0, _ = stats_chunked(kernel, draw(m0, rs["hessian0"]), theta)
    w0 = np.linalg.eigvalsh((A0 + A0.T) / 2)
    D0 = stabilized_inverse(A0, eps)
    diag["A0_kappa"] = float(w0[-1] / max(w0[0], 1e-300)); diag["A0_floor_active"] = bool(w0[0] < eps * w0[-1])
    t_phase["hessian0"] = _clock() - _t

    # (3) main recursion on the ORIGINAL clock.  `trace` only records the
    # iterate; it draws nothing extra and changes no arithmetic, so a traced
    # run and an untraced one with the same seed return identical results.
    path = [theta.copy()] if trace else None
    # score_mean repeats stats()'s score arithmetic exactly (same draws, same
    # float operations in the same order) and skips the loss, Hessian and
    # per-tuple scores the loop discards; end-to-end outputs are bit-identical
    # (verified over kernels x B x schemes x seeds) and the long-B=1 runs are
    # what it exists for.
    step_score = getattr(kernel, "score_mean", None)
    if step_score is None:
        step_score = lambda batch, th: kernel.stats(batch, th)[1]
    _t = _clock()
    r_rec = rs["recursion"]
    if streams is not None and scheme == "WR":
        # One ordered stream, cut into consecutive blocks of B.  `stream_len`
        # lets several minibatch sizes share a stream when their budgets differ
        # only by the integer part of the schedule: the longest is drawn and the
        # others consume prefixes of it, so the tuples and their order are
        # common to all of them.
        n_draw = stream_len if stream_len is not None else B * (T - t0)
        stream = draw(n_draw, r_rec)
        take = lambda i: kernel.batch_slice(stream, i * B, (i + 1) * B)
    else:
        take = lambda i: draw_mb(B, r_rec)
    for i, t in enumerate(range(t0 + 1, T + 1)):
        gb = step_score(take(i), theta)
        theta = theta - (D0 @ gb) / t
        if trace:
            path.append(theta.copy())
    t_phase["recursion"] = _clock() - _t

    # (4) post-run Hessian, gain, tuple-score covariance from m fresh tuples,
    #     with replacement
    _t = _clock()
    _, _, AT, ST = kernel.stats(draw(m, rs["terminal"]), theta)
    wT = np.linalg.eigvalsh((AT + AT.T) / 2)
    DT = stabilized_inverse(AT, eps)
    VT = np.cov(ST, rowvar=False, bias=False) if m > 1 else np.zeros((p, p))
    diag["AT_kappa"] = float(wT[-1] / max(wT[0], 1e-300)); diag["AT_floor_active"] = bool(wT[0] < eps * wT[-1])
    t_phase["terminal"] = _clock() - _t
    if trace:
        diag["path"] = np.asarray(path)

    # (5) anchor estimator of the first-projection covariances
    data_term = np.zeros((p, p))
    _t = _clock()
    if want_zeta:
        for kidx, (n_k, d_k) in enumerate(kernel.samples):
            qk = q if np.isscalar(q) else q[kidx]
            Zk, dz = anchor_zeta(kernel, kidx, theta, int(qk), rs["anchors"], k_per_stream)
            for key, val in dz.items():
                diag[f"{key}_k{kidx}"] = val
            data_term += (d_k ** 2) * Zk / n_k
    t_phase["anchors"] = _clock() - _t

    # (6) covariance.  Full factor (t0 + f (T-t0))/(B T^2): the pilot's tuples
    # are i.i.d. (correction 1), the T-t0 minibatches carry f_{n,B}; under WR
    # this is exactly 1/(BT).  The older manuscript factor (T-t0)/(BT^2) is
    # returned for the audit.
    N = kernel.N_tuples
    f = 1.0 if scheme == "WR" else (N - B) / (N - 1)
    alg_full = (t0 + f * (T - t0)) * VT / (B * T * T)
    alg_main = f * (T - t0) * VT / (B * T * T)
    Sigma_full = DT @ (data_term + alg_full) @ DT.T
    Sigma_main = DT @ (data_term + alg_main) @ DT.T
    Sigma_dat = DT @ data_term @ DT.T
    t_phase["total"] = _clock() - _t_run
    diag["t_phase"] = {k: float(t_phase.get(k, 0.0)) for k in PHASES}
    return {"theta": theta, "Sigma_full": Sigma_full, "Sigma_main": Sigma_main,
            "Sigma_dat": Sigma_dat, "alg_full": DT @ alg_full @ DT.T,
            "alg_main": DT @ alg_main @ DT.T, "D0": D0, "DT": DT, "VT": VT, **diag}


# ---------------------------------------------------------------------------
# defaults (frozen)
# ---------------------------------------------------------------------------
def legacy_default_t0(T, rule="log", n=None):
    """The pilot rule of the archived growing-B grid.  KEPT ONLY to reproduce
    those results; the fixed-B study must not call it.

    Fixed kernels: t0 = 2 ceil(log2 T).  Rank ('rank' rule): ceil(n^0.60).
    Both are capped at T/2 so that a main phase exists, and that cap is the
    reason this rule is not used again: where it binds, the pilot budget is set
    by the horizon rather than by the sample size, so t_0 is no longer a
    function of n alone and is not comparable across B.  The fixed-B study sets
    the pilot tuple budget directly; see experiment_design.pilot_tuple_budget.
    """
    if rule == "rank":
        t0 = int(np.ceil(n ** 0.60))
    else:
        t0 = int(2 * np.ceil(np.log2(max(T, 2))))
    return int(min(max(t0, 1), max(T // 2, 1)))


#: archived name; new code must call `legacy_default_t0` knowingly
default_t0 = legacy_default_t0


def default_m(B, n_ref, h_eff=1.0):
    """m0 = m = ceil(max(4B, n^0.6, 8192, 1500/h_eff)).

    n^0.6 makes m -> inf as the theorems require.  For the smoothed rank
    kernel the deterministic terminal-Hessian condition is m h_{0,n} -> inf,
    with h_{0,n} the deterministic bandwidth sequence (equivalently, in
    probability, with the random bandwidth) -- it is the number of pairs the
    kernel actually weights, m h_{0,n}, that must diverge, not m h_n^2; it never binds on the grids.  The two constants are
    finite-sample floors for the Hessian sample: the plug-in covariance uses
    D_T = S_n(A_T)^{-1}, and Jensen inflates E[A_T^{-1}] when A_T is noisy.
    Measured: 8192 tuples keep the inflation <= 1% for the logistic kernels
    (10x10 triplet Hessian: +12% at 512, +0.6% at 8192); for smoothed rank the
    curvature lives on the fraction h_n of pairs, so the effective sample is
    m h_n and 1500 effective pairs keep it <= 0.4% (+19% at m h_n = 100).
    h_eff = 1 for the fixed kernels, h_n for rank.  Cost is O(n^0.26) tuples,
    negligible against the B T = n log n training budget."""
    return int(np.ceil(max(4 * B, n_ref ** 0.6, 8192, 1500.0 / h_eff)))
