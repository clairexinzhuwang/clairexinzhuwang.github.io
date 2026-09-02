"""Data-generating processes and population targets for the simulations.

These are the settings the paper reports: the two-sample AUC design and its
Monte Carlo target, the anisotropic class means and sampler for triplet metric
learning, and the symmetric-vector index helpers.  They are collected here so
that the package contains the estimation algorithm exactly once, in
alg_paper.py, with nothing else that could be mistaken for it.
"""
import numpy as np
from scipy.special import expit
from scipy.stats import norm


# ---- two-sample AUC design ----

def make_dgp(p, seed=12345, mu_scale=1.1):
    """Two Gaussian classes with DIFFERENT means AND different covariances, so
    that Sig+ != Sig- (forces the two-piece term to genuinely matter)."""
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.standard_normal((p, p)))[0]
    mu_pos = +0.5 * mu_scale * Q[:, 0]
    mu_neg = -0.5 * mu_scale * Q[:, 0]
    # anisotropic, class-dependent covariances
    dpos = np.linspace(0.7, 1.4, p)
    dneg = np.linspace(1.5, 0.6, p)
    Sig_pos = (Q * dpos) @ Q.T
    Sig_neg = (Q * dneg) @ Q.T
    L_pos = np.linalg.cholesky(Sig_pos)
    L_neg = np.linalg.cholesky(Sig_neg)
    return mu_pos, mu_neg, L_pos, L_neg


def draw(m, n_neg, mu_pos, mu_neg, L_pos, L_neg, rng):
    p = mu_pos.shape[0]
    Xpos = mu_pos + rng.standard_normal((m, p)) @ L_pos.T
    Xneg = mu_neg + rng.standard_normal((n_neg, p)) @ L_neg.T
    return Xpos, Xneg


def compute_theta_star_auc(mu_pos, mu_neg, L_pos, L_neg, M=int(2e8),
                           chunk=4_000_000, seed=7):
    """Population minimizer of E[softplus(theta^T(X- - X+))] by streaming Newton
    on fresh Gaussian pairs (convex => unique root, no MC-noise floor)."""
    p = mu_pos.shape[0]
    rng = np.random.default_rng(seed)

    def accum(beta, Mt, rg):
        g = np.zeros(p); H = np.zeros((p, p)); Vs = np.zeros((p, p)); done = 0
        while done < Mt:
            b = min(chunk, Mt - done)
            Xp = mu_pos + rg.standard_normal((b, p)) @ L_pos.T
            Xn = mu_neg + rg.standard_normal((b, p)) @ L_neg.T
            a = Xn - Xp
            u = a @ beta; sig = expit(u); sd = sig * (1 - sig)
            gi = sig[:, None] * a
            g += gi.sum(0); H += (sd[:, None] * a).T @ a; Vs += gi.T @ gi
            done += b
        return g / Mt, H / Mt, Vs / Mt

    beta = np.zeros(p)
    for it in range(40):                      # stage 1: locate (convex, fast)
        g, H, _ = accum(beta, min(2_000_000, M), np.random.default_rng(seed + it))
        step = np.linalg.solve(H + 1e-12 * np.eye(p), g)
        beta -= step
        if np.linalg.norm(step) < 1e-9:
            break
    g2, H2, Vs = accum(beta, M, np.random.default_rng(seed + 999))  # stage 2: refine
    Hi = np.linalg.inv(H2)
    beta = beta - Hi @ g2
    ose = np.sqrt(np.diag(Hi @ Vs @ Hi / M))
    return beta, ose


# ---- triplet metric-learning design ----

def get_svec_indices(p):
    q = p * (p + 1) // 2
    rows = np.zeros(q, dtype=int); cols = np.zeros(q, dtype=int); scales = np.zeros(q)
    idx = 0
    for i in range(p):
        for j in range(i, p):
            rows[idx], cols[idx] = i, j
            scales[idx] = 1.0 if i == j else np.sqrt(2.0); idx += 1
    return rows, cols, scales


def svec(M, r, c, s): return M[r, c] * s


def generate_data(n, p, K, class_means, rng):
    Y = rng.integers(0, K, size=n)
    X = rng.standard_normal((n, p))
    for k in range(K):
        X[Y == k] += class_means[k]
    ci = {k: np.where(Y == k)[0] for k in range(K)}
    return X, Y, ci


# ---- class means for the triplet design ----

def aniso_class_means(p, K, rng, sep_lo=0.8, sep_hi=1.8):
    """Class means along rotated axes with DISTINCT per-axis scales,
    giving M* a spread-out (non-degenerate) eigenspectrum."""
    scales = np.linspace(sep_lo, sep_hi, p)
    Q = np.linalg.qr(rng.standard_normal((p, p)))[0]
    means = np.zeros((K, p))
    for k in range(min(K, p)):
        means[k] = scales[k] * Q[:, k]
    if K > p:
        cen = means[:p].mean(0)
        means[p] = -cen / max(np.linalg.norm(cen), 1e-10) * scales.mean()
    means -= means.mean(0)
    return means


# ---- triplet sampling helpers used by the kernels ----

def _anchor_probs(ci):
    classes = sorted(ci.keys()); ntot = sum(len(ci[c]) for c in classes)
    w = np.array([len(ci[c]) * (len(ci[c]) - 1) * (ntot - len(ci[c])) for c in classes], float)
    return classes, (w / w.sum() if w.sum() > 0 else np.ones(len(classes)) / len(classes))


def sample_triplets_wr(B, ci, rng):
    classes, p_cls = _anchor_probs(ci); K = len(classes)
    anc = np.empty(B, np.int64); pos = np.empty(B, np.int64); neg = np.empty(B, np.int64)
    aci = rng.choice(K, size=B, p=p_cls)
    for c in range(K):
        m = aci == c; cnt = int(m.sum())
        if cnt == 0: continue
        idx = ci[classes[c]]; ns = len(idx)
        al = rng.integers(0, ns, size=cnt)
        pl = rng.integers(0, ns - 1, size=cnt); pl[pl >= al] += 1
        anc[m] = idx[al]; pos[m] = idx[pl]
        others = [classes[j] for j in range(K) if j != c]
        osz = np.array([len(ci[o]) for o in others], float)
        nci = rng.choice(len(others), size=cnt, p=osz / osz.sum())
        gp = np.where(m)[0]
        for oi, o in enumerate(others):
            mm = nci == oi; k = int(mm.sum())
            if k == 0: continue
            ineg = ci[o]; neg[gp[mm]] = ineg[rng.integers(0, len(ineg), size=k)]
    return anc, pos, neg


def triplet_features(X, anc, pos, neg, r, c, s):
    dij = X[anc] - X[pos]; dik = X[anc] - X[neg]
    return (dij[:, r] * dij[:, c] - dik[:, r] * dik[:, c]) * s[None, :]


def sample_triplets_wor_fast(B, ci, n, rng):
    """Genuine without-replacement within the minibatch. Collisions are ~0
    for d>=2 at these (n,B), so this matches WR up to MC noise (Thm 4)."""
    anc, pos, neg = sample_triplets_wr(B, ci, rng)
    code = (anc.astype(np.int64) * n + pos) * n + neg
    _, idx = np.unique(code, return_index=True)
    # Redraw until the batch really is distinct, as the pair samplers do.  A
    # single redraw leaves a duplicate whenever a replacement collides again;
    # that is vanishingly rare at these sizes -- 6000 draws produced none -- but
    # the guarantee should come from the loop, not from the odds.
    while len(idx) < B:
        dup = np.ones(B, bool); dup[idx] = False
        nd = int(dup.sum())
        a2, p2, g2 = sample_triplets_wr(nd, ci, rng)
        anc[dup] = a2; pos[dup] = p2; neg[dup] = g2
        code = (anc.astype(np.int64) * n + pos) * n + neg
        _, idx = np.unique(code, return_index=True)
    return anc, pos, neg
