"""Kernel adapters for alg_paper.run_algorithm1: two-sample logistic AUC,
role-sampled triplet metric learning, smoothed rank regression.  Each exposes
the interface documented in alg_paper.py.  Conventions: `stats` returns the
mean LOSS, the mean SCORE = gradient of the loss, the mean Hessian, and the
per-tuple scores; the recursion is theta <- theta - D gbar / t."""
import numpy as np
from scipy.special import expit, ndtr
from scipy.stats import norm
from math import comb

from dgp import (get_svec_indices, svec, triplet_features, sample_triplets_wr,
                 sample_triplets_wor_fast)


# ---------------------------------------------------------------- AUC ----
class AUCKernel:
    """Two-sample logistic AUC surrogate: tuple = (positive i, negative j),
    a = X^-_j - X^+_i, loss log(1+e^{a.theta}).  Samples: k=0 positives (d=1),
    k=1 negatives (d=1)."""
    def __init__(self, Xp, Xn):
        self.Xp, self.Xn = Xp, Xn
        self.m, self.nn = len(Xp), len(Xn)
        self.p = Xp.shape[1]
        self.n_ref = self.m + self.nn      # legacy: TOTAL size, not the theorem's n_1
        self.n_total = self.m + self.nn
        self.n_theory_ref = self.m         # n_1 = size of sample 1 = the positives
        self.samples = [(self.m, 1), (self.nn, 1)]
        self.N_tuples = self.m * self.nn

    def draw(self, k, rng):
        return rng.integers(0, self.m, k), rng.integers(0, self.nn, k)

    def draw_wor(self, k, rng):
        ip, jn = self.draw(k, rng)
        code = ip.astype(np.int64) * self.nn + jn
        _, idx = np.unique(code, return_index=True)
        while len(idx) < k:                       # redraw duplicates
            mask = np.ones(k, bool); mask[idx] = False; nd = int(mask.sum())
            ip[mask] = rng.integers(0, self.m, nd); jn[mask] = rng.integers(0, self.nn, nd)
            code = ip.astype(np.int64) * self.nn + jn
            _, idx = np.unique(code, return_index=True)
        return ip, jn

    @staticmethod
    def batch_len(batch): return len(batch[0])

    @staticmethod
    def batch_slice(batch, a, b): return (batch[0][a:b], batch[1][a:b])

    def _feat(self, batch):
        ip, jn = batch
        return self.Xn[jn] - self.Xp[ip]

    def stats(self, batch, theta):
        a = self._feat(batch); u = a @ theta; sg = expit(u); sd = sg * (1 - sg)
        loss = float(np.mean(np.logaddexp(0.0, u)))
        S = sg[:, None] * a
        return loss, S.mean(0), (sd[:, None] * a).T @ a / len(a), S

    def score_mean(self, batch, theta):
        """stats()'s score, nothing else: bit-identical by construction."""
        a = self._feat(batch)
        S = expit(a @ theta)[:, None] * a
        return S.mean(0)

    def anchor_pair(self, kidx, anchor, rng, theta, k_per_stream=1):
        """Two independent streams, each the average of k_per_stream independent
        conditional tuple scores containing the anchor."""
        if kidx == 0:                             # positive anchor, negatives as partners
            j = rng.integers(0, self.nn, 2 * k_per_stream)
            a = self.Xn[j] - self.Xp[anchor]
        else:                                     # negative anchor, positives as partners
            i = rng.integers(0, self.m, 2 * k_per_stream)
            a = self.Xn[anchor] - self.Xp[i]
        S = expit(a @ theta)[:, None] * a
        return S[:k_per_stream].mean(0), S[k_per_stream:].mean(0)

    def anchor_streams(self, kidx, anchors, rng, theta, k):
        """Vectorised over anchors: returns (Y, Z), each (q, p), stream = mean of
        k independent conditional tuple scores."""
        q = len(anchors)
        if kidx == 0:
            j = rng.integers(0, self.nn, (q, 2 * k))
            a = self.Xn[j] - self.Xp[anchors][:, None, :]          # (q, 2k, p)
        else:
            i = rng.integers(0, self.m, (q, 2 * k))
            a = self.Xn[anchors][:, None, :] - self.Xp[i]
        S = expit(a @ theta)[..., None] * a
        return S[:, :k].mean(1), S[:, k:].mean(1)

    # exact first projections at small n (for the estimator test)
    def exact_zeta(self, theta):
        a = self.Xn[None, :, :] - self.Xp[:, None, :]           # (m, nn, p)
        S = expit(a @ theta)[..., None] * a                      # (m, nn, p)
        mu_pos = S.mean(1); mu_neg = S.mean(0); gbar = S.mean((0, 1))
        z1 = ((mu_pos - gbar).T @ (mu_pos - gbar)) / self.m
        z2 = ((mu_neg - gbar).T @ (mu_neg - gbar)) / self.nn
        return z1, z2


# ------------------------------------------------------------ triplet ----
class TripletKernel:
    """Role-sampled triplet metric learning.  Optimisation side: uniform VALID
    ordered (anchor, positive, negative) triples, role loss log(1+e^{theta.v}).
    Data side (anchor estimator): unordered triples containing the anchor
    observation, score of the SYMMETRIC kernel H = (1/6) sum_pi h_pi (only the
    valid orderings are non-zero), scaled by gamma_n = 6 C(n,3)/|R_n|."""
    def __init__(self, X, Y, ci):
        self.X, self.Y, self.ci = X, Y, ci
        self.n, self.p = X.shape
        self.r, self.c, self.s = get_svec_indices(self.p)
        self.q = len(self.s)
        self.n_ref = self.n
        self.n_total = self.n
        self.n_theory_ref = self.n         # one sample: n_1 = n
        self.samples = [(self.n, 3)]
        self.R = sum(len(ci[k]) * (len(ci[k]) - 1) * (self.n - len(ci[k])) for k in ci)
        self.N_tuples = self.R
        self.gamma_n = 6 * comb(self.n, 3) / self.R

    def draw(self, k, rng):
        return sample_triplets_wr(k, self.ci, rng)

    def draw_wor(self, k, rng):
        return sample_triplets_wor_fast(k, self.ci, self.n, rng)

    @staticmethod
    def batch_len(batch): return len(batch[0])

    @staticmethod
    def batch_slice(batch, a, b): return (batch[0][a:b], batch[1][a:b], batch[2][a:b])

    def _feat(self, batch):
        anc, pos, neg = batch
        return triplet_features(self.X, anc, pos, neg, self.r, self.c, self.s)

    def stats(self, batch, theta):
        v = self._feat(batch); u = v @ theta; sg = expit(u); sd = sg * (1 - sg)
        loss = float(np.mean(np.logaddexp(0.0, u)))
        S = sg[:, None] * v
        return loss, S.mean(0), (sd[:, None] * v).T @ v / len(v), S

    def score_mean(self, batch, theta):
        """stats()'s score, nothing else: bit-identical by construction."""
        v = self._feat(batch)
        S = expit(v @ theta)[:, None] * v
        return S.mean(0)

    def sym_scores(self, i, j, k, theta):
        """Vectorised: gamma_n * score of the symmetric kernel H on the unordered
        triples {i_r, j_r, k_r}.  H = (1/6) sum over the 6 orderings of the role
        loss; an ordering (a, p, neg) is valid iff Y_a = Y_p != Y_neg."""
        i = np.asarray(i); j = np.asarray(j); k = np.asarray(k)
        out = np.zeros((len(i), self.q))
        trip = (i, j, k); Y = self.Y
        for a in range(3):
            for pidx in range(3):
                if pidx == a: continue
                nidx = 3 - a - pidx
                ia, ip_, in_ = trip[a], trip[pidx], trip[nidx]
                valid = (Y[ia] == Y[ip_]) & (Y[ia] != Y[in_])
                if not valid.any(): continue
                v = triplet_features(self.X, ia[valid], ip_[valid], in_[valid], self.r, self.c, self.s)
                out[valid] += expit(v @ theta)[:, None] * v
        return self.gamma_n * out / 6.0

    def _sym_score(self, i, j, k, theta):
        return self.sym_scores([i], [j], [k], theta)[0]

    def _two_others(self, anchor, rng, size=1):
        """`size` uniform unordered pairs of distinct observations != anchor."""
        j = rng.integers(0, self.n - 1, size); j[j >= anchor] += 1
        k = rng.integers(0, self.n - 2, size)
        # k uniform over the n-2 indices excluding anchor and j
        lo = np.minimum(anchor, j); hi = np.maximum(anchor, j)
        k[k >= lo] += 1; k[k >= hi] += 1
        return j, k

    def anchor_pair(self, kidx, anchor, rng, theta, k_per_stream=1):
        j1, k1 = self._two_others(anchor, rng, k_per_stream)
        j2, k2 = self._two_others(anchor, rng, k_per_stream)
        a = np.full(k_per_stream, anchor)
        return self.sym_scores(a, j1, k1, theta).mean(0), self.sym_scores(a, j2, k2, theta).mean(0)

    def anchor_streams(self, kidx, anchors, rng, theta, k):
        q = len(anchors); A = np.repeat(anchors, 2 * k)                # (q*2k,)
        j = rng.integers(0, self.n - 1, q * 2 * k); j[j >= A] += 1
        kk = rng.integers(0, self.n - 2, q * 2 * k)
        lo = np.minimum(A, j); hi = np.maximum(A, j); kk[kk >= lo] += 1; kk[kk >= hi] += 1
        S = self.sym_scores(A, j, kk, theta).reshape(q, 2 * k, self.q)
        return S[:, :k].mean(1), S[:, k:].mean(1)

    def exact_zeta(self, theta):
        """Exact first projection at small n: mu_i = mean over all unordered
        triples containing i of the gamma_n-scaled symmetric score."""
        n = self.n
        mu = np.zeros((n, self.q)); cnt = np.zeros(n)
        tot = np.zeros(self.q); ntrip = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    sc = self._sym_score(i, j, k, theta)
                    for a in (i, j, k):
                        mu[a] += sc; cnt[a] += 1
                    tot += sc; ntrip += 1
        mu /= cnt[:, None]; gbar = tot / ntrip
        return ((mu - gbar).T @ (mu - gbar)) / n


# --------------------------------------------------------------- rank ----
class RankKernel:
    """Smoothed rank regression: pair (i,j), z = (dy - dx.theta)/h,
    loss = h (z Phi(z) + phi(z) - z/2)  (gradient -dx (Phi(z)-1/2)),
    Hessian phi(z)/h dx dx^T.  Bandwidth h = sigma_OLS n^{-beta} fixed at start."""
    def __init__(self, X, y, beta=0.26):
        self.X, self.y = X, y
        self.n, self.p = X.shape
        self.n_ref = self.n
        self.n_total = self.n
        self.n_theory_ref = self.n         # one sample: n_1 = n
        self.samples = [(self.n, 2)]
        self.N_tuples = self.n * (self.n - 1) // 2
        self.theta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        self.h = float(np.std(y - X @ self.theta_ols, ddof=1) * self.n ** (-beta))

    def draw(self, k, rng):
        i = rng.integers(0, self.n, k)
        j = rng.integers(0, self.n - 1, k); j[j >= i] += 1
        return i, j

    def draw_wor(self, k, rng):
        assert k <= self.N_tuples, f"WOR minibatch {k} exceeds the {self.N_tuples} distinct pairs"
        i, j = self.draw(k, rng)
        lo, hi = np.minimum(i, j), np.maximum(i, j)
        code = lo.astype(np.int64) * self.n + hi
        _, idx = np.unique(code, return_index=True)
        while len(idx) < k:
            mask = np.ones(k, bool); mask[idx] = False; nd = int(mask.sum())
            i2, j2 = self.draw(nd, rng); i[mask] = i2; j[mask] = j2
            lo, hi = np.minimum(i, j), np.maximum(i, j)
            code = lo.astype(np.int64) * self.n + hi
            _, idx = np.unique(code, return_index=True)
        return i, j

    @staticmethod
    def batch_len(batch): return len(batch[0])

    @staticmethod
    def batch_slice(batch, a, b): return (batch[0][a:b], batch[1][a:b])

    def stats(self, batch, theta):
        i, j = batch
        dx = self.X[i] - self.X[j]; dy = self.y[i] - self.y[j]
        z = (dy - dx @ theta) / self.h
        Phi = norm.cdf(z); phi = norm.pdf(z)
        loss = float(np.mean(self.h * (z * Phi + phi - z / 2)))
        S = -(Phi - 0.5)[:, None] * dx
        H = ((phi / self.h)[:, None] * dx).T @ dx / len(z)
        return loss, S.mean(0), H, S

    def score_mean(self, batch, theta):
        """stats()'s score, nothing else: bit-identical (ndtr == norm.cdf)."""
        i, j = batch
        dx = self.X[i] - self.X[j]; dy = self.y[i] - self.y[j]
        z = (dy - dx @ theta) / self.h
        S = -(ndtr(z) - 0.5)[:, None] * dx
        return S.mean(0)

    def anchor_pair(self, kidx, anchor, rng, theta, k_per_stream=1):
        j = rng.integers(0, self.n - 1, 2 * k_per_stream); j[j >= anchor] += 1
        dx = self.X[anchor] - self.X[j]; dy = self.y[anchor] - self.y[j]
        z = (dy - dx @ theta) / self.h
        S = -(norm.cdf(z) - 0.5)[:, None] * dx
        return S[:k_per_stream].mean(0), S[k_per_stream:].mean(0)

    def anchor_streams(self, kidx, anchors, rng, theta, k):
        q = len(anchors)
        j = rng.integers(0, self.n - 1, (q, 2 * k)); j[j >= anchors[:, None]] += 1
        dx = self.X[anchors][:, None, :] - self.X[j]; dy = self.y[anchors][:, None] - self.y[j]
        z = (dy - dx @ theta) / self.h
        S = -(norm.cdf(z) - 0.5)[..., None] * dx
        return S[:, :k].mean(1), S[:, k:].mean(1)

    def exact_zeta(self, theta):
        n = self.n
        dx = self.X[:, None, :] - self.X[None, :, :]; dy = self.y[:, None] - self.y[None, :]
        z = (dy - dx @ theta) / self.h
        S = -(norm.cdf(z) - 0.5)[..., None] * dx          # (n,n,p), diagonal is 0
        mu = S.sum(1) / (n - 1); gbar = S.sum((0, 1)) / (n * (n - 1))
        return ((mu - gbar).T @ (mu - gbar)) / n

    def full_minimizer(self, theta0=None, iters=50, chunk=2_000_000):
        """theta_hat_n: Newton on the COMPLETE pairwise smoothed objective.

        The pair set is walked in blocks of `chunk`, so the memory this needs
        does not grow with n and the exact minimizer stays reachable at sample
        sizes where materialising every pair at once would not fit.
        """
        th = self.theta_ols.copy() if theta0 is None else theta0.copy()
        iu = np.triu_indices(self.n, 1)
        npairs = len(iu[0])
        p = self.X.shape[1]
        for _ in range(iters):
            g = np.zeros(p); H = np.zeros((p, p))
            for s0 in range(0, npairs, chunk):
                s1 = min(s0 + chunk, npairs)
                i, j = iu[0][s0:s1], iu[1][s0:s1]
                dx = self.X[i] - self.X[j]; dy = self.y[i] - self.y[j]
                z = (dy - dx @ th) / self.h
                g += (-(norm.cdf(z) - 0.5)[:, None] * dx).sum(0)
                H += ((norm.pdf(z) / self.h)[:, None] * dx).T @ dx
            g /= npairs; H /= npairs
            d = np.linalg.solve(H, g); th = th - d
            if np.linalg.norm(d) < 1e-12: break
        return th
