"""Scenario-approach risk certificates for black-box robustness verification.

Theory audited from Kiguli/Scenario-Neural-Network-Verification (Wooding,
2025), which instantiates the Garatti-Campi wait-and-judge risk certificate
[GC22, "Risk and complexity in scenario optimization", Math. Prog. 2022].

Setup for VERIFICATION (no learner / no optimisation):
  * The object under test (the STT model + a fixed nominal decode of a
    fixed utterance) is FIXED before any sample is drawn  -> "freshness".
  * Draw N i.i.d. perturbations delta_i ~ P from the deployment
    perturbation distribution P.
  * Evaluate the violation indicator Phi(delta_i) in {0,1}: 1 iff the
    perturbed output is NON-robust (decode changes / argmax path flips).
  * k = sum_i Phi(delta_i) is the number of non-robust samples = the
    cardinality of the support set s*_N (each violating sample is a
    support constraint; non-violating samples are not).

Given (N, k, beta) the certificates bound the true risk
  V := P{ delta : Phi(delta) = 1 }      (probability a fresh perturbation is non-robust)

One-sided upper bound (Garatti 2025 / one-sided wait-and-judge):
  eps_up(k) = 1 - t,  t in (0,1) the root of
     (beta/N) * sum_{i=k}^{N-1} C(i,k) t^{i-k}  -  C(N,k) t^{N-k} = 0,
  and  P^N{ V > eps_up(k) } <= beta.

Two-sided bound (GC22 Thm 5, as used in the paper's Theorem 2 at k=0):
  roots t_lo <= t_hi in [0,inf) of
     C(N,k) t^{N-k}
       - (beta/2N) sum_{i=k}^{N-1}  C(i,k) t^{i-k}
       - (beta/6N) sum_{i=N+1}^{4N} C(i,k) t^{i-k} = 0,
  eps_lo(k) = max{0, 1 - t_hi},  eps_up2(k) = 1 - t_lo,
  and  P^N{ eps_lo(k) <= V <= eps_up2(k) } >= 1 - beta.

All computations use mpmath for numerical stability at large N.
"""
from functools import lru_cache

import mpmath as mp
import numpy as np
from scipy.optimize import brentq
from scipy.special import betainc, gammaln

mp.mp.dps = 50


def _comb(n, k):
    return mp.binomial(n, k)


def _bisect(f, a, b, iters=200):
    """Root of f on [a,b] with f(a) and f(b) of opposite sign."""
    a, b = mp.mpf(a), mp.mpf(b)
    fa = f(a)
    for _ in range(iters):
        m = (a + b) / 2
        fm = f(m)
        if fm == 0:
            return m
        if (fm > 0) == (fa > 0):
            a, fa = m, fm
        else:
            b = m
    return (a + b) / 2


def _sum_binom(k, a, b, t):
    """sum_{i=a}^{b} C(i,k) t^{i-k}, iterative O(b-a) (a >= k)."""
    if b < a:
        return mp.mpf(0)
    c = _comb(a, k)                 # C(a,k)
    tp = t ** (a - k)               # t^{a-k}
    s = c * tp
    for i in range(a, b):
        c = c * (i + 1) / (i + 1 - k)   # C(i+1,k)
        tp = tp * t
        s += c * tp
    return s


def _epsilon_upper_fast(N, k, beta):
    """Double-precision epsilon_upper for large N. Uses the closed form
    sum_{i=k}^{N-1} C(i,k) t^{i-k} = (1-t)^{-(k+1)} I_{1-t}(k+1, N-k) (regularized
    incomplete beta), so each root-finding step is O(1) instead of O(N). Matches
    the mpmath path to ~1e-15; the certified risk is an O(1) probability, so
    float64 is ample. Used where the O(N) mpmath loop would be prohibitive."""
    N = int(N); k = int(k)
    logCNk = gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1)
    logB = gammaln(k + 1) + gammaln(N - k) - gammaln(N + 1)        # log B(k+1, N-k)

    def g(eps):                                                    # log LHS - log RHS, root at eps_up
        Ib = betainc(k + 1, N - k, eps)                            # I_eps(k+1, N-k), regularized
        logIb = (np.log(Ib) if Ib > 0 else
                 (k + 1) * np.log(eps) + (N - k - 1) * np.log1p(-eps) - np.log(k + 1) - logB)
        return np.log(beta / N) - (k + 1) * np.log(eps) + logIb - logCNk - (N - k) * np.log1p(-eps)
    grid = np.concatenate([[1e-12], np.logspace(-11, np.log10(0.9999), 600)])
    gv = [g(e) for e in grid]
    for i in range(len(grid) - 1):
        if np.isfinite(gv[i]) and np.isfinite(gv[i + 1]) and np.sign(gv[i]) != np.sign(gv[i + 1]):
            return float(brentq(g, grid[i], grid[i + 1], xtol=1e-15, rtol=1e-13))
    return 1.0


@lru_cache(maxsize=None)
def epsilon_upper(N, k, beta):
    """One-sided upper risk bound eps_up(k): P{V > eps_up} <= beta."""
    if k >= N:
        return 1.0
    if N > 20000:                       # mpmath loop is O(N); use the fast closed form
        return _epsilon_upper_fast(N, k, beta)
    bN = mp.mpf(beta) / N
    CNk = _comb(N, k)

    def poly(t):
        return bN * _sum_binom(k, k, N - 1, t) - CNk * t**(N - k)
    t = _bisect(poly, mp.mpf("1e-40"), mp.mpf(1))
    return float(1 - t)


@lru_cache(maxsize=None)
def epsilon_two_sided(N, k, beta):
    """Two-sided bound (eps_lo, eps_up): P{eps_lo <= V <= eps_up} >= 1-beta."""
    if k >= N:
        return (0.0, 1.0)
    N = int(N); k = int(k)
    # Double precision: poly(t) for t in (0,1) via the incomplete-beta closed
    # form (fast + robust). poly(0+)<0, rises through the up-crossing t_lo
    # (-> eps_up = 1 - t_lo), then falls through the down-crossing t_hi
    # (-> eps_lo = max(0, 1 - t_hi); eps_lo=0 if t_hi >= 1, outside (0,1)).
    # The original fixed [.,1]/[1,.] bracket is unreliable when k is large
    # (both roots drop below 1); a sign-change scan over (0,1) is robust.
    logCNk = gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1)
    lc1 = np.log(beta / (2 * N)); lc2 = np.log(beta / (6 * N))

    def g(t):                                 # sign matches poly(t) on t in (0,1)
        x = 1.0 - t; lx = np.log(x)
        logT0 = logCNk + (N - k) * np.log(t)
        I1 = betainc(k + 1, N - k, x)
        logT1 = lc1 - (k + 1) * lx + (np.log(I1) if I1 > 0 else -745.0)
        diff = betainc(k + 1, 4 * N + 1 - k, x) - betainc(k + 1, N + 1 - k, x)
        logT2 = lc2 - (k + 1) * lx + (np.log(diff) if diff > 0 else -745.0)
        m = max(logT1, logT2)
        return logT0 - (m + np.log(np.exp(logT1 - m) + np.exp(logT2 - m)))
    # anchor at a tiny t where poly(0+)<0 is guaranteed, so a k~N up-crossing
    # (which can sit below 1e-6) is still bracketed and assigned to t_lo.
    ts = np.concatenate([[1e-15], np.linspace(1e-6, 1 - 1e-9, 8000)])
    t_lo = t_hi = None
    ps = g(ts[0]) > 0
    for i in range(1, len(ts)):
        s = g(ts[i]) > 0
        if s != ps:
            r = brentq(g, ts[i - 1], ts[i], xtol=1e-12)
            if t_lo is None:
                t_lo = r
            else:
                t_hi = r
                break
        ps = s
    if t_lo is None:
        return (0.0, 1.0)
    eps_lo = max(0.0, 1.0 - t_hi) if t_hi is not None else 0.0
    return (eps_lo, 1.0 - t_lo)


def required_N(eps, beta, k=0):
    """Smallest N such that the one-sided upper bound at k is <= eps."""
    N = max(k + 1, 1)
    while epsilon_upper(N, k, beta) > eps:
        N = int(N * 1.5) + 1
    lo, hiN = max(k + 1, 1), N
    while lo < hiN:
        mid = (lo + hiN) // 2
        if epsilon_upper(mid, k, beta) <= eps:
            hiN = mid
        else:
            lo = mid + 1
    return lo


def two_level(ks_inner, N_inner, tau, beta_inner, beta_outer):
    """Lift per-message certificates to a model-level (across-messages) one.

    ks_inner   : list of per-message violation counts k_j (each from N_inner draws)
    tau        : per-message risk threshold; message j is 'tau-robust' if its
                 true per-perturbation risk V_j <= tau.
    Sound chain: V_j <= eps_up(k_j, N_inner, beta_inner) w.p. >= 1-beta_inner.
      So eps_up(k_j) <= tau  =>  message j is tau-robust (sound direction).
    Level 2 (messages iid): count k2 = #{ j : eps_up(k_j) > tau } as a sound
      over-count of non-tau-robust messages; apply the certificate again.
    Returns dict with the model-level fragile-fraction bounds and the
    composed confidence 1 - (M*beta_inner + beta_outer).
    """
    M = len(ks_inner)
    eps_j = [epsilon_upper(N_inner, k, beta_inner) for k in ks_inner]
    k2 = sum(1 for e in eps_j if e > tau)
    frac_up = epsilon_upper(M, k2, beta_outer)        # P_msg{ V > tau } <= frac_up
    lo2, hi2 = epsilon_two_sided(M, k2, beta_outer)
    composed = M * beta_inner + beta_outer
    return dict(M=M, tau=tau, k2=k2, eps_per_msg=eps_j,
                frac_nonrobust_up=frac_up, frac_nonrobust_lo=lo2,
                frac_nonrobust_hi=hi2, composed_confidence=1 - composed)


if __name__ == "__main__":
    print("Sanity checks (beta = 1e-6):")
    for N in (100, 1000, 5000):
        e0 = epsilon_upper(N, 0, 1e-6)
        simple = 1 - (1e-6) ** (1.0 / N)     # classic (1-eps)^N=beta bound
        print(f"  N={N:5d} k=0: eps_up={e0:.4f}   (classic 1-beta^(1/N)={simple:.4f})")
    print("Risk vs observed violations k (N=2000, beta=1e-6):")
    for k in (0, 1, 5, 20, 100):
        eu = epsilon_upper(2000, k, 1e-6)
        print(f"  k={k:4d}: eps_up={eu:.4f}  (empirical k/N={k/2000:.4f})")
    print("Two-sided (N=2000, beta=1e-6):")
    for k in (0, 10, 100):
        lo, hi = epsilon_two_sided(2000, k, 1e-6)
        print(f"  k={k:4d}: [{lo:.4f}, {hi:.4f}]")
    print("required_N for eps=0.01, k=0:", required_N(0.01, 1e-6, 0))
