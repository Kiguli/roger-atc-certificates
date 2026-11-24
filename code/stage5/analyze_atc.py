"""Certificates, out-of-sample validation, and ablations for the ATC split study.

Reads results_atc_<model>_<dataset>.csv (from run_atc_split.py; certificate +
ablation roles, per-draw bitstrings) and split_manifest.csv, and writes -- all
in this directory:

  cert_summary.csv       one row per (model,dataset,spec,snr): the scenario
                         certificates computed on the CERTIFICATE split only
                         (joint single-level eps_up over the pooled M*N draws;
                         two-level tau-robust message fraction, tau=0.05,
                         beta=1e-6, beta1=beta/2M, beta2=beta/2).
  per_message_cert.csv   per (model,dataset,spec,snr,file): k_j, eps_j,
                         tau-robust flag -- the level-1 certificates.
  atc_summary.json       the same certificates keyed for figure generators and
                         a future analyze_validation.py lookup branch:
                         ["<model>/<ds>"]["per_spec"][spec]["curve"][snr].
  validation_oos.csv     out-of-sample check on the ABLATION split: cell
                         (model,dataset,spec,snr) holds iff G <= R_test where
                         R_test = frac of held-out msgs with k/N <= tau
                         (same criterion as Stage-4 validation harness),
                         with a 95% Wilson CI on R_test.
  ablation_tau.csv       G(tau) on cert split + R_test(tau) on ablation split
                         over TAU_GRID.
  ablation_beta.csv      G(beta) on cert split over BETA_GRID (tau=0.05).
  ablation_N.csv         certificates recomputed from the first N' draws
                         (bitstring prefixes) on the cert split, and R_test at
                         N' on the ablation split, N' in NPRIME_GRID.
  ablation_M.csv         G over seeded size-M' subsamples of the cert split
                         (R=25 repeats; beta1 rescaled to beta/(2M')).
  ablation_redraw.csv    fresh-vs-June per-(file,snr) violation counts (the
                         June run drew an independent noise stream), plus
                         per-(model,dataset,spec) correlation summary in
                         ablation_redraw_summary.csv.

Statistical conventions follow scenario/aggregate_datasets.py (certificates)
and Stage-4-final/validation/analyze_validation.py (validation): a message is
tau-robust (certified) iff eps_up(k_j, N, beta1) <= tau; the certified robust
fraction is G = 1 - hi2 with (lo2,hi2) = epsilon_two_sided(M, k2, beta2);
an OOS message counts as robust iff k/N <= tau; a cell holds iff G <= R_test.

epsilon_upper uses risk._epsilon_upper_fast (float64 closed form) after an
startup agreement check against the exact mpmath path (rel. diff < 1e-9 on a
(N,k) probe grid); pass --exact to force mpmath everywhere.

Run:  python analyze_atc.py [--partial] [--exact]
        --partial : analyse whatever files are complete so far (mid-run);
                    certificates then use the reduced M and are NOT final.
"""
import csv
import json
import os
import sys
import zlib
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCEN = ROOT / "Stage-2-verification" / "scenario"
sys.path.insert(0, str(SCEN))
import risk  # noqa: E402

csv.field_size_limit(1 << 24)

MODELS = ["quartznet", "deepspeech2", "whisper", "whisper_atc"]
DATASETS = ["atcosim", "atco2"]
SPECS = ["msg", "cs"]
SNRS = [60, 50, 45, 40, 35, 30, 25, 20]
TAG = os.environ.get("ATC_TAG", "")   # analyse a tagged (smoke) run instead
BETA0 = 1e-6
TAU0 = 0.05
TAU_GRID = [0.01, 0.02, 0.05, 0.10, 0.20]
BETA_GRID = [1e-2, 1e-4, 1e-6, 1e-9]
NPRIME_GRID = [50, 100, 200, 500, 1000]
MPRIME_GRID = [50, 100, 200, 400, 600, 1000]
R_SUB = 25
SEED = 20260709
Z95 = 1.959963984540054

PARTIAL = "--partial" in sys.argv
EXACT = "--exact" in sys.argv


# ---------------------------------------------------------------- risk plumbing
def _verify_fast_path():
    """The fast closed form is only auto-used for N>20000 in risk.py; check it
    against the exact mpmath path on a probe grid before trusting it here.
    Requirements: never anti-conservative (fast >= exact - fp noise), and tight
    (rel diff < 1e-7) wherever exact < 0.999. The only observed deviation is a
    saturation to 1.0 at k ~ N where exact ~ 0.99995 -- conservative, and in a
    regime where the message is counted fragile for any sane tau anyway."""
    worst_tight, worst_anti = 0.0, 0.0
    for N in (50, 200, 1000):
        for k in (0, 1, 5, N // 20, N // 3, N - 2):
            for beta in (1e-2, 1e-6, 5e-10):
                a = risk.epsilon_upper(N, k, beta)          # exact (N<=20000)
                b = risk._epsilon_upper_fast(N, k, beta)
                worst_anti = max(worst_anti, a - b)         # >0 = under-reports
                if a < 0.999:
                    worst_tight = max(worst_tight, abs(a - b) / max(a, 1e-12))
    print(f"fast-path: worst rel diff {worst_tight:.2e} (exact<0.999), "
          f"worst anti-conservative {worst_anti:.2e}")
    return worst_tight < 1e-7 and worst_anti < 1e-9


if EXACT:
    eps_up = risk.epsilon_upper
else:
    assert _verify_fast_path(), "fast epsilon_upper disagrees with mpmath path"

    @lru_cache(maxsize=None)
    def eps_up(N, k, beta):
        return 1.0 if k >= N else risk._epsilon_upper_fast(N, k, beta)


def wilson(k, n):
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + Z95**2 / n
    c = p + Z95**2 / (2 * n)
    h = Z95 * np.sqrt(p * (1 - p) / n + Z95**2 / (4 * n**2))
    return ((c - h) / d, (c + h) / d)


def two_level_G(ks, N, tau, beta):
    """Two-level certificate on per-message counts ks (beta split as in
    aggregate_datasets.py: beta1=beta/(2M) per message, beta2=beta/2 outer)."""
    M = len(ks)
    b1, b2 = beta / (2 * M), beta / 2
    ej = [eps_up(N, k, b1) for k in ks]
    k2 = sum(1 for e in ej if e > tau)
    lo2, hi2 = risk.epsilon_two_sided(M, k2, b2)
    return dict(M=M, k2=k2, eps_j=ej, frac_fragile_hi=hi2,
                frac_robust_lo=1.0 - hi2, beta1=b1, beta2=b2)


# ------------------------------------------------------------------- data load
def load_results(model, ds):
    """-> role -> file -> snr -> row (dedup: last occurrence wins).
    Validates completeness (all SNRs per file; all split files present)."""
    path = HERE / f"results_atc_{model}_{ds}{TAG}.csv"
    if not path.exists():
        return {}
    by = defaultdict(dict)
    for r in csv.DictReader(open(path, newline="")):
        by[(r["role"], r["file"])][int(r["snr"])] = r
    split = defaultdict(list)
    for r in csv.DictReader(open(HERE / "split_manifest.csv", newline="")):
        if r["dataset"] == ds:
            split[r["role"]].append(r["file"])
    out = {}
    for role, files in split.items():
        good = {}
        for fn in files:
            rows = by.get((role, fn), {})
            if set(rows) >= set(SNRS):
                good[fn] = rows
        missing = len(files) - len(good)
        if missing and not PARTIAL:
            raise SystemExit(f"{model}/{ds}/{role}: {missing}/{len(files)} files "
                             f"incomplete -- run still going? (use --partial)")
        if missing:
            print(f"  [partial] {model}/{ds}/{role}: using {len(good)}/{len(files)}")
        out[role] = good
    return out


def load_june(model, ds):
    path = SCEN / f"results_{model}_{ds}.csv"
    if not path.exists():          # whisper has no June ATC baseline
        return {}
    out = {}
    for r in csv.DictReader(open(path, newline="")):
        out[(r["file"], int(r["snr"]))] = (int(r["k_msg"]), int(r["k_cs"]))
    return out


def ks_of(rows_by_file, spec, snr, nprime=None):
    """Per-message counts (sorted by filename for determinism). If nprime is
    given, recount from the first nprime draws of the bitstring prefix."""
    ks, files = [], sorted(rows_by_file)
    for fn in files:
        r = rows_by_file[fn][snr]
        if nprime is None:
            ks.append(int(r[f"k_{spec}"]))
        else:
            ks.append(r[f"bits_{spec}"].count("1", 0, nprime))
    return files, ks


# ---------------------------------------------------------------------- main
def main():
    cert_rows, permsg_rows, oos_rows = [], [], []
    tau_rows, beta_rows, n_rows, m_rows = [], [], [], []
    redraw_rows, redraw_sum = [], []
    summary = {"meta": dict(beta=BETA0, tau=TAU0, snrs=SNRS, seed=SEED,
                            tau_grid=TAU_GRID, beta_grid=BETA_GRID,
                            nprime_grid=NPRIME_GRID, mprime_grid=MPRIME_GRID)}

    for model in MODELS:
        for ds in DATASETS:
            print(f"== {model}/{ds}")
            data = load_results(model, ds)
            cert, abl = data.get("certificate", {}), data.get("ablation", {})
            if not cert or not abl:
                print(f"  no data yet, skipping"); continue
            june = load_june(model, ds)
            all_N = {int(r["N"]) for fb in data.values()
                     for rows in fb.values() for r in rows.values()}
            assert len(all_N) == 1, f"mixed N in {model}/{ds}: {all_N}"
            N0 = all_N.pop()
            nprimes = [n for n in NPRIME_GRID if n <= N0]
            key = f"{model}/{ds}"
            summary[key] = {"M_cert": len(cert), "M_abl": len(abl), "N": N0,
                            "per_spec": {s: {"curve": {}} for s in SPECS}}

            for spec in SPECS:
                for snr in SNRS:
                    cf, cks = ks_of(cert, spec, snr)
                    af, aks = ks_of(abl, spec, snr)
                    M, Mt = len(cks), len(aks)

                    # -- certificate (cert split only) --
                    tl = two_level_G(cks, N0, TAU0, BETA0)
                    joint = risk.epsilon_upper(M * N0, sum(cks), BETA0)
                    G = tl["frac_robust_lo"]
                    cert_rows.append(dict(
                        model=model, dataset=ds, spec=spec, snr=snr, M=M, N=N0,
                        beta=BETA0, tau=TAU0, k_sum=sum(cks),
                        joint_eps_up=joint, k2=tl["k2"],
                        frac_fragile_hi=tl["frac_fragile_hi"], frac_robust_lo=G,
                        beta1=tl["beta1"], beta2=tl["beta2"]))
                    for fn, k, e in zip(cf, cks, tl["eps_j"]):
                        permsg_rows.append(dict(
                            model=model, dataset=ds, spec=spec, snr=snr,
                            file=fn, k=k, N=N0, eps_j=e,
                            tau_robust=int(e <= TAU0)))
                    summary[key]["per_spec"][spec]["curve"][str(snr)] = dict(
                        joint_eps_up=joint, k2=tl["k2"],
                        frac_robust=G, frac_fragile=tl["frac_fragile_hi"])

                    # -- out-of-sample validation (ablation split) --
                    # lenient rule (repo convention, analyze_validation.py):
                    # robust iff empirical k/N <= tau. strict rule: apply the
                    # certificate's own per-message test eps_up(k,N,beta1)<=tau
                    # to the held-out messages (systematically harder).
                    nrob = sum(1 for k in aks if k <= TAU0 * N0 + 1e-9)
                    nrob_s = sum(1 for k in aks
                                 if eps_up(N0, k, tl["beta1"]) <= TAU0)
                    rt, rt_s = nrob / Mt, nrob_s / Mt
                    lo, hi = wilson(nrob, Mt)
                    oos_rows.append(dict(
                        model=model, dataset=ds, spec=spec, snr=snr, tau=TAU0,
                        G=G, M_test=Mt, n_robust=nrob, R_test=rt,
                        ci_lo=lo, ci_hi=hi, gap=rt - G,
                        holds=int(G <= rt + 1e-9),
                        R_test_strict=rt_s,
                        holds_strict=int(G <= rt_s + 1e-9)))

                    # -- tau ablation --
                    for tau in TAU_GRID:
                        k2t = sum(1 for e in tl["eps_j"] if e > tau)
                        _, hi2t = risk.epsilon_two_sided(M, k2t, tl["beta2"])
                        nrt = sum(1 for k in aks if k <= tau * N0 + 1e-9)
                        tau_rows.append(dict(
                            model=model, dataset=ds, spec=spec, snr=snr,
                            tau=tau, k2=k2t, G=1 - hi2t, R_test=nrt / Mt,
                            M=M, M_test=Mt,
                            holds=int(1 - hi2t <= nrt / Mt + 1e-9)))

                    # -- beta ablation (cert side; tau=0.05) --
                    for beta in BETA_GRID:
                        tlb = two_level_G(cks, N0, TAU0, beta)
                        beta_rows.append(dict(
                            model=model, dataset=ds, spec=spec, snr=snr,
                            beta=beta, tau=TAU0, k2=tlb["k2"],
                            G=tlb["frac_robust_lo"]))

                    # -- M ablation (cert side; seeded subsamples) --
                    rng = np.random.default_rng(
                        [SEED, zlib.crc32(f"{model}|{ds}|{spec}|{snr}".encode())])
                    cks_arr = np.array(cks)
                    for Mp in [m for m in MPRIME_GRID if m <= M]:
                        reps = 1 if Mp == M else R_SUB
                        gs = []
                        for _ in range(reps):
                            idx = (np.arange(M) if Mp == M
                                   else rng.choice(M, Mp, replace=False))
                            tlm = two_level_G(tuple(cks_arr[idx]), N0, TAU0, BETA0)
                            gs.append(tlm["frac_robust_lo"])
                        m_rows.append(dict(
                            model=model, dataset=ds, spec=spec, snr=snr,
                            M_sub=Mp, reps=reps, G_mean=float(np.mean(gs)),
                            G_std=float(np.std(gs)), G_min=float(np.min(gs)),
                            G_max=float(np.max(gs))))

                # -- N ablation (prefix recount; cert G + ablation R_test) --
                for snr in SNRS:
                    for npr in nprimes:
                        _, cksn = ks_of(cert, spec, snr, nprime=npr)
                        _, aksn = ks_of(abl, spec, snr, nprime=npr)
                        tln = two_level_G(tuple(cksn), npr, TAU0, BETA0)
                        nrn = sum(1 for k in aksn if k <= TAU0 * npr + 1e-9)
                        n_rows.append(dict(
                            model=model, dataset=ds, spec=spec, snr=snr,
                            N_prime=npr, k2=tln["k2"],
                            G=tln["frac_robust_lo"], R_test=nrn / len(aksn),
                            holds=int(tln["frac_robust_lo"]
                                      <= nrn / len(aksn) + 1e-9)))

            # -- redraw comparison vs June (independent noise stream) --
            if not june:
                continue               # whisper: no June ATC baseline
            for role, rows_by_file in (("certificate", cert), ("ablation", abl)):
                for fn in sorted(rows_by_file):
                    for snr in SNRS:
                        r = rows_by_file[fn][snr]
                        jj = june.get((fn, snr))
                        if jj is None:
                            continue
                        redraw_rows.append(dict(
                            model=model, dataset=ds, role=role, file=fn,
                            snr=snr, k_msg_fresh=int(r["k_msg"]),
                            k_msg_june=jj[0], k_cs_fresh=int(r["k_cs"]),
                            k_cs_june=jj[1]))
            sub = [r for r in redraw_rows
                   if r["model"] == model and r["dataset"] == ds]
            for spec, a, b in (("msg", "k_msg_fresh", "k_msg_june"),
                               ("cs", "k_cs_fresh", "k_cs_june")):
                x = np.array([r[a] for r in sub], float)
                y = np.array([r[b] for r in sub], float)
                r_p = (float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else
                       float("nan"))
                redraw_sum.append(dict(
                    model=model, dataset=ds, spec=spec, n_pairs=len(x),
                    pearson_r=r_p, mean_abs_dk=float(np.mean(np.abs(x - y))),
                    max_abs_dk=float(np.max(np.abs(x - y))),
                    mean_k_fresh=float(np.mean(x)),
                    mean_k_june=float(np.mean(y))))

    def dump(name, rows):
        name = name.replace(".csv", f"{TAG}.csv")
        if not rows:
            print(f"  (no rows for {name})"); return
        with open(HERE / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"wrote {name} ({len(rows)} rows)")

    dump("cert_summary.csv", cert_rows)
    dump("per_message_cert.csv", permsg_rows)
    dump("validation_oos.csv", oos_rows)
    dump("ablation_tau.csv", tau_rows)
    dump("ablation_beta.csv", beta_rows)
    dump("ablation_N.csv", n_rows)
    dump("ablation_M.csv", m_rows)
    dump("ablation_redraw.csv", redraw_rows)
    dump("ablation_redraw_summary.csv", redraw_sum)
    with open(HERE / f"atc_summary{TAG}.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"wrote atc_summary{TAG}.json")

    ok = [r for r in oos_rows if r["holds"]]
    if oos_rows:
        gaps = [r["gap"] for r in oos_rows]
        print(f"\nOOS validation: {len(ok)}/{len(oos_rows)} cells hold "
              f"(mean gap {np.mean(gaps):+.3f}, min {np.min(gaps):+.3f})")


if __name__ == "__main__":
    main()
