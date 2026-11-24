"""PER-INPUT (single-level) certificates for the ATC campaign.

The companion analysis to analyze_atc.py / analyze_atc_sem.py: instead of the
two-level population fraction G, this emits the per-message object -- for each
(model, corpus, spec, SNR, message) an exact-binomial Clopper-Pearson upper
confidence bound on that message's own noise-draw violation probability V_x,

    CP+(k, N, beta):  P[ V_x > CP+ ] <= beta,   k ~ Bin(N, V_x) exactly,

which is the tightest distribution-free bound from (k, N) and is the honest
primary bound for a per-input paper (the wait-and-judge scenario bound is
uniformly CP-dominated; reported here as a comparison column only).

Pairing discipline (GPU nondeterminism across runs):
  * msg + cs come from the ORIGINAL campaign CSVs (same decode pass, paired);
  * exact_repro + sem90 + sem95 come from the SEMANTIC rerun (same pass,
    paired).  Cross-pass comparisons inherit up to ~0.04 drift in borderline
    certified fractions (fp16 models; emitted as \PiXShiftMax) and are
    avoided in the paper's claims.

Outputs (Stage-5-atc/):
  per_input_cp.csv        one row per (model,corpus,spec,snr,file):
                          k, N, cp_up, cp_lo, eps_wj (scenario, comparison),
                          certified = 1[cp_up <= tau]
  perinput_summary.csv    quantiles of cp_up + certified fraction per cell
  snr_budget.csv          per message: worst (lowest) SNR at which the message
                          is still certified at tau, per spec; 999 = never
  numbers_pi.json         headline numbers
  figures/numbers_pi.tex  LaTeX macros for the manuscript

Per-certificate confidence is beta = 1e-6.  A reader consulting all
K certificates simultaneously pays a union bound K*beta; with the largest
per-model table here (857*8*5 = 34,280 cells) simultaneous validity still
exceeds 1 - 3.5e-2; per-figure families are far smaller.  The manuscript
states this explicitly.

Revision mode (ATC_DATA_DIR=<dir> ATC_REV=1, see atc_paths.py): inputs and
outputs in <dir>, macros in <dir>/figures, plus one more spec, csid (call-sign
identity, callsign_grammar.identity): k_csid comes from the k_csid column of
results_atc_<model>_<ds>.csv (v2 models: same pass as msg/cs) or from
csid_<model>_<ds>.csv (models not rerun, i.e. whisper_atc: SEMANTIC pass).
Its macros use spec letter I: \PiFrac<m><d>I<snr> and \PiMed<m><d>I<snr> at
all eight SNRs, \PiBudgetMed<m><d>I and \PiBudgetNone<m><d>I.

Run:  python analyze_atc_perinput.py
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import beta as _beta

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "Stage-2-verification" / "scenario"))

import risk  # noqa: E402  (scenario epsilon_upper, for the comparison column)
from atc_paths import DATA, FIGS, REV  # noqa: E402  (ATC_DATA_DIR / ATC_REV)

SNRS = [60, 50, 45, 40, 35, 30, 25, 20]
BETA = 1e-6
TAU = 0.05
MODELS = ["quartznet", "deepspeech2", "whisper", "whisper_atc"]
CORPORA = ["atcosim", "atco2"]
QS = [0.10, 0.25, 0.50, 0.75, 0.90]

csv.field_size_limit(1 << 24)


def cp_upper(k, n, b):
    """Exact binomial (Clopper-Pearson) upper bound: P[V > cp] <= b."""
    if k >= n:
        return 1.0
    return float(_beta.ppf(1.0 - b, k + 1, n - k))


def cp_lower(k, n, b):
    if k <= 0:
        return 0.0
    return float(_beta.ppf(b, k, n - k + 1))


_wj_cache = {}


def wj_upper(k, n, b):
    if (k, n) not in _wj_cache:
        _wj_cache[(k, n)] = risk.epsilon_upper(n, k, b)
    return _wj_cache[(k, n)]


def load_original(model, ds):
    """(file, snr) -> dict(msg=k_msg, cs=k_cs, role=...) from the original run."""
    p = DATA / f"results_atc_{model}_{ds}.csv"
    out = {}
    if REV and not p.exists():          # revision rerun still in progress
        return out
    for r in csv.DictReader(open(p, newline="")):
        try:
            if int(r["N"]) != 1000 or len(r["bits_msg"]) != 1000:
                continue
        except (TypeError, ValueError):
            continue
        out[(r["file"], int(r["snr"]))] = dict(          # last-wins on dupes
            role=r["role"], msg=int(r["k_msg"]), cs=int(r["k_cs"]))
        if REV and r.get("k_csid") not in (None, ""):
            out[(r["file"], int(r["snr"]))]["csid"] = int(r["k_csid"])
    if REV:                             # csid of models without a v2 rerun
        pc = DATA / f"csid_{model}_{ds}.csv"
        if pc.exists():
            for r in csv.DictReader(open(pc, newline="")):
                d = out.get((r["file"], int(r["snr"])))
                if d is not None and int(r["N"]) == 1000:
                    d.setdefault("csid", int(r["k_csid"]))
    return out


def load_sem(model, ds):
    p = DATA / f"results_sem_{model}_{ds}.csv"
    out = {}
    if REV and not p.exists():
        return out
    for r in csv.DictReader(open(p, newline="")):
        out[(r["file"], int(r["snr"]))] = dict(
            role=r["role"], exact_repro=int(r["kexact"]),
            sem90=int(r["ksem_90"]), sem95=int(r["ksem_95"]))
    return out


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    i = q * (len(xs) - 1)
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def main():
    rows, nums = [], {}
    paired = inversions = 0             # nested-spec pipeline-integrity check
    budget = defaultdict(dict)          # (model,ds,spec) -> file -> budget SNR
    for model in MODELS:
        for ds in CORPORA:
            orig, sem = load_original(model, ds), load_sem(model, ds)
            for (fn, snr), d in sorted(orig.items()):
                specs = [("msg", d["msg"]), ("cs", d["cs"])]
                if "csid" in d:                  # revision mode only
                    specs.append(("csid", d["csid"]))
                s = sem.get((fn, snr))
                if s:
                    paired += 1
                    if (s["sem90"] > s["exact_repro"]
                            or s["sem95"] > s["exact_repro"]):
                        inversions += 1
                    specs += [("exact_repro", s["exact_repro"]),
                              ("sem90", s["sem90"]), ("sem95", s["sem95"])]
                for spec, k in specs:
                    up = cp_upper(k, 1000, BETA)
                    rows.append(dict(
                        model=model, dataset=ds, spec=spec, snr=snr, file=fn,
                        role=d["role"], k=k, N=1000,
                        cp_up=round(up, 6),
                        cp_lo=round(cp_lower(k, 1000, BETA), 6),
                        eps_wj=round(wj_upper(k, 1000, BETA), 6),
                        certified=int(up <= TAU)))

    with open(DATA / "per_input_cp.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- summaries per (model, corpus, spec, snr) ----
    cells = defaultdict(list)
    for r in rows:
        cells[(r["model"], r["dataset"], r["spec"], r["snr"])].append(r)
    srows = []
    for (m, ds, spec, snr), rs_ in sorted(cells.items()):
        ups = [r["cp_up"] for r in rs_]
        cert = sum(r["certified"] for r in rs_)
        srow = dict(model=m, dataset=ds, spec=spec, snr=snr, M=len(rs_),
                    certified=cert, certified_frac=round(cert / len(rs_), 4))
        for q in QS:
            srow[f"q{int(q*100)}"] = round(quantile(ups, q), 6)
        srows.append(srow)
    with open(DATA / "perinput_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(srows[0].keys()))
        w.writeheader(); w.writerows(srows)

    # ---- per-message SNR budget: lowest SNR still certified, scanning from
    #      60 dB down and stopping at the first failure (budget = last pass) ----
    per_file = defaultdict(dict)
    for r in rows:
        per_file[(r["model"], r["dataset"], r["spec"], r["file"])][r["snr"]] = \
            r["certified"]
    brows = []
    for (m, ds, spec, fn), by_snr in sorted(per_file.items()):
        b = None
        for snr in SNRS:                       # 60 down to 20
            if by_snr.get(snr) == 1:
                b = snr
            else:
                break
        brows.append(dict(model=m, dataset=ds, spec=spec, file=fn,
                          budget_snr=b if b is not None else 999))
        budget[(m, ds, spec)][fn] = b
    with open(DATA / "snr_budget.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(brows[0].keys()))
        w.writeheader(); w.writerows(brows)

    # ---- headline numbers ----
    def cell(m, ds, spec, snr):
        for r in srows:
            if (r["model"], r["dataset"], r["spec"], r["snr"]) == \
                    (m, ds, spec, snr):
                return r
        return None
    for m in MODELS:
        for ds in CORPORA:
            for spec in ("msg", "cs", "sem90", "sem95", "exact_repro"):
                for snr in (60, 40, 20):
                    c = cell(m, ds, spec, snr)
                    if c:
                        nums[f"med_{m}_{ds}_{spec}_{snr}"] = c["q50"]
                        nums[f"cfrac_{m}_{ds}_{spec}_{snr}"] = c["certified_frac"]
            if REV:                     # csid: every SNR of the grid
                for snr in SNRS:
                    c = cell(m, ds, "csid", snr)
                    if c:
                        nums[f"med_{m}_{ds}_csid_{snr}"] = c["q50"]
                        nums[f"cfrac_{m}_{ds}_csid_{snr}"] = c["certified_frac"]
                        nums[f"M_{m}_{ds}_csid_{snr}"] = c["M"]
    for (m, ds, spec), by in budget.items():
        bs = [v for v in by.values() if v is not None]
        nums[f"budget_med_{m}_{ds}_{spec}"] = (
            quantile(sorted(bs), 0.5) if bs else None)
        nums[f"budget_none_{m}_{ds}_{spec}"] = round(
            sum(1 for v in by.values() if v is None) / len(by), 4)
    # CP vs scenario looseness on the per-input object
    rel = [(r["eps_wj"] - r["cp_up"]) / r["cp_up"] for r in rows
           if 0 < r["cp_up"] < 0.999]
    nums["wj_vs_cp_mean_rel_loose"] = round(sum(rel) / len(rel), 4)
    nums["rows"] = len(rows)
    # structural constants of the certificate (not campaign-dependent)
    nums["cp_floor_k0"] = round(cp_upper(0, 1000, BETA), 6)
    kstar = 0
    while cp_upper(kstar + 1, 1000, BETA) <= TAU:
        kstar += 1
    nums["kstar"] = kstar               # largest k with CP+ <= tau
    # nested-spec integrity check (thm:decomp hypothesis)
    nums["paired_rows"] = paired
    nums["sem_inversions"] = inversions
    # max cross-pass certified-fraction drift: msg vs same-pass exact_repro
    cf = {(r["model"], r["dataset"], r["spec"], r["snr"]): r["certified_frac"]
          for r in srows}
    nums["xpass_max_shift"] = round(max(
        abs(cf[(m, ds, "msg", snr)] - cf[(m, ds, "exact_repro", snr)])
        for m in MODELS for ds in CORPORA for snr in SNRS
        if (m, ds, "exact_repro", snr) in cf), 4)
    (DATA / "numbers_pi.json").write_text(json.dumps(nums, indent=2))

    # ---- LaTeX macros ----
    L = ["% AUTO-GENERATED by analyze_atc_perinput.py -- do not edit"]
    ml = {"quartznet": "Q", "deepspeech2": "D", "whisper": "W",
          "whisper_atc": "F"}
    dl = {"atcosim": "A", "atco2": "O"}
    sl = {"msg": "M", "cs": "C", "sem90": "Sn", "sem95": "Sf",
          "exact_repro": "X"}
    for prefix, tag in (("med_", "PiMed"), ("cfrac_", "PiFrac")):
        for m in MODELS:
            for ds in CORPORA:
                for spec in ("msg", "cs", "sem90", "sem95", "exact_repro"):
                    for snr in (60, 40, 20):
                        v = nums.get(f"{prefix[:-1]}_{m}_{ds}_{spec}_{snr}")
                        if v is None:
                            continue
                        name = f"\\{tag}{ml[m]}{dl[ds]}{sl[spec]}{snr}"
                        L.append(f"\\newcommand{{{name}}}{{{v}}}")
    if REV:      # call-sign identity spec: letter I, all eight SNRs
        for prefix, tag in (("med_", "PiMed"), ("cfrac_", "PiFrac")):
            for m in MODELS:
                for ds in CORPORA:
                    for snr in SNRS:
                        v = nums.get(f"{prefix[:-1]}_{m}_{ds}_csid_{snr}")
                        if v is None:
                            continue
                        name = f"\\{tag}{ml[m]}{dl[ds]}I{snr}"
                        L.append(f"\\newcommand{{{name}}}{{{v}}}")
    L.append("\\newcommand{\\PiLooseWJ}{%.1f}"
             % (100 * nums["wj_vs_cp_mean_rel_loose"]))
    L.append("\\newcommand{\\PiRows}{%s}" % f"{nums['rows']:,}")
    L.append("\\newcommand{\\PiFloor}{%s}" % nums["cp_floor_k0"])
    L.append("\\newcommand{\\PiKstar}{%d}" % nums["kstar"])
    L.append("\\newcommand{\\PiPairedRows}{%s}" % f"{nums['paired_rows']:,}")
    L.append("\\newcommand{\\PiSemInversions}{%d}" % nums["sem_inversions"])
    L.append("\\newcommand{\\PiXShiftMax}{%s}" % nums["xpass_max_shift"])
    # per-message SNR-budget summaries (msg + cs): median budget among
    # messages certified at least once, and %% of messages never certified
    for m in MODELS:
        for ds in CORPORA:
            for spec, sltr in (("msg", "M"), ("cs", "C")) + (
                    (("csid", "I"),) if REV else ()):
                bm = nums.get(f"budget_med_{m}_{ds}_{spec}")
                bn = nums.get(f"budget_none_{m}_{ds}_{spec}")
                if bm is not None:
                    bm = int(bm) if float(bm) == int(bm) else bm
                    L.append("\\newcommand{\\PiBudgetMed%s%s%s}{%s}"
                             % (ml[m], dl[ds], sltr, bm))
                if bn is not None:
                    L.append("\\newcommand{\\PiBudgetNone%s%s%s}{%.1f}"
                             % (ml[m], dl[ds], sltr, 100 * bn))
    out_tex = FIGS / "numbers_pi.tex"   # default: the paper repo's figures/
    out_tex.write_text("\n".join(L) + "\n")

    print(f"wrote per_input_cp.csv ({len(rows)} rows), perinput_summary.csv "
          f"({len(srows)} cells), snr_budget.csv ({len(brows)} rows), "
          f"numbers_pi.json, numbers_pi.tex")
    print(f"  scenario-vs-CP mean relative looseness (per-input): "
          f"{nums['wj_vs_cp_mean_rel_loose']:+.1%}")


if __name__ == "__main__":
    main()
