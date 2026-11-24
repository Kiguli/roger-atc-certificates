"""Certificates for the SEMANTIC-equivalence spec on the ATC corpora.

Consumes results_sem_<model>_<corpus>.csv (produced by run_atc_sem.py) and
applies the identical two-level construction analyze_atc.py uses for the
message and call-sign specs -- per-message eps_j = eps_up(k_j, N, beta1),
then G = 1 - hi2 over the M messages -- for both stored floors (0.90, 0.95).

Three outputs:

  cert_summary_sem.csv     certificates on the certificate split, one row per
                           (model, corpus, spec, snr) with spec in
                           {sem90, sem95, exact_repro}
  validation_sem.csv       out-of-sample check on the held-out ablation split:
                           a cell holds iff G <= R_test, R_test = fraction of
                           held-out messages with k/N <= tau
  numbers_sem.json         headline values for the writeup

`exact_repro` is the exact-match spec recomputed from THIS run's kexact.
Because the perturbation stream was reproduced bit-exactly (verified per
combo), it should agree with the original campaign's message-spec
certificate; the agreement is reported as a self-check rather than assumed.

Run:  python analyze_atc_sem.py
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_atc as A                                  # noqa: E402

SNRS = A.SNRS
TAU, BETA = A.TAU0, A.BETA0
MODELS = ["quartznet", "deepspeech2", "whisper", "whisper_atc"]
CORPORA = ["atcosim", "atco2"]
SPECS = [("sem90", "ksem_90"), ("sem95", "ksem_95"), ("exact_repro", "kexact")]

csv.field_size_limit(1 << 24)


def load(model, ds):
    """-> {role: {(file, snr): row}}"""
    p = HERE / f"results_sem_{model}_{ds}.csv"
    if not p.exists():
        return {}
    out = defaultdict(dict)
    for r in csv.DictReader(open(p, newline="")):
        out[r["role"]][(r["file"], int(r["snr"]))] = r
    return out


def main():
    cert_rows, val_rows, nums = [], [], {}
    for model in MODELS:
        for ds in CORPORA:
            data = load(model, ds)
            if not data:
                print(f"  (skip {model}/{ds}: no results)")
                continue
            cert, abl = data.get("certificate", {}), data.get("ablation", {})
            for spec, key in SPECS:
                for snr in SNRS:
                    ks = [int(r[key]) for (f, s), r in cert.items() if s == snr]
                    if not ks:
                        continue
                    N = int(next(iter(cert.values()))["N"])
                    tl = A.two_level_G(ks, N, TAU, BETA)
                    cert_rows.append(dict(
                        model=model, dataset=ds, spec=spec, snr=snr,
                        M=tl["M"], N=N, beta=BETA, tau=TAU,
                        k_sum=sum(ks), k2=tl["k2"],
                        frac_fragile_hi=tl["frac_fragile_hi"],
                        frac_robust_lo=tl["frac_robust_lo"],
                        beta1=tl["beta1"], beta2=tl["beta2"]))
                    # out-of-sample: held-out empirical robust fraction
                    kh = [int(r[key]) for (f, s), r in abl.items() if s == snr]
                    if kh:
                        r_test = sum(1 for k in kh if k / N <= TAU) / len(kh)
                        G = tl["frac_robust_lo"]
                        val_rows.append(dict(
                            model=model, dataset=ds, spec=spec, snr=snr,
                            M_cert=tl["M"], M_test=len(kh), G=G,
                            R_test=r_test, margin=r_test - G,
                            holds=int(G <= r_test + 1e-12)))

    with open(HERE / "cert_summary_sem.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cert_rows[0].keys()))
        w.writeheader(); w.writerows(cert_rows)
    with open(HERE / "validation_sem.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(val_rows[0].keys()))
        w.writeheader(); w.writerows(val_rows)

    # ---- self-check: exact_repro vs the original message-spec certificate ----
    orig = {}
    for r in csv.DictReader(open(HERE / "cert_summary.csv", newline="")):
        if r["spec"] == "msg":
            orig[(r["model"], r["dataset"], int(r["snr"]))] = float(
                r["frac_robust_lo"])
    diffs = []
    for r in cert_rows:
        if r["spec"] != "exact_repro":
            continue
        o = orig.get((r["model"], r["dataset"], r["snr"]))
        if o is not None:
            diffs.append(abs(o - r["frac_robust_lo"]))
    nums["repro_max_abs_diff_vs_original_msg"] = max(diffs) if diffs else None
    nums["repro_cells_compared"] = len(diffs)

    nholds = sum(r["holds"] for r in val_rows)
    nums["validation_cells"] = len(val_rows)
    nums["validation_holds"] = nholds
    nums["validation_mean_margin"] = (
        sum(r["margin"] for r in val_rows) / len(val_rows)) if val_rows else None

    def g(model, ds, spec, snr):
        for r in cert_rows:
            if (r["model"], r["dataset"], r["spec"], r["snr"]) == \
                    (model, ds, spec, snr):
                return round(r["frac_robust_lo"], 4)
        return None
    for m in MODELS:
        for ds in CORPORA:
            for spec in ("sem90", "sem95", "exact_repro"):
                nums[f"G_{m}_{ds}_{spec}_60dB"] = g(m, ds, spec, 60)
                nums[f"G_{m}_{ds}_{spec}_20dB"] = g(m, ds, spec, 20)
    (HERE / "numbers_sem.json").write_text(json.dumps(nums, indent=2))

    print(f"wrote cert_summary_sem.csv ({len(cert_rows)} rows), "
          f"validation_sem.csv ({len(val_rows)} rows), numbers_sem.json")
    print(f"  reproduction self-check: max |G_exact_repro - G_msg_original| = "
          f"{nums['repro_max_abs_diff_vs_original_msg']} "
          f"over {nums['repro_cells_compared']} cells")
    print(f"  out-of-sample: {nholds}/{len(val_rows)} cells hold "
          f"(mean margin {nums['validation_mean_margin']:+.3f})")


if __name__ == "__main__":
    main()
