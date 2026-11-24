"""Aggregate Whisper-ATC per-message certificates across all corpora into the
cross-corpus comparison table used in the held-out results section.

Certified-robust <=> k <= k*(N,beta,tau), where k* is the largest k with
CP+(k,N,beta) = Beta_ppf(1-beta; k+1, N-k) <= tau. For N=1000, beta=1e-6,
tau=0.05 this is k*=20. Reports, per corpus and SNR, the fraction of files that
are certified-robust on the exact-message (msg) and call-sign (cs) specs.

Reads only what exists, so it can be run mid-campaign; missing corpora are noted.
"""
import csv, sys
from pathlib import Path
from collections import defaultdict
from scipy.stats import beta as betadist

HERE = Path(__file__).resolve().parent
csv.field_size_limit(1 << 24)
N, BETA, TAU = 1000, 1e-6, 0.05
SNRS = [60, 40, 20]


def kstar(N, b, tau):
    k = 0
    while k <= N and betadist.ppf(1 - b, k + 1, N - k) <= tau:
        k += 1
    return k - 1


KSTAR = kstar(N, BETA, TAU)

# label -> (csv filename, condition, region, in-training?)
CORPORA = [
    ("ATCOSIM",  "results_atc_whisper_atc_atcosim.csv",      "clean",  "EU",     True),
    ("ATCO2",    "results_atc_whisper_atc_atco2.csv",        "noisy",  "EU",     True),
    ("UWB-ATCC", "results_atc_whisper_atc_heldout_uwb.csv",  "noisy*", "EU/CZ",  False),
    ("Poland",   "results_atc_whisper_atc_heldout_eu.csv",   "noisy",  "EU",     False),
    ("NL",       "results_atc_whisper_atc_heldout_nl.csv",   "noisy",  "EU",     False),
    ("Norway",   "results_atc_whisper_atc_heldout_no.csv",   "noisy",  "EU",     False),
    ("Sweden",   "results_atc_whisper_atc_heldout_se.csv",   "noisy",  "EU",     False),
    ("Canada",   "results_atc_whisper_atc_heldout_ca.csv",   "noisy",  "NAM",    False),
    ("US",       "results_atc_whisper_atc_heldout.csv",      "noisy",  "NAM",    False),
]


def load(fn):
    """snr -> (n_files, n_cert_msg, n_cert_cs)."""
    agg = defaultdict(lambda: [0, 0, 0])
    p = HERE / fn
    if not p.exists():
        return None
    for r in csv.DictReader(open(p, newline="")):
        try:
            if int(r["N"]) != N:
                continue
            snr = int(r["snr"])
            km, kc = int(r["k_msg"]), int(r["k_cs"])
        except (TypeError, ValueError, KeyError):
            continue
        a = agg[snr]
        a[0] += 1
        a[1] += (km <= KSTAR)
        a[2] += (kc <= KSTAR)
    return agg


def main():
    print(f"k*(N={N}, beta={BETA}, tau={TAU}) = {KSTAR}   "
          f"(CP+({KSTAR})={betadist.ppf(1-BETA, KSTAR+1, N-KSTAR):.5f} <= {TAU} "
          f"< CP+({KSTAR+1})={betadist.ppf(1-BETA, KSTAR+2, N-KSTAR-1):.5f})\n")
    hdr = f"{'corpus':<10} {'cond':<7} {'reg':<6} {'train':<6} {'n':>4}"
    for s in SNRS:
        hdr += f"  msg@{s:<3} cs@{s:<3}"
    print(hdr)
    print("-" * len(hdr))
    for label, fn, cond, reg, intrain in CORPORA:
        agg = load(fn)
        if agg is None:
            print(f"{label:<10} {cond:<7} {reg:<6} {str(intrain):<6}   -- (not yet run)")
            continue
        nmax = max((agg[s][0] for s in SNRS), default=0)
        row = f"{label:<10} {cond:<7} {reg:<6} {str(intrain):<6} {nmax:>4}"
        for s in SNRS:
            n, cm, cc = agg[s]
            if n:
                row += f"  {cm/n:>6.2f}  {cc/n:>5.2f}"
            else:
                row += f"  {'  -':>6}  {'  -':>5}"
        print(row)
    print("\n* UWB-ATCC: 8 kHz upsampled; noisy real VHF; disjoint from Whisper-ATC "
          "fine-tuning (ATCO2-ASR+ATCOSIM only).")
    print("msg = exact-message certified fraction; cs = call-sign certified fraction.")


if __name__ == "__main__":
    main()
