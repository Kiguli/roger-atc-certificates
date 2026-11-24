"""Did scaling a held-out corpus from 90 to 250 clips change the conclusion?

Splits a corpus into the ORIGINAL clips (those recorded in heldout_manifest.csv,
which was generated before the scale-up) and the NEW clips added afterwards, then
compares certified fractions. Two questions:

  (1) sampling check -- are the new clips consistent with the originals, or did
      the deeper segmentation pull in systematically different audio?
  (2) estimate stability -- how far did the reported number move, relative to
      its own confidence interval?

Two-proportion z-test for (1); Wilson intervals throughout.

Usage: python compare_scaleup.py <results_csv> [label]
"""
import csv, sys
from math import sqrt
from pathlib import Path

HERE = Path(__file__).resolve().parent
csv.field_size_limit(1 << 24)
N, KSTAR = 1000, 20
SNRS = [60, 40, 20]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def ztest(k1, n1, k2, n2):
    if not n1 or not n2:
        return float("nan")
    p = (k1 + k2) / (n1 + n2)
    se = sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0
    return (k1 / n1 - k2 / n2) / se


def main():
    resfn = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else resfn
    manifest = HERE / "heldout_manifest.csv"
    tag = Path(resfn).stem.replace("results_atc_whisper_atc_", "")
    orig = set()
    if manifest.exists():
        for r in csv.DictReader(open(manifest, encoding="utf-8")):
            if r["file"] and r.get("corpus"):
                orig.add((r["corpus"], r["file"]))
    # match on filename only, restricted to this corpus's rows in the manifest
    corp = {c for c, _ in orig}
    files_by_corpus = {c: {f for cc, f in orig if cc == c} for c in corp}

    rows = [r for r in csv.DictReader(open(HERE / resfn, newline=""))
            if int(r["N"]) == N]
    allf = {r["file"] for r in rows}
    # the corpus whose manifest filenames overlap these results
    best, bestov = None, 0
    for c, fs in files_by_corpus.items():
        ov = len(fs & allf)
        if ov > bestov:
            best, bestov = c, ov
    ofiles = files_by_corpus.get(best, set()) & allf
    nfiles = allf - ofiles
    print(f"{label}: {len(allf)} clips total = {len(ofiles)} original + {len(nfiles)} new"
          f"  (matched against manifest corpus '{best}')\n")
    print(f"{'SNR':>4} {'spec':>4} | {'original':>22} | {'new':>22} | {'combined':>22} | z    move")
    print("-" * 108)
    for s in SNRS:
        for spec in ("msg", "cs"):
            key = f"k_{spec}"
            o = [int(r[key]) for r in rows if int(r["snr"]) == s and r["file"] in ofiles]
            nw = [int(r[key]) for r in rows if int(r["snr"]) == s and r["file"] in nfiles]
            ko, kn = sum(1 for v in o if v <= KSTAR), sum(1 for v in nw if v <= KSTAR)
            no, nn = len(o), len(nw)
            kc, nc = ko + kn, no + nn
            z = ztest(ko, no, kn, nn)
            lo_o, hi_o = wilson(ko, no)
            lo_n, hi_n = wilson(kn, nn)
            lo_c, hi_c = wilson(kc, nc)
            move = (kc / nc - ko / no) if no and nc else float("nan")
            flag = "  <-- differs" if abs(z) > 1.96 else ""
            print(f"{s:>4} {spec:>4} | {ko/no if no else float('nan'):.2f} [{lo_o:.2f},{hi_o:.2f}] n={no:<3} "
                  f"| {kn/nn if nn else float('nan'):.2f} [{lo_n:.2f},{hi_n:.2f}] n={nn:<3} "
                  f"| {kc/nc if nc else float('nan'):.2f} [{lo_c:.2f},{hi_c:.2f}] n={nc:<3} "
                  f"| {z:+.2f} {move:+.3f}{flag}")
    print("\nz = two-proportion z-test, original vs new clips (|z|>1.96 => differ at 5%).")
    print("move = combined estimate minus original-only estimate (how far the paper number shifts).")


if __name__ == "__main__":
    main()
