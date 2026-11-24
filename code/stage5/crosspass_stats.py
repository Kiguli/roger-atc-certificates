"""Cross-pass decode drift for the models whose semantic counts come from a
second pass (DeepSpeech2, Whisper-ATC): per (message, SNR) cell, compare the
first-pass exact count k_msg (results_atc_*) with the second-pass re-decoded
exact count kexact (results_sem_*).  The second pass re-decodes only draws
that violated in the first pass, so the comparison is one-sided.

Output: crosspass_stats.json (read by emit_revision_macros.py)
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KSTAR = 20
csv.field_size_limit(1 << 24)


def main():
    diffs, flips, cells = [], 0, 0
    per = {}
    for m in ("deepspeech2", "whisper_atc"):
        for ds in ("atcosim", "atco2"):
            k1 = {}
            for r in csv.DictReader(open(HERE / f"results_atc_{m}_{ds}.csv", newline="")):
                if r["N"] == "1000" and len(r["bits_msg"]) == 1000:
                    k1[(r["file"], r["snr"])] = int(r["k_msg"])
            n = ch = fl = 0
            for r in csv.DictReader(open(HERE / f"results_sem_{m}_{ds}.csv", newline="")):
                key = (r["file"], r["snr"])
                if key not in k1:
                    continue
                a, b = k1[key], int(r["kexact"])
                d = abs(a - b)
                diffs.append(d)
                n += 1
                ch += d > 0
                fl += (a <= KSTAR) != (b <= KSTAR)
            per[f"{m}_{ds}"] = dict(cells=n, changed=ch, flips=fl)
            cells += n
            flips += fl
    diffs.sort()
    changed = sum(d > 0 for d in diffs)
    out = dict(cells=cells, changed=changed,
               changed_pct=round(100 * changed / cells, 1),
               p99=diffs[int(0.99 * (len(diffs) - 1))], max=diffs[-1],
               flips=flips, flips_pct=round(100 * flips / cells, 2), per=per)
    (HERE / "crosspass_stats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
