"""Summarize the zero-noise control (run_zeronoise.py) against the campaign.

For each zeronoise_<model>_<dataset>[_notf32].csv:
  * share of messages with ANY zero-noise "violation" (decode of the
    unchanged audio, in the campaign's batches, differs from the batch-1
    clean decode), per readout (raw, msg, cs; old and digit-preserving
    normalization);
  * share whose zero-noise count alone exceeds k* (would fail certification
    with no channel noise at all), scaled to N=1000 when N0 < 1000;
  * batch-1 repeat disagreements and mixed-batch disagreements;
  * against the campaign's 60 dB certificates (results_atc_*): of the
    messages NOT certified at 60 dB, the share that also fail at zero noise
    (failure attributable to inference, not to the channel).

Output: zeronoise_summary.json / .csv
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KSTAR = 20                       # CP+(k,1000,1e-6) <= 0.05  <=>  k <= 20
csv.field_size_limit(1 << 24)
NAMES = {"quartznet": "qn", "deepspeech2": "ds2", "whisper": "wh",
         "whisper_atc": "whatc"}


def campaign(model, ds):
    out = {}
    p = HERE / f"results_atc_{model}_{ds}.csv"      # (_test = old smoke runs)
    for r in csv.DictReader(open(p, newline="")):
        if r["snr"] == "60" and r["N"] == "1000":
            out[r["file"]] = (int(r["k_msg"]), int(r["k_cs"]))
    return out


def main():
    rows = []
    for p in sorted(HERE.glob("zeronoise_*.csv")):
        if p.name == "zeronoise_summary.csv":
            continue
        stem = p.stem[len("zeronoise_"):]
        cfg = "notf32" if stem.endswith("_notf32") else "campaign"
        stem = stem.replace("_notf32", "")
        model, ds = stem.rsplit("_", 1)
        R = list(csv.DictReader(open(p, newline="", encoding="utf-8")))
        if not R:
            continue
        n = len(R)
        N0 = int(R[0]["N0"])
        scale = 1000 / N0
        camp = campaign(model, ds)
        row = dict(model=model, dataset=ds, config=cfg, n_msgs=n, N0=N0)
        for k in ("k0_raw", "k0_msg_old", "k0_cs_old", "k0_msg_new",
                  "k0_cs_new"):
            row[f"any_{k}"] = sum(int(r[k]) > 0 for r in R)
            row[f"fail_{k}"] = sum(int(r[k]) * scale > KSTAR for r in R)
        row["b1_repeat_diff"] = sum(int(r["b1_rep_diff_raw"]) > 0 for r in R)
        row["mixed_diff"] = sum(int(r["mixed_diff_raw"]) > 0 for r in R)
        row["draws_k0_raw"] = sum(int(r["k0_raw"]) for r in R)
        row["draws_total"] = n * N0
        # attribution against the campaign's 60 dB certificates
        for j, spec, kz in ((0, "msg", "k0_msg_old"), (1, "cs", "k0_cs_old")):
            fails = [r for r in R if r["file"] in camp
                     and camp[r["file"]][j] > KSTAR]
            row[f"camp60_{spec}_notcert"] = len(fails)
            row[f"camp60_{spec}_notcert_zero_fail"] = sum(
                int(r[kz]) * scale > KSTAR for r in fails)
            row[f"camp60_{spec}_notcert_zero_any"] = sum(
                int(r[kz]) > 0 for r in fails)
            row[f"camp60_{spec}_matched"] = sum(r["file"] in camp for r in R)
        rows.append(row)
        print(json.dumps(row))
    if rows:
        with open(HERE / "zeronoise_summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        (HERE / "zeronoise_summary.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
