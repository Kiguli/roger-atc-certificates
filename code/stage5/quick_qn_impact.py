"""QuartzNet: certified fractions with TF32 (campaign) vs TF32 off (v2), per
dataset, and what happens to the messages whose zero-noise decode changed under
TF32.  Datasets whose v2 rerun is incomplete are skipped.
Output: quick_qn_impact.json  {"<ds>_<snr>": {...}}
"""
import csv
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "Stage-2-verification" / "scenario"))
sys.path.insert(0, str(HERE))
import run_scenario_datasets as RS                      # noqa: E402
from atc_text import norm_text_digits                   # noqa: E402

KSTAR = 20
SNRS = ("60", "50", "45", "40", "35", "30", "25", "20")
FULL = {"atcosim": 1500, "atco2": 857}
csv.field_size_limit(1 << 24)
out = {}
for ds in ("atcosim", "atco2"):
    src = HERE / f"results_v2_quartznet_{ds}.jsonl.gz"
    if not src.exists():
        continue
    new = {}
    with gzip.open(src, "rt", encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            c0 = norm_text_digits(r["clean_b1"])
            cs0 = RS.callsign(c0)
            for snr in SNRS:
                d = r["snr"][snr]
                cnt = Counter(d["idx"])
                km = sum(c for i, c in cnt.items() if norm_text_digits(d["uniq"][i]) != c0)
                kc = sum(c for i, c in cnt.items() if RS.callsign(norm_text_digits(d["uniq"][i])) != cs0)
                new[(r["file"], snr)] = (km, kc)
    files = {f for f, _ in new}
    if len(files) < FULL[ds]:
        print(f"skip {ds}: {len(files)}/{FULL[ds]} messages")
        continue
    old = {}
    for r in csv.DictReader(open(HERE / f"results_atc_quartznet_{ds}.csv", newline="")):
        if r["N"] == "1000":
            old[(r["file"], r["snr"])] = (int(r["k_msg"]), int(r["k_cs"]))
    aff = {r["file"] for r in csv.DictReader(open(HERE / f"zeronoise_quartznet_{ds}.csv", encoding="utf-8"))
           if int(r["k0_raw"]) > 0}
    for snr in SNRS:
        o = [old[(f, snr)] for f in files]
        n = [new[(f, snr)] for f in files]
        out[f"{ds}_{snr}"] = dict(
            cert_msg_tf32=round(sum(k[0] <= KSTAR for k in o) / len(files), 3),
            cert_msg_notf32=round(sum(k[0] <= KSTAR for k in n) / len(files), 3),
            cert_cs_tf32=round(sum(k[1] <= KSTAR for k in o) / len(files), 3),
            cert_cs_notf32=round(sum(k[1] <= KSTAR for k in n) / len(files), 3),
            affected=len(aff),
            affected_cert_msg_tf32=sum(old[(f, snr)][0] <= KSTAR for f in aff),
            affected_cert_msg_notf32=sum(new[(f, snr)][0] <= KSTAR for f in aff))
print(json.dumps(out, indent=1))
(HERE / "quick_qn_impact.json").write_text(json.dumps(out, indent=2))
