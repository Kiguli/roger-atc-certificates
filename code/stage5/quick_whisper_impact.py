"""Headline numbers for the Whisper models after the reruns (before the full
rev1 pipeline): certified fractions under the batch-1 reference (the paper's
definition) and under the batched reference (clean audio decoded in the draws'
own batch configuration), plus Whisper-base WER with numerals kept.

Output: quick_whisper_impact.json
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

KSTAR, SNRS = 20, ("60", "50", "45", "40", "35", "30", "25", "20")
csv.field_size_limit(1 << 24)


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i]
        for j, wb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (wa != wb)))
        prev = cur
    return prev[-1]


out = {}
# ---------------- whisper-base, v2 rerun
for ds in ("atcosim", "atco2"):
    refs = {r["file"]: norm_text_digits(r["text"]) for r in csv.DictReader(
        open(ROOT / "external/stt/data" / ds / "refs.csv", encoding="utf-8"))}
    cert = {ref: {sp: Counter() for sp in ("msg", "cs")} for ref in ("b1", "beff")}
    n = 0
    wer = []
    disagree = 0
    with gzip.open(HERE / f"results_v2_whisper_{ds}.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            n += 1
            c_b1 = norm_text_digits(r["clean_b1"])
            beff_raw = max(r["clean_beff"].items(), key=lambda kv: kv[1])[0]
            c_beff = norm_text_digits(beff_raw)
            disagree += c_b1 != c_beff
            ref_words = refs[r["file"]].split()
            wer.append(lev(c_b1.split(), ref_words) / max(len(ref_words), 1))
            for snr in SNRS:
                d = r["snr"][snr]
                cnt = Counter(d["idx"])
                texts = {norm_text_digits(d["uniq"][i]): 0 for i in cnt}
                for i, c in cnt.items():
                    texts[norm_text_digits(d["uniq"][i])] += c
                for ref, c0 in (("b1", c_b1), ("beff", c_beff)):
                    cs0 = RS.callsign(c0)
                    k_msg = sum(c for t, c in texts.items() if t != c0)
                    k_cs = sum(c for t, c in texts.items() if RS.callsign(t) != cs0)
                    cert[ref]["msg"][snr] += k_msg <= KSTAR
                    cert[ref]["cs"][snr] += k_cs <= KSTAR
    old = {sp: Counter() for sp in ("msg", "cs")}
    for r in csv.DictReader(open(HERE / f"results_atc_whisper_{ds}.csv", newline="")):
        if r["snr"] in SNRS and r["N"] == "1000":
            old["msg"][r["snr"]] += int(r["k_msg"]) <= KSTAR
            old["cs"][r["snr"]] += int(r["k_cs"]) <= KSTAR
    f = lambda c: {s: round(c[s] / n, 3) for s in SNRS}
    out[f"whisper_{ds}"] = dict(
        n=n, wer_new=round(sum(wer) / n, 4), clean_b1_vs_beff_disagree=disagree,
        old_campaign={sp: f(old[sp]) for sp in old},
        new_b1={sp: f(cert["b1"][sp]) for sp in ("msg", "cs")},
        new_beff={sp: f(cert["beff"][sp]) for sp in ("msg", "cs")})

# ---------------- Whisper-ATC: effect of the batched reference (semantic pass)
for ds in ("atcosim", "atco2"):
    zn = {r["file"]: r for r in csv.DictReader(open(HERE / f"zeronoise_whisper_atc_{ds}.csv", encoding="utf-8"))}
    beff = {}
    for fn, r in zn.items():
        if r["uniq_json"]:
            u = json.loads(r["uniq_json"])
            beff[fn] = RS.norm_text(max(u.items(), key=lambda kv: kv[1])[0])
    clean = {}
    for r in csv.DictReader(open(HERE / f"results_sem_whisper_atc_{ds}.csv", newline="")):
        clean[(r["file"], r["snr"])] = (r["clean_msg"], int(r["N"]))
    n_msgs = len({f for f, _ in clean})
    cnt = {ref: Counter() for ref in ("b1", "beff")}
    changed = Counter()
    with gzip.open(HERE / f"texts_sem_whisper_atc_{ds}.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            snr = str(rec["snr"])
            if snr not in SNRS:
                continue
            m0, N = clean[(rec["file"], snr)]
            uniq = rec["uniq"]
            unchanged = N - sum(uniq.values())
            k_b1 = N - unchanged
            ref = beff.get(rec["file"], m0)
            k_be = N - (unchanged if ref == m0 else uniq.get(ref, 0))
            cnt["b1"][snr] += k_b1 <= KSTAR
            cnt["beff"][snr] += k_be <= KSTAR
            changed[snr] += (k_b1 <= KSTAR) != (k_be <= KSTAR)
    # messages absent from texts_sem at an SNR had no changed draws (k = 0)
    for snr in SNRS:
        seen = cnt["b1"][snr] + 0
    out[f"whisper_atc_{ds}"] = dict(
        n=n_msgs, affected_messages=len(beff),
        note="semantic pass; exact spec; messages with no changed draws at an SNR are not listed in texts_sem",
        cert_b1_listed=dict(cnt["b1"]), cert_beff_listed=dict(cnt["beff"]),
        decisions_changed=dict(changed))

(HERE / "quick_whisper_impact.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
