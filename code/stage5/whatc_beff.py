"""Batched clean decode (batch of 32 and of 8, as in the campaign) for the
Whisper-ATC messages whose zero-noise decode changed, then the effect of
using it as the certification reference (exact and call-sign, semantic pass).
Output: whatc_beff.json
"""
import csv
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import run_zeronoise as ZN                              # noqa: E402
RS = ZN.RS
KSTAR = 20
csv.field_size_limit(1 << 24)

name, decode = ZN.make_raw_decoder("whatc")
out = {}
for ds in ("atcosim", "atco2"):
    aff = [r for r in csv.DictReader(open(HERE / f"zeronoise_whisper_atc_{ds}.csv", encoding="utf-8"))
           if int(r["k0_raw"]) > 0]
    beff = {}
    for r in aff:
        wav, sr = sf.read(str(ROOT / "external/stt/data" / ds / r["file"]), dtype="float32")
        w = torch.tensor(wav, device=ZN.DEV)
        d32 = Counter(decode(w[None].expand(32, -1).contiguous()))
        d8 = Counter(decode(w[None].expand(8, -1).contiguous()))
        beff[r["file"]] = dict(b32=RS.norm_text(d32.most_common(1)[0][0]),
                               b8=RS.norm_text(d8.most_common(1)[0][0]),
                               b1=RS.norm_text(r["clean_raw"]))
    clean = {(r["file"], r["snr"]): (r["clean_msg"], int(r["N"]))
             for r in csv.DictReader(open(HERE / f"results_sem_whisper_atc_{ds}.csv", newline=""))}
    res = []
    with gzip.open(HERE / f"texts_sem_whisper_atc_{ds}.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["file"] not in beff:
                continue
            m0, N = clean[(rec["file"], str(rec["snr"]))]
            uniq = rec["uniq"]
            unchanged = N - sum(uniq.values())
            texts = dict(uniq)
            texts[m0] = texts.get(m0, 0) + unchanged
            ref = beff[rec["file"]]["b32"]
            k_b1 = sum(c for t, c in texts.items() if t != m0)
            k_32 = sum(c for t, c in texts.items() if t != ref)
            c_b1 = sum(c for t, c in texts.items() if RS.callsign(t) != RS.callsign(m0))
            c_32 = sum(c for t, c in texts.items() if RS.callsign(t) != RS.callsign(ref))
            res.append(dict(file=rec["file"], snr=rec["snr"], k_b1=k_b1, k_b32=k_32,
                            cs_b1=c_b1, cs_b32=c_32))
    ch = Counter()
    for r in res:
        ch[(r["snr"], "msg")] += (r["k_b1"] <= KSTAR) != (r["k_b32"] <= KSTAR)
        ch[(r["snr"], "cs")] += (r["cs_b1"] <= KSTAR) != (r["cs_b32"] <= KSTAR)
    gained = Counter()
    for r in res:
        gained[(r["snr"], "msg")] += (r["k_b32"] <= KSTAR) and not (r["k_b1"] <= KSTAR)
        gained[(r["snr"], "cs")] += (r["cs_b32"] <= KSTAR) and not (r["cs_b1"] <= KSTAR)
    out[ds] = dict(affected=len(aff),
                   b32_equals_b8=sum(v["b32"] == v["b8"] for v in beff.values()),
                   b32_differs_b1=sum(v["b32"] != v["b1"] for v in beff.values()),
                   decisions_changed={f"{s}_{sp}": v for (s, sp), v in sorted(ch.items())},
                   newly_certified={f"{s}_{sp}": v for (s, sp), v in sorted(gained.items())})
    print(ds, json.dumps(out[ds]), flush=True)
(HERE / "whatc_beff.json").write_text(json.dumps(out, indent=2))
