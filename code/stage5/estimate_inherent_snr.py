"""Inherent SNR of the certified recordings, and total SNR after perturbation.

The campaign's SNR s is an ADDED-noise SNR: noise power = P_x 10^(-s/10)
with P_x the mean power of the recording over the whole utterance (speech,
pauses and any noise already on the recording).  ATCO2 audio is a real VHF
channel recording, so it already carries channel noise; the total SNR the
model sees is lower than s.  This script makes that explicit with a simple,
documented blind estimate (no reference noise is available):

  * 20 ms frames, 10 ms hop; frame power in dB;
  * noise power  P_n  = mean power of the quietest 10% of frames;
  * active frames     = frames more than 6 dB above P_n;
  * speech power P_s  = mean power of active frames minus P_n;
  * inherent active-speech SNR  = 10 log10(P_s / P_n);
  * total SNR at added-noise s  = 10 log10(P_s / (P_n + P_x 10^(-s/10))),
    P_x = whole-utterance mean power (the campaign's scaling reference).

Rough by construction (short clips with few pauses bias P_n upward), so it is
reported as an estimate.  Output: inherent_snr.csv, inherent_snr.json
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "Stage-2-verification" / "scenario"))
import run_scenario_datasets as RS                      # noqa: E402

FS = 16000


def frames(x, n=320, h=160):
    if len(x) < n:
        return x[None]
    idx = np.arange(n)[None] + h * np.arange((len(x) - n) // h + 1)[:, None]
    return x[idx]


def main():
    rows, out = [], {}
    for ds in ("atcosim", "atco2"):
        tot = {s: [] for s in RS.SNRS}
        inh, act, peak = [], [], []
        for p in sorted((ROOT / "external/stt/data" / ds).glob("*.wav")):
            x, sr = sf.read(str(p), dtype="float64")
            assert sr == FS
            px = float(np.mean(x ** 2))
            fp = np.mean(frames(x) ** 2, axis=1) + 1e-12
            q = np.sort(fp)
            pn = float(np.mean(q[:max(1, len(q) // 10)]))
            a = fp > pn * 10 ** 0.6
            ps = float(np.mean(fp[a]) - pn) if a.any() else 0.0
            snr0 = 10 * np.log10(max(ps, 1e-12) / pn)
            r = dict(dataset=ds, file=p.name, inherent_snr_db=round(snr0, 2),
                     active_frac=round(float(a.mean()), 3),
                     peak=round(float(np.max(np.abs(x))), 4))
            for s in RS.SNRS:
                t = 10 * np.log10(max(ps, 1e-12) / (pn + px * 10 ** (-s / 10)))
                r[f"total_{s}"] = round(float(t), 2)
                tot[s].append(t)
            rows.append(r)
            inh.append(snr0); act.append(a.mean()); peak.append(np.max(np.abs(x)))
        pc = lambda v: {k: round(float(np.percentile(v, q)), 2)
                        for k, q in (("p10", 10), ("p50", 50), ("p90", 90))}
        out[ds] = dict(n=len(inh), inherent_snr_db=pc(inh),
                       active_frac=pc(act),
                       clipped_peak_ge_0_99=int(sum(p >= 0.99 for p in peak)),
                       total_snr_db={str(s): pc(v) for s, v in tot.items()})
    with open(HERE / "inherent_snr.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    (HERE / "inherent_snr.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
