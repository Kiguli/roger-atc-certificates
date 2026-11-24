"""Segment long LiveATC recording blocks into individual transmissions (utterance
clips) with a simple energy VAD, resampled to 16 kHz mono. Writes wavs to
heldout/seg/ up to a target count. Local use only (LiveATC ToS)."""
import csv, os, sys, warnings
from math import gcd
from pathlib import Path
import numpy as np
import soundfile as sf
import scipy.signal as ss

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
BASE = os.environ.get("HELDOUT_DIR", "heldout")
SRC = HERE / BASE / "clips"
OUT = HERE / BASE / "seg"
OUT.mkdir(parents=True, exist_ok=True)
FS = 16000
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 120


def load16k(fp):
    x, sr = sf.read(str(fp), dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != FS:
        g = gcd(FS, int(sr)); x = ss.resample_poly(x, FS // g, int(sr) // g)
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return (x / m).astype("float32") if m > 0 else x


def vad(x, sr=FS, hop=0.02, thr_db=8.0, min_sil=0.30, min_dur=1.5,
        max_dur=15.0, pad=0.10):
    w = int(hop * sr)
    n = len(x) // w
    if n < 3:
        return []
    e = np.sqrt(np.array([np.mean(x[i * w:(i + 1) * w] ** 2) for i in range(n)]) + 1e-10)
    edb = 20 * np.log10(e + 1e-8)
    floor = np.percentile(edb, 20)
    voiced = edb > (floor + thr_db)
    # fill short silence gaps
    gap = int(min_sil / hop)
    i = 0
    while i < n:
        if not voiced[i]:
            j = i
            while j < n and not voiced[j]:
                j += 1
            if (j - i) < gap and 0 < i and j < n:
                voiced[i:j] = True
            i = j
        else:
            i += 1
    segs, i = [], 0
    while i < n:
        if voiced[i]:
            j = i
            while j < n and voiced[j]:
                j += 1
            a = max(0, int((i * hop - pad) * sr))
            b = min(len(x), int((j * hop + pad) * sr))
            dur = (b - a) / sr
            if min_dur <= dur <= max_dur:
                segs.append((a, b))
            i = j
        else:
            i += 1
    return segs


def main():
    files = sorted(SRC.glob("*.mp3")) + sorted(SRC.glob("*.wav"))
    kept = 0
    # Sidecar index: lets anyone cut byte-identical clips from the source
    # recording without re-running this VAD (offsets are in the 16 kHz timebase
    # the source is resampled to). Written alongside the clips.
    idx_rows = []
    for fp in files:
        if kept >= TARGET:
            break
        try:
            x = load16k(fp)
        except Exception as e:
            print(f"  skip {fp.name}: {e}"); continue
        segs = vad(x)
        for k, (a, b) in enumerate(segs):
            if kept >= TARGET:
                break
            clip = x[a:b]
            if np.sqrt(np.mean(clip ** 2)) < 0.01:      # drop near-silent
                continue
            name = f"h{kept:03d}_{fp.stem[:24]}_{k}.wav"
            sf.write(str(OUT / name), clip, FS, subtype="PCM_16")
            idx_rows.append({"file": name, "source_file": fp.name,
                             "seg_in_source": k,
                             "start_sample_16k": a, "end_sample_16k": b,
                             "start_s": round(a / FS, 3), "end_s": round(b / FS, 3)})
            kept += 1
        print(f"{fp.name}: {len(segs)} segs -> total kept {kept}")
    with open(OUT.parent / "segment_index.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "source_file", "seg_in_source",
                                           "start_sample_16k", "end_sample_16k",
                                           "start_s", "end_s"])
        w.writeheader(); w.writerows(idx_rows)
    print(f"wrote {OUT.parent / 'segment_index.csv'} ({len(idx_rows)} rows)")
    durs = [sf.info(str(f)).duration for f in OUT.glob("*.wav")]
    if durs:
        d = np.array(durs)
        print(f"\nWROTE {len(d)} clips to {OUT}")
        print(f"durations (s): min={d.min():.1f} med={np.median(d):.1f} "
              f"mean={d.mean():.1f} max={d.max():.1f} total={d.sum()/60:.1f} min")


if __name__ == "__main__":
    main()
