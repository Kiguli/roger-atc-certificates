"""Fetch UWB-ATCC (Jzuluaga/uwb_atcc): curated real Czech ATC utterances, disjoint
from ATCOSIM/ATCO2 (NOT in Whisper-ATC fine-tuning -- van Doorn trained only on
ATCO2-ASR + ATCOSIM; see github.com/jlvdoorn/WhisperATC). Already utterance-
segmented with gold transcripts, so no VAD needed. It is NOISY real 8 kHz VHF
radio, i.e. an ATCO2-equivalent, in-region (European), out-of-sample held-out --
NOT a clean ATCOSIM-equivalent. Saves 16 kHz clips to heldout_uwb/seg/.

Bypasses torchcodec: casts the audio column with decode=False and reads the raw
container bytes with soundfile (libsndfile), so no torchcodec/ffmpeg is required.
"""
import io, os
from math import gcd
from pathlib import Path
import numpy as np, soundfile as sf, scipy.signal as ss
from datasets import load_dataset, Audio

HERE = Path(__file__).resolve().parent
OUT = HERE / "heldout_uwb" / "seg"
OUT.mkdir(parents=True, exist_ok=True)
FS = 16000
LIMIT = int(os.environ.get("UWB_LIMIT", 120))


def read_clip(a):
    """a is a decode=False audio dict: {'path':..., 'bytes':...}. Return float32 mono."""
    data = a.get("bytes")
    if data is not None:
        x, sr = sf.read(io.BytesIO(data), dtype="float32")
    else:
        p = a.get("path")
        if not p or not os.path.exists(p):
            return None, None
        x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype("float32"), int(sr)


kept = 0
for split in ("test", "train"):
    if kept >= LIMIT:
        break
    try:
        ds = load_dataset("Jzuluaga/uwb_atcc", split=split, streaming=True)
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception as e:
        print(f"  {split}: {type(e).__name__}: {e}")
        continue
    for r in ds:
        if kept >= LIMIT:
            break
        try:
            x, sr = read_clip(r["audio"])
        except Exception as e:
            print(f"  read fail: {type(e).__name__}: {e}")
            continue
        if x is None or len(x) == 0:
            continue
        txt = (r.get("text") or r.get("transcription") or "")
        if len(txt.split()) < 3:
            continue
        if sr != FS:
            g = gcd(FS, sr)
            x = ss.resample_poly(x, FS // g, sr // g).astype("float32")
        dur = len(x) / FS
        if not (1.5 <= dur <= 15):
            continue
        m = float(np.max(np.abs(x)))
        if m > 0:
            x = x / m
        sf.write(str(OUT / f"u{kept:03d}.wav"), x.astype("float32"), FS, subtype="PCM_16")
        kept += 1
print(f"wrote {kept} UWB-ATCC clips to {OUT}")
