"""Clean-decode filter: keep held-out clips whose Whisper-ATC clean transcript is a
plausible ATC utterance (3..40 words), so the held-out certification runs on real
transmissions rather than segmentation artifacts. Copies keepers to heldout/good/."""
import os, shutil, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_atc_split as R
RS = R.RS
BASE = os.environ.get("HELDOUT_DIR", "heldout")
SEG = HERE / BASE / "seg"
GOOD = HERE / BASE / "good"
GOOD.mkdir(parents=True, exist_ok=True)
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 90

name, decode = R.make_decoder("whatc")
files = sorted(SEG.glob("*.wav"))
kept, examples = 0, []
for fp in files:
    if kept >= TARGET:
        break
    try:
        wav, sr = sf.read(str(fp), dtype="float32")
        m0 = decode(torch.tensor(wav)[None].to(RS.DEV))[0]
    except Exception:
        continue
    nw = len(m0.split())
    if 3 <= nw <= 40:
        shutil.copy(str(fp), str(GOOD / fp.name))
        kept += 1
        if len(examples) < 12:
            examples.append((RS.callsign(m0), m0))
print(f"kept {kept}/{len(files)} clips -> {GOOD}")
for cs, m in examples:
    print(f"  cs='{cs}' :: {m[:75]}")
