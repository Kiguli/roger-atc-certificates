"""Fetch + extract LiveATC held-out audio (US airports, disjoint from ATCO2)
from the public HuggingFace mirror nyuuzyou/liveatc. Extracts audio clips to
heldout/clips/ and reports durations. Audio stays local (LiveATC ToS: no
redistribution) -- only the resulting certificate CSV is analysed/committed.
"""
import io, os, sys, tarfile
from pathlib import Path
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
BASE = os.environ.get("HELDOUT_DIR", "heldout")
OUT = HERE / BASE / "clips"
OUT.mkdir(parents=True, exist_ok=True)

# a few busy US airports (disjoint from ATCO2's mostly-European feeds)
ARCHIVES = [a for a in sys.argv[1:]] or ["kbos", "kbdl", "kbna", "kaus"]

try:
    import zstandard as zstd
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "zstandard"])
    import zstandard as zstd

import soundfile as sf

dctx = zstd.ZstdDecompressor()
durs = []
for a in ARCHIVES:
    try:
        fp = hf_hub_download("nyuuzyou/liveatc", f"data/{a}.tar.zst",
                             repo_type="dataset", local_dir=str(HERE / BASE / "dl"))
    except Exception as e:
        print(f"  {a}: download failed ({e})"); continue
    with open(fp, "rb") as f:
        raw = dctx.stream_reader(f).read()
    tf = tarfile.open(fileobj=io.BytesIO(raw))
    members = [m for m in tf.getmembers()
               if m.name.lower().endswith((".mp3", ".wav")) and m.isfile()]
    print(f"{a}: {len(members)} audio files")
    for m in members:
        data = tf.extractfile(m).read()
        stem = f"{a}__{Path(m.name).name}"
        (OUT / stem).write_bytes(data)
        try:
            info = sf.info(io.BytesIO(data))
            durs.append(info.duration)
        except Exception:
            pass

import numpy as np
if durs:
    d = np.array(durs)
    print(f"\nEXTRACTED {len(list(OUT.glob('*')))} clips to {OUT}")
    print(f"durations (s): n={len(d)} min={d.min():.1f} med={np.median(d):.1f} "
          f"mean={d.mean():.1f} max={d.max():.1f}")
    print(f"  <=30s: {(d<=30).sum()}   30-120s: {((d>30)&(d<=120)).sum()}   "
          f">120s: {(d>120).sum()}")
else:
    print("no durations read")
