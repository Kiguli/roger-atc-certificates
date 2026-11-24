"""Provenance manifest for every held-out clip that carries a certificate.

LiveATC audio cannot be redistributed (personal/non-commercial use only), so
instead of shipping the audio we ship everything needed to (a) re-derive it
byte-for-byte from the public source and (b) verify any clip a reviewer
reconstructs is the one we certified.

Per clip: SHA-256, duration, sample rate, the source archive member it was cut
from, the model's clean decode, and k_msg/k_cs at every scanned SNR.

Source for the six streamed feeds is the public HuggingFace mirror
nyuuzyou/liveatc (data/<feed>.tar.zst); UWB-ATCC is Jzuluaga/uwb_atcc
(CC BY-NC-SA 4.0) and IS redistributable.

Writes heldout_manifest.csv.
"""
import csv, hashlib
from pathlib import Path
import soundfile as sf

HERE = Path(__file__).resolve().parent
csv.field_size_limit(1 << 24)

# dir tag -> (paper label, results csv, source dataset, redistributable?)
SETS = [
    ("heldout_uwb", "UWB-ATCC",      "results_atc_whisper_atc_heldout_uwb.csv",
     "HuggingFace Jzuluaga/uwb_atcc (CC BY-NC-SA 4.0)", "yes"),
    ("heldout_eu",  "Poland",        "results_atc_whisper_atc_heldout_eu.csv",
     "HuggingFace nyuuzyou/liveatc", "no"),
    ("heldout_nl",  "Netherlands",   "results_atc_whisper_atc_heldout_nl.csv",
     "HuggingFace nyuuzyou/liveatc", "no"),
    ("heldout_no",  "Norway",        "results_atc_whisper_atc_heldout_no.csv",
     "HuggingFace nyuuzyou/liveatc", "no"),
    ("heldout_se",  "Sweden",        "results_atc_whisper_atc_heldout_se.csv",
     "HuggingFace nyuuzyou/liveatc", "no"),
    ("heldout_ca",  "Canada",        "results_atc_whisper_atc_heldout_ca.csv",
     "HuggingFace nyuuzyou/liveatc", "no"),
    ("heldout",     "United States", "results_atc_whisper_atc_heldout.csv",
     "HuggingFace nyuuzyou/liveatc", "no"),
]
SNRS = [60, 40, 20]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def source_member(fn):
    """h000_eham__EHAA-Radar-124230-_0.wav -> (feed, archive member stem, seg idx)."""
    stem = fn[:-4] if fn.endswith(".wav") else fn
    if "__" not in stem:
        return "", "", ""
    head, rest = stem.split("__", 1)
    feed = head.split("_", 1)[1] if "_" in head else ""
    blk, _, idx = rest.rpartition("_")
    return feed, blk, idx


rows = []
for tag, label, resfn, source, redist in SETS:
    good = HERE / tag / "good"
    res = HERE / resfn
    if not good.is_dir() or not res.exists():
        print(f"  skip {tag} (missing dir or results)")
        continue
    # offsets written by segment_heldout.py: cut the identical clip from the
    # source recording without re-running the VAD
    offs = {}
    idxf = HERE / tag / "segment_index.csv"
    if idxf.exists():
        for r in csv.DictReader(open(idxf, encoding="utf-8")):
            offs[r["file"]] = r
    ks = {}
    for r in csv.DictReader(open(res, newline="")):
        try:
            ks.setdefault(r["file"], {})[int(r["snr"])] = (int(r["k_msg"]), int(r["k_cs"]))
        except (TypeError, ValueError, KeyError):
            continue
    decode = {}
    for r in csv.DictReader(open(res, newline="")):
        decode.setdefault(r["file"], r.get("clean_msg", ""))
    for p in sorted(good.glob("*.wav")):
        if p.name not in ks:
            continue
        info = sf.info(str(p))
        feed, blk, idx = source_member(p.name)
        o = offs.get(p.name, {})
        row = {
            "corpus": label, "file": p.name, "sha256": sha256(p),
            "samplerate": info.samplerate, "duration_s": round(info.duration, 3),
            "source_dataset": source, "redistributable": redist,
            "source_feed": feed, "source_member": blk, "segment_index": idx,
            "source_file": o.get("source_file", ""),
            "start_s": o.get("start_s", ""), "end_s": o.get("end_s", ""),
            "start_sample_16k": o.get("start_sample_16k", ""),
            "end_sample_16k": o.get("end_sample_16k", ""),
            "clean_decode": decode.get(p.name, ""),
        }
        for s in SNRS:
            km, kc = ks[p.name].get(s, ("", ""))
            row[f"k_msg_{s}dB"] = km
            row[f"k_cs_{s}dB"] = kc
        rows.append(row)
    print(f"  {label:<14} {sum(1 for r in rows if r['corpus'] == label):>3} clips")

fields = ["corpus", "file", "sha256", "samplerate", "duration_s", "source_dataset",
          "redistributable", "source_feed", "source_member", "segment_index",
          "source_file", "start_s", "end_s", "start_sample_16k", "end_sample_16k",
          "clean_decode"] + [f"k_{a}_{s}dB" for s in SNRS for a in ("msg", "cs")]
out = HERE / "heldout_manifest.csv"
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
print(f"\nwrote {out}  ({len(rows)} clips)")
