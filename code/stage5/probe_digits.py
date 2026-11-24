"""Reviewer 3CgC probe: how often do the Whisper models emit numerals that
norm_text() silently deletes?

norm_text keeps only [a-z' ], so a raw decode "Emirates 81, good morning"
becomes "emirates good morning".  CTC models (QuartzNet, DeepSpeech2) have a
letters-only vocabulary and are unaffected.  This script decodes every clean
utterance once (batch 1, exactly the campaign's clean-decode path but keeping
the RAW text) for whisper-base and Whisper-ATC and records:
  raw text, norm_text(raw), whether raw contains a digit, and whether the
  digit-stripped normalized transcript lost any token.

Output: probe_digits_<model>.csv  (dataset,file,raw,norm,has_digit)
Run:    python probe_digits.py <wh|whatc> [heldout]
"""
import csv
import inspect
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "Stage-2-verification" / "scenario"))
import run_scenario_datasets as RS                      # noqa: E402

FS, DEV = RS.FS, RS.DEV


def make_raw_decoder(which):
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    mid = ("openai/whisper-base" if which == "wh"
           else "jlvdoorn/whisper-large-v3-atco2-asr-atcosim")
    proc = WhisperProcessor.from_pretrained(mid)
    wm = WhisperForConditionalGeneration.from_pretrained(mid).to(DEV).eval()
    if DEV.type == "cuda":
        wm = wm.half()
    gpu_mel = (DEV.type == "cuda" and "device" in inspect.signature(
        proc.feature_extractor.__call__).parameters)

    @torch.no_grad()
    def decode(wavs):
        arrs = list(wavs.detach().float().cpu().numpy())
        kw = dict(device="cuda") if gpu_mel else {}
        feats = proc(arrs, sampling_rate=FS, return_tensors="pt",
                     **kw).input_features.to(DEV)
        if DEV.type == "cuda":
            feats = feats.half()
        ids = wm.generate(feats, max_new_tokens=128, num_beams=1,
                          do_sample=False, language="en", task="transcribe")
        return proc.batch_decode(ids, skip_special_tokens=True)
    return decode


def main():
    which = sys.argv[1]
    heldout = len(sys.argv) > 2 and sys.argv[2] == "heldout"
    decode = make_raw_decoder(which)
    jobs = []
    if heldout:
        for r in csv.DictReader(open(HERE / "heldout_manifest.csv", newline="",
                                     encoding="utf-8")):
            jobs.append((r["corpus"], r["file"]))
        import run_heldout as RH                         # load_wav_16k
        dirs = {}
        for d in HERE.glob("heldout*"):
            if d.is_dir():
                for p in d.glob("*.wav"):
                    dirs.setdefault(p.name, []).append(p)
    else:
        for ds in ("atcosim", "atco2"):
            for p in sorted((ROOT / "external/stt/data" / ds).glob("*.wav")):
                jobs.append((ds, p.name))
    tag = which + ("_heldout" if heldout else "")
    out = HERE / f"probe_digits_{tag}.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "file", "raw", "norm", "has_digit"])
        for i, (ds, fn) in enumerate(jobs):
            if heldout:
                cands = dirs.get(fn, [])
                if len(cands) != 1:
                    continue
                wav = RH.load_wav_16k(cands[0])
            else:
                wav, sr = sf.read(str(ROOT / "external/stt/data" / ds / fn),
                                  dtype="float32")
                assert sr == FS
            raw = decode(torch.tensor(wav)[None].to(DEV))[0]
            w.writerow([ds, fn, raw, RS.norm_text(raw),
                        int(any(c.isdigit() for c in raw))])
            if i % 200 == 0:
                print(i, len(jobs), repr(raw), flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
