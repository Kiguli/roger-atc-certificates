"""Held-out certification run for Whisper-ATC on out-of-fine-tuning ATC audio.

Reuses the *exact* threat model, call-sign extractor, seeding, and per-draw
recording of run_atc_split.py (band-limited 300-2500 Hz Gaussian noise at the
SNRs in $SNRS, N draws, CP+), but reads audio from an arbitrary directory of
.wav files (resampled to 16 kHz, mono) instead of the ATCOSIM/ATCO2 manifest.
The point is a memorization control: Whisper-ATC was fine-tuned on ATCOSIM+ATCO2
but NOT on this audio, so its certified fractions here test noise-robustness
rather than memorization.

Output (same schema as run_atc_split): results_atc_whisper_atc_<tag>.csv
Run:  SNRS=60,40,20 python run_heldout.py <audio_dir> [N] [tag]
"""
import csv, os, sys, time, zlib
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import scipy.signal as ss
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_atc_split as R                       # noqa: E402  make_decoder, RS
RS = R.RS
FS, DEV, BLBUDGET, SNRS = RS.FS, RS.DEV, RS.BLBUDGET, RS.SNRS
CAP = int(os.environ.get("HELDOUT_BATCH", 32))   # decode batch cap (VRAM)
csv.field_size_limit(1 << 24)


def load_wav_16k(path):
    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != FS:
        g = gcd(FS, int(sr))
        wav = ss.resample_poly(wav, FS // g, int(sr) // g).astype("float32")
    m = float(np.max(np.abs(wav))) if wav.size else 0.0
    if m > 1.0:                                  # int-decoded / hot; normalize
        wav = wav / m
    return np.ascontiguousarray(wav, dtype="float32")


def run(indir, N, beta, tag):
    name, decode = R.make_decoder("whatc")       # jlvdoorn/whisper-large-v3-atco2-asr-atcosim
    files = sorted(Path(indir).glob("*.wav"))
    out = HERE / f"results_atc_whisper_atc_{tag}.csv"
    fieldnames = ["model", "dataset", "role", "file", "snr", "N", "beta",
                  "k_msg", "k_cs", "clean_msg", "clean_cs", "bits_msg", "bits_cs"]
    done = set()
    if out.exists():
        for r in csv.DictReader(open(out, newline="")):
            try:
                if int(r["N"]) == N and len(r["bits_msg"]) == N:
                    done.add((r["file"], int(r["snr"])))
            except (TypeError, ValueError):
                continue
    fresh = not out.exists() or out.stat().st_size == 0
    fh = open(out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=fieldnames)
    if fresh:
        w.writeheader(); fh.flush()
    on_gpu = DEV.type != "cpu"
    print(f"[whisper_atc / {tag}] {len(files)} files, N={N}, SNRs={SNRS}, dev={DEV}",
          flush=True)
    for fi, fp in enumerate(files):
        fn = fp.name
        if all((fn, snr) in done for snr in SNRS):
            continue
        try:
            wav = load_wav_16k(fp)
            if wav.size < FS // 2:               # < 0.5 s -> skip junk
                print(f"  [{fi+1}] {fn}: too short ({wav.size})"); continue
            m0 = decode(torch.tensor(wav)[None].to(DEV))[0]
            cs0 = RS.callsign(m0)
            eff_batch = CAP
            if on_gpu:
                eff_batch = max(8, min(CAP, int(BLBUDGET / max(len(wav), 1))))
                wavg = torch.tensor(wav, dtype=torch.float32, device=DEV)
                Lpad = RS._next_fast_len(len(wav))
                freqs = np.fft.rfftfreq(Lpad, 1 / FS)
                fmask = torch.tensor(
                    ~((freqs < RS.PASSBAND[0]) | (freqs > RS.PASSBAND[1])), device=DEV)
            rows = []
            for snr in SNRS:
                if (fn, snr) in done:
                    continue
                t0 = time.time()
                seed = zlib.crc32(f"{name}|{tag}|{fn}|{snr}".encode())
                if on_gpu:
                    gen = torch.Generator(device=DEV); gen.manual_seed(seed)
                else:
                    rng = np.random.default_rng(seed)
                vm, vc, d = [], [], 0
                while d < N:
                    b = min(eff_batch, N - d)
                    if on_gpu:
                        pb = RS.perturb_batch_gpu(wavg, snr, b, gen, fmask, Lpad)
                    else:
                        pb = torch.tensor(RS.perturb_batch(
                            wav.astype(np.float64), snr, b, rng)).to(DEV)
                    for s in decode(pb):
                        vm.append("1" if s != m0 else "0")
                        vc.append("1" if RS.callsign(s) != cs0 else "0")
                    d += b
                rows.append(dict(model=name, dataset=tag, role="certificate",
                                 file=fn, snr=snr, N=N, beta=beta,
                                 k_msg=vm.count("1"), k_cs=vc.count("1"),
                                 clean_msg=m0, clean_cs=cs0,
                                 bits_msg="".join(vm), bits_cs="".join(vc)))
                print(f"  [{fi+1}/{len(files)}] {fn} {snr}dB: msg {vm.count('1')}/{N}"
                      f" cs {vc.count('1')}/{N} ({time.time()-t0:.0f}s)", flush=True)
            if rows:
                w.writerows(rows); fh.flush(); os.fsync(fh.fileno())
            if on_gpu:
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  !! [{fi+1}] {fn} FAILED ({type(e).__name__}: {e})", flush=True)
            continue
    fh.close()
    print("wrote", out, flush=True)


if __name__ == "__main__":
    indir = sys.argv[1]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("N", 1000))
    tag = sys.argv[3] if len(sys.argv) > 3 else "heldout"
    run(indir, N, float(os.environ.get("BETA", 1e-6)), tag)
