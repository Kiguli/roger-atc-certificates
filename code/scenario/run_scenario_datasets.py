"""Task-A follow-up: scenario risk analysis over MANY messages, per dataset.

The two ATC corpora (ATCOSIM, ATCO2) are treated as SEPARATE i.i.d. message
distributions -- NOT combined -- so each yields its own model-level
robustness for that message *type*. For a (model, dataset) pair we glob ALL
available utterances, and for each draw N i.i.d. band-limited-noise
perturbations at each SNR, counting violations for two specs:

  message   transcript changes vs the clean decode
  callsign  the aircraft call-sign changes  (the "wrong aircraft acts on a
            clearance" hazard) -- extracted as the leading run of tokens
            before the first ATC command word (airline/phonetic + number).

Output: results_<model>_<dataset>.csv with one row per (utterance, SNR)
holding k_msg / k_cs. aggregate_datasets.py then turns these per-message
scores into (i) a joint single-level model risk and (ii) the two-level
'fraction-of-messages-robust' guarantee, per dataset.

This is the HEAVY run -- designed to be triggered on a powerful machine.
Configure via CLI/env; MPS/CUDA/CPU auto-selected and batched.

Run:  python run_scenario_datasets.py <qn|ds2> <atcosim|atco2> [N] [Mmax] [beta]
or:   bash run_all_heavy.sh    (all four model x dataset combos + aggregate)
"""
import csv
import glob
import os
import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
from radiostar.stt_models import DeepSpeech2, QuartzNet15x5, ctc_greedy  # noqa: E402
import batched_models as bm  # noqa: E402

FS = 16000
PASSBAND = (300.0, 2500.0)
SNRS = [int(s) for s in os.environ.get("SNRS", "60,50,45,40,35,30,25,20").split(",")]
# Cap the GPU batch by utterance length: DeepSpeech2 activation memory grows
# with T ~ L/320, so long utterances at full batch can exceed VRAM. eff_batch
# = min(BATCH, BLBUDGET / L) keeps peak bounded (~13 GB at the cap on a 32 GB
# card). Chunking does NOT change which perturbations are drawn (same RNG
# stream / row layout), so results are identical -- this is pure memory safety.
BLBUDGET = float(os.environ.get("BLBUDGET", 6e7))
DEV = (torch.device("cuda") if torch.cuda.is_available()
       else torch.device("mps") if torch.backends.mps.is_available()
       else torch.device("cpu"))

# ATC instruction verbs / keywords; the call-sign is everything before the first one.
COMMAND_WORDS = {
    "descend", "climb", "maintain", "turn", "taxi", "set", "cleared", "clear",
    "hold", "holding", "line", "expect", "reduce", "increase", "continue",
    "report", "squawk", "contact", "proceed", "fly", "join", "vacate",
    "cross", "stop", "start", "push", "pushback", "request", "confirm",
    "roger", "wind", "runway", "qnh", "ils", "approach", "departure",
    "direct", "resume", "speed", "heading", "level", "flight", "altitude",
    "via", "to", "and", "then", "after", "before", "is", "are", "will",
    "please", "now", "you", "your", "we", "affirm", "negative", "standby",
    "go", "ahead", "say", "again", "passing", "established", "leaving",
    "descending", "climbing", "turning", "approved", "unable", "wilco",
}


def norm_text(s):
    keep = "abcdefghijklmnopqrstuvwxyz' "
    return " ".join("".join(c if c in keep else " " for c in s.lower()).split())


def callsign(msg):
    """Leading run of tokens up to the first ATC command word (>=1 token)."""
    out = []
    for w in msg.split():
        if w in COMMAND_WORDS and out:
            break
        out.append(w)
        if len(out) >= 8:        # safety cap
            break
    return " ".join(out)


def _bl(d, lo, hi, f):
    D = np.fft.rfft(d, axis=-1); D[..., (f < lo) | (f > hi)] = 0
    return np.fft.irfft(D, n=d.shape[-1], axis=-1)


def perturb_batch(x, snr, n, rng):
    L = len(x); f = np.fft.rfftfreq(L, 1 / FS)
    d = _bl(rng.standard_normal((n, L)), *PASSBAND, f)
    px = np.sum(x ** 2); pd = np.sum(d ** 2, axis=1, keepdims=True) + 1e-20
    return (x[None] + d * np.sqrt(px / pd) * 10 ** (-snr / 20)).astype(np.float32)


def _next_fast_len(n):
    """Smallest power of two >= n. cuFFT is fast and uses bounded workspace on
    power-of-two lengths; native ATC lengths often have large prime factors
    (e.g. 144960 = 2^6*3*5*151) that push cuFFT onto Bluestein -- slow and a
    huge workspace that can exhaust VRAM and hard-crash the CUDA context."""
    return 1 << max(0, (int(n) - 1)).bit_length()


def perturb_batch_gpu(x, snr, n, gen, fmask, Lpad):
    """GPU equivalent of perturb_batch: passband-Gaussian perturbations scaled
    to the target SNR by signal power, randn + FFT on x.device. x: (L,) float32
    tensor; fmask: (Lpad//2+1,) in-band mask on the Lpad grid; Lpad: power-of-2
    FFT length (>= L). Noise is generated at Lpad then cropped to L so cuFFT
    stays on its fast, bounded-memory radix path. Same threat model as
    perturb_batch; only the RNG/length detail differs. Returns (n, L) float32."""
    L = x.shape[0]
    d = torch.randn(n, Lpad, generator=gen, device=x.device, dtype=torch.float32)
    D = torch.fft.rfft(d, dim=-1)
    D[:, ~fmask] = 0
    d = torch.fft.irfft(D, n=Lpad, dim=-1)[:, :L]
    xf = x.to(torch.float32)
    px = (xf * xf).sum()
    pd = (d * d).sum(dim=1, keepdim=True) + 1e-20
    return xf[None] + d * torch.sqrt(px / pd) * (10.0 ** (-snr / 20.0))


def load_dataset(dataset, mmax):
    d = ROOT / "external/stt/data" / dataset
    wavs = sorted(glob.glob(str(d / "*.wav")))
    if mmax:
        wavs = wavs[:mmax]
    return wavs


def to_dev(mdl, name):
    mdl.sd = {k: v.to(DEV) for k, v in mdl.sd.items()}
    mdl.window = mdl.window.to(DEV)
    if name == "quartznet":
        mdl.fb = mdl.fb.to(DEV)
    return mdl


def run(model, name, dataset, N, mmax, beta):
    labels = (model.labels + ["_"]) if name == "quartznet" else model.LABELS
    blank = model.blank if name == "quartznet" else model.BLANK
    lf = bm.qn_logits_batch if name == "quartznet" else bm.ds2_logits_batch
    model = to_dev(model, name)
    batch_sz = int(os.environ.get("BATCH", 256 if name == "quartznet" else 48))
    wavs = load_dataset(dataset, mmax)
    out = HERE / f"results_{name}_{dataset}.csv"
    fieldnames = ["model", "dataset", "file", "snr", "N", "beta", "k_msg", "k_cs"]
    # --- resume: a file fully recorded (all SNRs) is skipped; per-file atomic
    #     append + fsync means a crash loses at most the in-progress message ---
    from collections import Counter
    done_files = set()
    if out.exists():
        cnt = Counter(r["file"] for r in csv.DictReader(open(out)))
        done_files = {f for f, c in cnt.items() if c >= len(SNRS)}
    fresh = not out.exists()
    fh = open(out, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    if fresh:
        writer.writeheader(); fh.flush()
    on_gpu = DEV.type != "cpu"
    print(f"[{name} / {dataset}] {len(wavs)} messages, N={N}, dev={DEV}, "
          f"resuming with {len(done_files)} already done", flush=True)
    for fi, p in enumerate(wavs):
        fn = Path(p).name
        if fn in done_files:
            continue
        try:
            wav, sr = sf.read(p, dtype="float32")
            if sr != FS:                  # refs say 16k/32k orig; soundfile gives native
                wav = wav.astype(np.float32)
            lg0 = lf(model, torch.tensor(wav)[None].to(DEV))[0].cpu()
            m0 = norm_text(ctc_greedy(lg0, labels, blank)); cs0 = callsign(m0)
            eff_batch = batch_sz
            if on_gpu:                               # precompute GPU sampler state once/utterance
                eff_batch = max(8, min(batch_sz, int(BLBUDGET / max(len(wav), 1))))
                wavg = torch.tensor(wav, dtype=torch.float32, device=DEV)
                Lpad = _next_fast_len(len(wav))      # pow2 FFT len -> cuFFT fast/bounded
                freqs = np.fft.rfftfreq(Lpad, 1 / FS)
                fmask = torch.tensor(~((freqs < PASSBAND[0]) | (freqs > PASSBAND[1])), device=DEV)
            file_rows = []
            for snr in SNRS:
                t0 = time.time(); k_msg = k_cs = 0; done = 0
                seed = hash((name, dataset, fi, snr)) % (2**32)
                if on_gpu:
                    gen = torch.Generator(device=DEV); gen.manual_seed(seed)
                else:
                    rng = np.random.default_rng(seed)
                while done < N:
                    b = min(eff_batch, N - done)
                    if on_gpu:
                        pb = perturb_batch_gpu(wavg, snr, b, gen, fmask, Lpad)   # (b,L) f32 on DEV
                        lg = lf(model, pb).cpu()
                    else:
                        pb = perturb_batch(wav.astype(np.float64), snr, b, rng)
                        lg = lf(model, torch.tensor(pb).to(DEV)).cpu()
                    for j in range(b):
                        s = norm_text(ctc_greedy(lg[j], labels, blank))
                        if s != m0:
                            k_msg += 1
                        if callsign(s) != cs0:
                            k_cs += 1
                    done += b
                file_rows.append(dict(model=name, dataset=dataset, file=fn,
                                      snr=snr, N=N, beta=beta, k_msg=k_msg, k_cs=k_cs))
                print(f"  [{fi+1}/{len(wavs)}] {fn} {snr}dB: "
                      f"msg {k_msg}/{N} cs {k_cs}/{N} ({time.time()-t0:.0f}s)", flush=True)
            writer.writerows(file_rows); fh.flush(); os.fsync(fh.fileno())   # checkpoint this message
            if on_gpu:
                torch.cuda.empty_cache()             # release per-utterance peak; bound fragmentation
        except Exception as e:
            print(f"  !! [{fi+1}/{len(wavs)}] {fn} FAILED ({type(e).__name__}: {e}) -- skipping", flush=True)
            if on_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    fh.close()
    print("wrote", out, flush=True)


if __name__ == "__main__":
    which = sys.argv[1]; dataset = sys.argv[2]
    N = int(sys.argv[3]) if len(sys.argv) > 3 else int(os.environ.get("N", 1000))
    mmax = int(sys.argv[4]) if len(sys.argv) > 4 else int(os.environ.get("MMAX", 0)) or None
    beta = float(sys.argv[5]) if len(sys.argv) > 5 else float(os.environ.get("BETA", 1e-6))
    mdl = QuartzNet15x5() if which == "qn" else DeepSpeech2()
    run(mdl, "quartznet" if which == "qn" else "deepspeech2", dataset, N, mmax, beta)
