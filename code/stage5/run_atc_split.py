"""ATC certificate/ablation runs with per-draw outcome recording.

Same threat model and inference path as scenario/run_scenario_datasets.py
(band-limited 300-2500 Hz Gaussian noise at fixed SNRs, batched QuartzNet /
DeepSpeech2 CTC greedy decode), with three changes for the split study:

  1. utterances come from split_manifest.csv (dataset x role), not a glob;
  2. seeds are deterministic (crc32 of "model|dataset|file|snr"), so a rerun
     reproduces the same perturbation stream (the June run used python
     hash(), which is salted per process);
  3. besides the violation counts k_msg/k_cs we record the PER-DRAW outcome
     bitstrings, so certificates can be recomputed post-hoc at any N' <= N
     (prefix of an i.i.d. stream is i.i.d.) and bootstrapped.

Output: results_atc_<model>_<dataset>.csv, one row per (file, snr):
  model,dataset,role,file,snr,N,beta,k_msg,k_cs,clean_msg,clean_cs,
  bits_msg,bits_cs
where bits_* are length-N strings of 0/1 (1 = violation on that draw).
Rows for one file (all SNRs) are appended atomically with fsync; a file
already holding all its SNR rows is skipped on resume. If a crash ever
leaves duplicate (file,snr) rows, analysis keeps the LAST occurrence.

Run:  python run_atc_split.py <qn|ds2|wh> <atcosim|atco2> <certificate|ablation> [N]

Models: qn = QuartzNet15x5, ds2 = DeepSpeech2 (batched CTC greedy, as in the
June run); wh = openai/whisper-base fp16 greedy decode, integrated exactly as
in libri/run_scenario_libri.py make_model("whisper") (HF weights are already
in the local cache). Whisper has no June ATC baseline, so the redraw ablation
skips it.
"""
import csv
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCEN = ROOT / "Stage-2-verification" / "scenario"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCEN))

import batched_models as bm                      # noqa: E402
import run_scenario_datasets as RS               # noqa: E402
from radiostar.stt_models import DeepSpeech2, QuartzNet15x5, ctc_greedy  # noqa: E402

FS = RS.FS
SNRS = RS.SNRS
DEV = RS.DEV
BLBUDGET = RS.BLBUDGET

csv.field_size_limit(1 << 24)                    # bits_* fields are N chars


def load_split(dataset, role):
    with open(HERE / "split_manifest.csv", newline="") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["dataset"] == dataset and r["role"] == role]
    rows.sort(key=lambda r: int(r["idx_shuffled"]))
    return [r["file"] for r in rows]


def make_decoder(which):
    """-> (name, decode) with decode: (B,L) float32 waveforms on DEV -> list
    of normalized transcripts. qn/ds2 mirror the June scenario run; whisper
    mirrors libri/run_scenario_libri.py make_model('whisper') verbatim."""
    if which in ("qn", "ds2"):
        name = "quartznet" if which == "qn" else "deepspeech2"
        mdl = QuartzNet15x5() if which == "qn" else DeepSpeech2()
        mdl = RS.to_dev(mdl, name)
        labels = (mdl.labels + ["_"]) if which == "qn" else mdl.LABELS
        blank = mdl.blank if which == "qn" else mdl.BLANK
        lf = bm.qn_logits_batch if which == "qn" else bm.ds2_logits_batch

        def decode(wavs):
            lg = lf(mdl, wavs).cpu()
            return [RS.norm_text(ctc_greedy(lg[j], labels, blank))
                    for j in range(lg.shape[0])]
        return name, decode
    if which in ("wh", "whatc"):
        import inspect
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        # whatc = van Doorn's Whisper-ATC: whisper-large-v3 fine-tuned on
        # ATCO2-ASR + ATCOSIM (trained on the corpora we certify -- the
        # deployed-in-domain configuration; disclosed in the writeup).
        mid = ("openai/whisper-base" if which == "wh"
               else "jlvdoorn/whisper-large-v3-atco2-asr-atcosim")
        proc = WhisperProcessor.from_pretrained(mid)
        wm = (WhisperForConditionalGeneration
              .from_pretrained(mid).to(DEV).eval())
        if DEV.type == "cuda":
            wm = wm.half()
        # transformers >= 4.36 can run the log-mel STFT on GPU (device=...):
        # measured 286 vs 167 samp/s on the 5090 with identical transcripts
        # (384/384 agreement probe vs the CPU-mel path used by libri).
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
            return [RS.norm_text(t)
                    for t in proc.batch_decode(ids, skip_special_tokens=True)]
        return ("whisper" if which == "wh" else "whisper_atc"), decode
    raise ValueError(which)


# ds2 raised 48->128 mid-campaign: with the desktop in use, WDDM submission
# contention starves DS2's many small BiLSTM timestep kernels (5s/cell vs
# June's 1s at ~50% GPU util); bigger batches halve launch rounds. Peak VRAM
# is still bounded by BLBUDGET (eff_batch = min(BATCH, BLBUDGET/L)), the
# June-proven ~13 GB budget, independent of BATCH.
BATCH_DEFAULT = {"quartznet": 256, "deepspeech2": 256, "whisper": 128,
                 "whisper_atc": 32}


def run(which, dataset, role, N, beta):
    if os.name == "nt":
        # block system sleep while the run lives (2026-07-17: Windows idle-
        # slept mid-campaign; GPU load does not count as activity). Cleared
        # automatically by the OS when the process exits.
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    name, decode = make_decoder(which)
    batch_sz = int(os.environ.get("BATCH", BATCH_DEFAULT[name]))
    files = load_split(dataset, role)
    mmax = int(os.environ.get("ATC_MMAX", 0))     # smoke-test cap, 0 = all
    if mmax:
        files = files[:mmax]
    tag = os.environ.get("ATC_TAG", "")           # e.g. "_smoke" for tests
    out = HERE / f"results_atc_{name}_{dataset}{tag}.csv"
    fieldnames = ["model", "dataset", "role", "file", "snr", "N", "beta",
                  "k_msg", "k_cs", "clean_msg", "clean_cs", "bits_msg", "bits_cs"]
    from collections import defaultdict
    done_files = set()
    if out.exists():
        # a file is done only if rows cover EVERY SNR at THIS N with intact
        # bitstrings -- a torn/short row or an old different-N row never
        # counts (post-review hardening; duplicates resolved last-wins later)
        per = defaultdict(set)
        for r in csv.DictReader(open(out, newline="")):
            try:
                if (int(r["N"]) == N and len(r["bits_msg"]) == N
                        and len(r["bits_cs"]) == N):
                    per[r["file"]].add(int(r["snr"]))
            except (TypeError, ValueError):
                continue                       # torn row -> not done
        done_files = {f for f, s in per.items() if s >= set(SNRS)}
    fresh = not out.exists() or out.stat().st_size == 0
    fh = open(out, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    if fresh:
        writer.writeheader(); fh.flush()
    on_gpu = DEV.type != "cpu"
    todo = [f for f in files if f not in done_files]
    print(f"[{name} / {dataset} / {role}] {len(files)} files "
          f"({len(todo)} to do), N={N}, dev={DEV}", flush=True)
    consec_fail = 0
    for fi, fn in enumerate(todo):
        try:
            wav, sr = sf.read(str(ROOT / "external/stt/data" / dataset / fn),
                              dtype="float32")
            assert sr == FS, f"{fn}: sr={sr} != {FS}"
            m0 = decode(torch.tensor(wav)[None].to(DEV))[0]
            cs0 = RS.callsign(m0)
            eff_batch = batch_sz
            if on_gpu:
                eff_batch = max(8, min(batch_sz, int(BLBUDGET / max(len(wav), 1))))
                wavg = torch.tensor(wav, dtype=torch.float32, device=DEV)
                Lpad = RS._next_fast_len(len(wav))
                freqs = np.fft.rfftfreq(Lpad, 1 / FS)
                fmask = torch.tensor(
                    ~((freqs < RS.PASSBAND[0]) | (freqs > RS.PASSBAND[1])),
                    device=DEV)
            file_rows = []
            for snr in SNRS:
                t0 = time.time()
                seed = zlib.crc32(f"{name}|{dataset}|{fn}|{snr}".encode())
                if on_gpu:
                    gen = torch.Generator(device=DEV); gen.manual_seed(seed)
                else:
                    rng = np.random.default_rng(seed)
                vm, vc = [], []                       # per-draw violation bits
                done = 0
                while done < N:
                    b = min(eff_batch, N - done)
                    if on_gpu:
                        pb = RS.perturb_batch_gpu(wavg, snr, b, gen, fmask, Lpad)
                    else:
                        pb = torch.tensor(RS.perturb_batch(
                            wav.astype(np.float64), snr, b, rng)).to(DEV)
                    for s in decode(pb):
                        vm.append("1" if s != m0 else "0")
                        vc.append("1" if RS.callsign(s) != cs0 else "0")
                    done += b
                k_msg, k_cs = vm.count("1"), vc.count("1")
                file_rows.append(dict(
                    model=name, dataset=dataset, role=role, file=fn, snr=snr,
                    N=N, beta=beta, k_msg=k_msg, k_cs=k_cs,
                    clean_msg=m0, clean_cs=cs0,
                    bits_msg="".join(vm), bits_cs="".join(vc)))
                print(f"  [{fi+1}/{len(todo)}] {fn} {snr}dB: "
                      f"msg {k_msg}/{N} cs {k_cs}/{N} ({time.time()-t0:.0f}s)",
                      flush=True)
            writer.writerows(file_rows); fh.flush(); os.fsync(fh.fileno())
            consec_fail = 0
            if on_gpu:
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  !! [{fi+1}/{len(todo)}] {fn} FAILED "
                  f"({type(e).__name__}: {e}) -- skipping", flush=True)
            consec_fail += 1
            if consec_fail >= 10:      # dead CUDA context / systematic error:
                fh.close()             # bail out; the combo resumes cleanly
                sys.exit(f"{consec_fail} consecutive failures -- aborting "
                         f"{name}/{dataset}/{role} (resume to retry)")
            try:
                if on_gpu and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass                   # cleanup on a poisoned context
            continue
    fh.close()
    print("wrote", out, flush=True)


if __name__ == "__main__":
    which, dataset, role = sys.argv[1], sys.argv[2], sys.argv[3]
    assert which in ("qn", "ds2", "wh", "whatc") and dataset in ("atcosim", "atco2")
    assert role in ("certificate", "ablation")
    N = int(sys.argv[4]) if len(sys.argv) > 4 else int(os.environ.get("N", 1000))
    beta = float(os.environ.get("BETA", 1e-6))
    run(which, dataset, role, N, beta)
