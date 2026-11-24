"""Reviewer 3CgC control: repeated decoding with ZERO added noise.

Runs the campaign pipeline of run_atc_split.py with the noise generator
switched off, so any "violation" can only come from inference variability
(GPU nondeterminism, batch-size-dependent kernels), never from the channel.

Per utterance (all 1,500 ATCOSIM / 857 ATCO2 messages, both split roles):

  clean0      the campaign's clean decode: batch of 1, decoded first
  rep_b1      R1 further batch-1 decodes of the same clean audio
              (run-to-run variability at the clean-decode batch size)
  zero draws  N0 "draws" with delta = 0, decoded in EXACTLY the campaign's
              batch schedule (eff_batch = max(8, min(BATCH, BLBUDGET/L)),
              final partial batch included), same precision (fp32 CTC with
              PyTorch defaults, fp16 Whisper)
  mixed       one campaign-size batch whose first and last rows are the
              clean audio and whose other rows are the campaign's own first
              60 dB noise draws (same crc32 seed): does a row's decode
              depend on what else is in the batch?

Every decode is kept as RAW text, so both normalizations can be applied:
  old  = RS.norm_text           (campaign: keeps [a-z' ], deletes digits)
  new  = norm_text_digits       (revision: numerals spoken digit-by-digit)

Output: zeronoise_<model>_<dataset>.csv, one row per utterance.
Run:    python run_zeronoise.py <qn|ds2|wh|whatc> <atcosim|atco2> [N0]
        ZN_LIMIT=<k> processes only the first k utterances of a fixed
        pseudo-random order (crc32 of the file name), so any prefix is a
        random subset; reruns resume per utterance.
"""
import csv
import json
import os
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "Stage-2-verification" / "scenario"))
sys.path.insert(0, str(HERE))

import run_scenario_datasets as RS                      # noqa: E402
import run_atc_split as RA                              # noqa: E402
from atc_text import norm_text_digits                   # noqa: E402

FS, DEV, BLBUDGET = RS.FS, RS.DEV, RS.BLBUDGET
R1 = int(os.environ.get("ZN_R1", 2))
# ZN_NO_TF32=1: disable cuDNN TF32 convolutions (PyTorch enables them by
# default) and request deterministic kernels -- tests whether batch-size-
# dependent decodes come from reduced-precision convolutions.
NO_TF32 = os.environ.get("ZN_NO_TF32", "0") == "1"
if NO_TF32:
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
csv.field_size_limit(1 << 24)


def make_raw_decoder(which):
    """Same models, precision and decode call as RA.make_decoder, but the
    decode returns RAW text (normalization applied by the caller)."""
    if which in ("qn", "ds2"):
        import batched_models as bm
        from radiostar.stt_models import DeepSpeech2, QuartzNet15x5, ctc_greedy
        name = "quartznet" if which == "qn" else "deepspeech2"
        mdl = QuartzNet15x5() if which == "qn" else DeepSpeech2()
        mdl = RS.to_dev(mdl, name)
        labels = (mdl.labels + ["_"]) if which == "qn" else mdl.LABELS
        blank = mdl.blank if which == "qn" else mdl.BLANK
        lf = bm.qn_logits_batch if which == "qn" else bm.ds2_logits_batch

        def decode(wavs):
            lg = lf(mdl, wavs).cpu()
            return [ctc_greedy(lg[j], labels, blank) for j in range(lg.shape[0])]
        return name, decode
    import inspect
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
    return ("whisper" if which == "wh" else "whisper_atc"), decode


def specs(raw):
    o = RS.norm_text(raw)
    n = norm_text_digits(raw)
    return o, RS.callsign(o), n, RS.callsign(n)


def main():
    which, dataset = sys.argv[1], sys.argv[2]
    N0 = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    name, decode = make_raw_decoder(which)
    batch_sz = int(os.environ.get("BATCH", RA.BATCH_DEFAULT[name]))
    if name == "whisper_atc":
        # cost control (1.54B decode ~29/s): 8 full batches of 32 plus one
        # remainder batch of 8 -- the campaign's batch STRUCTURE (31x32 + 8)
        # at about a quarter of the decodes
        N0 = min(N0, int(os.environ.get("ZN_WHATC_N0", 264)))
    files = (RA.load_split(dataset, "certificate")
             + RA.load_split(dataset, "ablation"))
    files.sort(key=lambda f: zlib.crc32(f"{dataset}|{f}".encode()))
    lim = int(os.environ.get("ZN_LIMIT", 0))
    if lim:
        files = files[:lim]
    out = HERE / (f"zeronoise_{name}_{dataset}"
                  f"{'_notf32' if NO_TF32 else ''}.csv")
    fields = ["model", "dataset", "file", "eff_batch", "N0", "clean_raw",
              "b1_rep_diff_raw", "k0_raw", "k0_msg_old", "k0_cs_old",
              "k0_msg_new", "k0_cs_new", "mixed_diff_raw", "mixed_diff_msg_old",
              "mixed_diff_msg_new", "n_uniq", "uniq_json", "sec"]
    done = set()
    if out.exists():
        # drop a torn last row left by a killed run (wrong field count)
        rows = list(csv.reader(open(out, newline="", encoding="utf-8")))
        if rows and any(len(r) != len(fields) for r in rows[1:]):
            with open(out, "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerows(
                    [rows[0]] + [r for r in rows[1:] if len(r) == len(fields)])
            print(f"[repair] dropped torn rows from {out.name}", flush=True)
        done = {r["file"] for r in csv.DictReader(open(out, newline="",
                                                       encoding="utf-8"))
                if r.get("N0") == str(N0)}
    fresh = not out.exists() or out.stat().st_size == 0
    fh = open(out, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=fields)
    if fresh:
        w.writeheader(); fh.flush()
    todo = [f for f in files if f not in done]
    print(f"[zeronoise {name}/{dataset}] {len(files)} files, {len(todo)} to "
          f"do, N0={N0}, batch={batch_sz}, R1={R1}", flush=True)
    t_all = time.time()
    for fi, fn in enumerate(todo):
        t0 = time.time()
        wav, sr = sf.read(str(ROOT / "external/stt/data" / dataset / fn),
                          dtype="float32")
        assert sr == FS
        wavg = torch.tensor(wav, dtype=torch.float32, device=DEV)
        clean = decode(wavg[None])[0]                      # campaign clean0
        o0, c0, n0, cn0 = specs(clean)
        rep = [decode(wavg[None])[0] for _ in range(R1)]
        eff = max(8, min(batch_sz, int(BLBUDGET / max(len(wav), 1))))
        texts, done_n = [], 0
        while done_n < N0:                                 # campaign schedule
            b = min(eff, N0 - done_n)
            texts.extend(decode(wavg[None].expand(b, -1).contiguous()))
            done_n += b
        # mixed batch: clean rows at both ends, campaign 60 dB draws between
        Lpad = RS._next_fast_len(len(wav))
        freqs = np.fft.rfftfreq(Lpad, 1 / FS)
        fmask = torch.tensor(~((freqs < RS.PASSBAND[0]) |
                               (freqs > RS.PASSBAND[1])), device=DEV)
        gen = torch.Generator(device=DEV)
        gen.manual_seed(zlib.crc32(f"{name}|{dataset}|{fn}|60".encode()))
        mb = max(eff, 2)
        pb = RS.perturb_batch_gpu(wavg, 60, mb, gen, fmask, Lpad)
        pb[0] = wavg; pb[-1] = wavg
        mt = decode(pb)
        mixed = [mt[0], mt[-1]]
        k = dict(raw=0, mo=0, co=0, mn=0, cn=0)
        for t in texts:
            o, c, n, cn = specs(t)
            k["raw"] += t != clean
            k["mo"] += o != o0
            k["co"] += c != c0
            k["mn"] += n != n0
            k["cn"] += cn != cn0
        uq = Counter(texts)
        w.writerow(dict(
            model=name, dataset=dataset, file=fn, eff_batch=eff, N0=N0,
            clean_raw=clean, b1_rep_diff_raw=sum(r != clean for r in rep),
            k0_raw=k["raw"], k0_msg_old=k["mo"], k0_cs_old=k["co"],
            k0_msg_new=k["mn"], k0_cs_new=k["cn"],
            mixed_diff_raw=sum(m != clean for m in mixed),
            mixed_diff_msg_old=sum(RS.norm_text(m) != o0 for m in mixed),
            mixed_diff_msg_new=sum(norm_text_digits(m) != n0 for m in mixed),
            n_uniq=len(uq),
            uniq_json=json.dumps(dict(uq), ensure_ascii=False)
            if len(uq) > 1 else "",
            sec=round(time.time() - t0, 2)))
        fh.flush()
        if fi % 25 == 0 or k["raw"]:
            el = time.time() - t_all
            print(f"  [{fi + 1}/{len(todo)}] {fn} eff={eff} k0_raw={k['raw']} "
                  f"msg_old={k['mo']} cs_old={k['co']} msg_new={k['mn']} "
                  f"b1rep={sum(r != clean for r in rep)} "
                  f"mixed={sum(m != clean for m in mixed)} "
                  f"({time.time() - t0:.1f}s; {el / 60:.1f} min)", flush=True)
        if DEV.type == "cuda":
            torch.cuda.empty_cache()
    fh.close()
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
