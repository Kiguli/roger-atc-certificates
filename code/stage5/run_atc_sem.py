"""Semantic-equivalence oracle for the ATC split campaign (floors 0.90 / 0.95).

The ATC campaign stored per-draw violation BITS, not transcripts, so the
semantic spec cannot be derived post hoc -- the audio has to be decoded again.
Two facts make that affordable:

  1. A draw whose transcript is UNCHANGED has cosine 1.0 against the clean
     transcript and so can never violate a semantic floor.  The stored
     `bits_msg` says exactly which draws changed, so only those need decoding
     (4.3% of draws for Whisper-ATC/ATCOSIM, 47.5% over the whole campaign).
  2. The perturbation stream is deterministic (crc32 seeds and a fixed batch
     schedule), so the SAME draws are reproduced and ksem/kexact become a
     paired comparison on one sample instead of two independent ones.

Fact 2 is *verified, not assumed*.  For the first utterance of every combo the
runner decodes ALL N draws and checks the derived change-bits against the
stored ones; the clean decode is checked against the stored clean transcript
for every utterance.  On any disagreement that combo falls back to decoding
every draw -- correct regardless of stream reproduction -- and says so loudly
in the log.  Fallback is per-utterance, so one bad utterance cannot silently
corrupt the rest.

GPU safety (this machine has lost runs to memory pressure before):
  * decode batches are adaptive: an OOM halves the batch, empties the cache
    and retries, and the batch recovers slowly afterwards;
  * the NOISE schedule is never touched by that backoff, so the perturbation
    stream stays reproducible whatever the memory pressure does;
  * free VRAM is checked before each utterance, with a cache empty + short
    wait if it is low;
  * expandable_segments keeps fragmentation from creeping up over a long run;
  * system sleep is blocked for the lifetime of the run (Windows idle-slept a
    previous campaign).

Checkpointing: one atomic fsync'd append per utterance (all its SNRs); the
resume scan counts an utterance done only if every SNR row is present at this
N with intact bitstrings -- the convention of run_atc_split.py.  Unique
transcripts are also appended (gzip) so any FUTURE oracle is free.

Output: results_sem_<model>_<dataset>.csv, one row per (file, snr):
  model,dataset,role,file,snr,N,beta,clean_msg,kexact,ksem_90,ksem_95,
  n_decoded,n_uniq,min_cos,mode,bits_sem90,bits_sem95

Run:  python run_atc_sem.py <qn|ds2|wh|whatc> <atcosim|atco2>
                            <certificate|ablation> [N]
"""
import os
# must precede `import torch` to take effect
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import csv                                              # noqa: E402
import gzip                                             # noqa: E402
import json                                             # noqa: E402
import sys                                              # noqa: E402
import time                                             # noqa: E402
import zlib                                             # noqa: E402
from collections import defaultdict                     # noqa: E402
from pathlib import Path                                # noqa: E402

import numpy as np                                      # noqa: E402
import soundfile as sf                                  # noqa: E402
import torch                                            # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "Stage-2-verification" / "scenario"))

import run_scenario_datasets as RS                      # noqa: E402
import run_atc_split as RA                              # noqa: E402

FS, SNRS, DEV = RS.FS, RS.SNRS, RS.DEV
BLBUDGET = RS.BLBUDGET
FLOORS = (0.90, 0.95)                    # the two floors the paper reports
EMB_BATCH = int(os.environ.get("EMB_BATCH", 256))
VRAM_FLOOR_MB = float(os.environ.get("VRAM_FLOOR_MB", 2000))

csv.field_size_limit(1 << 24)


def _is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or (
        isinstance(e, RuntimeError) and "out of memory" in str(e).lower())


class AdaptiveDecoder:
    """Decode with an OOM-backoff batch size, independent of the noise schedule."""

    def __init__(self, decode, batch0):
        self.decode, self.batch0, self.batch = decode, batch0, batch0
        self.good, self.ooms = 0, 0

    def __call__(self, wavs):
        out, i, n = [], 0, wavs.shape[0]
        while i < n:
            b = min(self.batch, n - i)
            try:
                out.extend(self.decode(wavs[i:i + b]))
            except Exception as e:                       # noqa: BLE001
                if not _is_oom(e):
                    raise
                self.ooms += 1
                self.good = 0
                torch.cuda.empty_cache()
                if self.batch <= 1:
                    raise
                self.batch = max(1, self.batch // 2)
                print(f"    [mem] OOM -> decode batch {self.batch}", flush=True)
                continue
            i += b
            self.good += 1
            if self.good >= 64 and self.batch < self.batch0:
                self.batch = min(self.batch0, self.batch * 2)
                self.good = 0
                print(f"    [mem] decode batch recovered to {self.batch}",
                      flush=True)
        return out


def load_embedder():
    """all-MiniLM-L6-v2, exactly as the LibriSpeech semantic study used it."""
    from sentence_transformers import SentenceTransformer
    em = SentenceTransformer("all-MiniLM-L6-v2", device=str(DEV))
    em.eval()

    @torch.no_grad()
    def embed(texts):
        return em.encode(texts, convert_to_tensor=True,
                         normalize_embeddings=True, batch_size=EMB_BATCH,
                         show_progress_bar=False, device=str(DEV))
    return embed


def load_stored(name, dataset, role, N):
    """(file,snr) -> (bits_msg, clean_msg) from the committed campaign CSV."""
    src = HERE / f"results_atc_{name}_{dataset}.csv"
    if not src.exists():
        sys.exit(f"missing {src.name}: run the message-spec campaign first")
    out = {}
    for r in csv.DictReader(open(src, newline="")):
        if r["role"] != role:
            continue
        try:
            if int(r["N"]) != N or len(r["bits_msg"]) != N:
                continue
        except (TypeError, ValueError):
            continue
        out[(r["file"], int(r["snr"]))] = (r["bits_msg"], r["clean_msg"])
    return out


def wait_for_vram(tag=""):
    if DEV.type != "cuda":
        return
    for attempt in range(30):
        free = torch.cuda.mem_get_info()[0] / 2**20
        if free >= VRAM_FLOOR_MB:
            return
        torch.cuda.empty_cache()
        print(f"    [mem] only {free:.0f} MiB free {tag}; waiting "
              f"({attempt + 1}/30)", flush=True)
        time.sleep(10)


def run(which, dataset, role, N, beta):
    if os.name == "nt":                       # block idle-sleep for the run
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

    name, decode = RA.make_decoder(which)
    batch_sz = int(os.environ.get("BATCH", RA.BATCH_DEFAULT[name]))
    dec = AdaptiveDecoder(decode, batch_sz)
    embed = load_embedder()
    stored = load_stored(name, dataset, role, N)
    files = RA.load_split(dataset, role)
    mmax = int(os.environ.get("ATC_MMAX", 0))
    if mmax:
        files = files[:mmax]

    out = HERE / f"results_sem_{name}_{dataset}.csv"
    txt = HERE / f"texts_sem_{name}_{dataset}.jsonl.gz"
    fields = ["model", "dataset", "role", "file", "snr", "N", "beta",
              "clean_msg", "kexact", "ksem_90", "ksem_95", "n_decoded",
              "n_uniq", "min_cos", "mode", "bits_sem90", "bits_sem95"]
    done_files = set()
    if out.exists():
        per = defaultdict(set)
        for r in csv.DictReader(open(out, newline="")):
            try:
                if (int(r["N"]) == N and len(r["bits_sem90"]) == N
                        and len(r["bits_sem95"]) == N):
                    per[r["file"]].add(int(r["snr"]))
            except (TypeError, ValueError):
                continue
        done_files = {f for f, s in per.items() if s >= set(SNRS)}
    fresh = not out.exists() or out.stat().st_size == 0
    fh = open(out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=fields)
    if fresh:
        w.writeheader(); fh.flush()

    todo = [f for f in files if f not in done_files]
    print(f"[sem {name} / {dataset} / {role}] {len(files)} files "
          f"({len(todo)} to do), N={N}, floors={FLOORS}, dev={DEV}, "
          f"decode batch={batch_sz}", flush=True)

    verified = None          # None = not yet checked; True = stream reproduces
    consec_fail, t_start, n_dec_tot = 0, time.time(), 0

    for fi, fn in enumerate(todo):
        try:
            wait_for_vram(f"before {fn}")
            wav, sr = sf.read(
                str(ROOT / "external/stt/data" / dataset / fn), dtype="float32")
            assert sr == FS, f"{fn}: sr={sr} != {FS}"
            wavg = torch.tensor(wav, dtype=torch.float32, device=DEV)
            m0 = dec(wavg.unsqueeze(0))[0]
            e0 = embed([m0])[0]

            # the noise schedule MUST match the original run exactly
            eff = max(8, min(batch_sz, int(BLBUDGET / max(len(wav), 1))))
            Lpad = RS._next_fast_len(len(wav))
            freqs = np.fft.rfftfreq(Lpad, 1 / FS)
            fmask = torch.tensor(
                ~((freqs < RS.PASSBAND[0]) | (freqs > RS.PASSBAND[1])),
                device=DEV)

            # first utterance of the combo: decode everything and verify
            check = verified is None
            rows, uniq_dump, t0 = [], [], time.time()

            for snr in SNRS:
                bits_msg, clean_stored = stored.get((fn, snr), (None, None))
                # decode-all whenever we cannot trust the stored bits
                full = (check or verified is False or bits_msg is None
                        or clean_stored != m0)
                gen = torch.Generator(device=DEV)
                gen.manual_seed(zlib.crc32(
                    f"{name}|{dataset}|{fn}|{snr}".encode()))

                idxs, buf, texts, done = [], [], [], 0
                while done < N:
                    b = min(eff, N - done)
                    pb = RS.perturb_batch_gpu(wavg, snr, b, gen, fmask, Lpad)
                    if full:
                        loc = list(range(b))
                    else:
                        loc = [j for j in range(b) if bits_msg[done + j] == "1"]
                    if loc:
                        sel = pb.index_select(
                            0, torch.tensor(loc, device=DEV)).clone()
                        buf.append(sel)
                        idxs.extend(done + j for j in loc)
                    del pb
                    done += b
                    if buf and sum(x.shape[0] for x in buf) >= dec.batch:
                        chunk = torch.cat(buf); buf = []
                        texts.extend(dec(chunk))
                        del chunk
                if buf:
                    chunk = torch.cat(buf); buf = []
                    texts.extend(dec(chunk))
                    del chunk
                n_dec_tot += len(texts)

                # cosine against the clean transcript, deduped
                cos = {}
                uq = sorted({t for t in texts if t != m0})
                if uq:
                    E = embed(uq)
                    sims = (E @ e0).tolist()
                    cos = dict(zip(uq, sims))
                b90 = ["0"] * N
                b95 = ["0"] * N
                kexact = 0
                mn = 1.0
                for gi, t in zip(idxs, texts):
                    if t == m0:
                        continue
                    kexact += 1
                    c = cos[t]
                    mn = min(mn, c)
                    if c < FLOORS[0]:
                        b90[gi] = "1"
                    if c < FLOORS[1]:
                        b95[gi] = "1"

                if check:            # stream-reproduction verification
                    derived = "".join("1" if (gi in set(idxs) and t != m0)
                                      else "0"
                                      for gi, t in zip(range(N), texts)) \
                        if False else "".join(
                            "1" if t != m0 else "0" for t in texts)
                    ok = (bits_msg is not None and derived == bits_msg)
                    print(f"    [verify] {fn} {snr}dB stream-reproduction: "
                          f"{'EXACT' if ok else 'MISMATCH'}"
                          f" (kexact {kexact} vs stored "
                          f"{bits_msg.count('1') if bits_msg else '?'})",
                          flush=True)
                    if not ok:
                        verified = False

                rows.append(dict(
                    model=name, dataset=dataset, role=role, file=fn, snr=snr,
                    N=N, beta=beta, clean_msg=m0, kexact=kexact,
                    ksem_90=b90.count("1"), ksem_95=b95.count("1"),
                    n_decoded=len(texts), n_uniq=len(uq),
                    min_cos=round(mn, 4),
                    mode="full" if full else "selective",
                    bits_sem90="".join(b90), bits_sem95="".join(b95)))
                uniq_dump.append({"file": fn, "snr": snr,
                                  "uniq": {t: texts.count(t) for t in uq}})

            if check and verified is None:
                verified = True
                print(f"  [verify] {fn}: stream reproduces exactly -- "
                      f"switching to SELECTIVE decode for the rest of "
                      f"{name}/{dataset}/{role}", flush=True)
            elif check:
                print(f"  [verify] {fn}: MISMATCH -- staying in FULL decode "
                      f"for {name}/{dataset}/{role} (correct, just slower)",
                      flush=True)

            w.writerows(rows); fh.flush(); os.fsync(fh.fileno())
            with gzip.open(txt, "at", encoding="utf-8") as tf:
                for rec in uniq_dump:
                    tf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            consec_fail = 0
            if DEV.type == "cuda":
                torch.cuda.empty_cache()

            r = rows[len(rows) // 2]
            el = time.time() - t_start
            rate = n_dec_tot / max(el, 1e-9)
            print(f"  [{fi + 1}/{len(todo)}] {fn} @{r['snr']}dB "
                  f"exact={r['kexact']} sem90={r['ksem_90']} "
                  f"sem95={r['ksem_95']} ({time.time() - t0:.0f}s; "
                  f"{rate:.0f} dec/s; {el / 3600:.2f}h elapsed)", flush=True)

        except Exception as e:                            # noqa: BLE001
            print(f"  !! [{fi + 1}/{len(todo)}] {fn} FAILED "
                  f"({type(e).__name__}: {e}) -- skipping", flush=True)
            consec_fail += 1
            try:
                if DEV.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass
            if consec_fail >= 10:
                fh.close()
                sys.exit(f"{consec_fail} consecutive failures -- aborting "
                         f"{name}/{dataset}/{role} (resume to retry)")
            continue

    fh.close()
    print(f"wrote {out.name}  ({n_dec_tot:,} decodes, "
          f"{(time.time() - t_start) / 3600:.2f}h, {dec.ooms} OOM backoffs)",
          flush=True)


if __name__ == "__main__":
    which, dataset, role = sys.argv[1], sys.argv[2], sys.argv[3]
    assert which in ("qn", "ds2", "wh", "whatc")
    assert dataset in ("atcosim", "atco2")
    assert role in ("certificate", "ablation")
    N = int(sys.argv[4]) if len(sys.argv) > 4 else int(os.environ.get("N", 1000))
    run(which, dataset, role, N, float(os.environ.get("BETA", 1e-6)))
