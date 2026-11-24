"""Revision campaign (TMLR, reviewer 3CgC): store RAW per-draw transcripts.

Identical threat model, seeds, batch schedule and decode path to
run_atc_split.py, with three changes:

  1. every draw's RAW transcript is stored (deduplicated per (file, SNR):
     the unique strings plus a per-draw index list), so every specification
     -- exact, call-sign, semantic floors, any normalization or extractor --
     is computed post hoc from ONE decode pass (analyze_atc_v2.py);
  2. V2_NO_TF32=1 disables cuDNN/cuBLAS TF32 and selects deterministic cuDNN
     kernels.  With PyTorch's default TF32 convolutions, a CTC model's decode
     of the SAME audio can differ between batch size 1 (the clean decode) and
     the campaign's batch of 256 (the draws); run_zeronoise.py measures this;
  3. besides the batch-1 clean decode, the clean audio is also decoded once in
     a full campaign-size batch (clean_beff), so certificates can be checked
     against a reference decoded under the draws' own batch configuration.

Output: results_v2_<model>_<dataset><tag>.jsonl.gz -- one JSON record per
utterance (gzip members appended per utterance; resume skips finished files):
  {model, dataset, role, file, eff_batch, no_tf32, clean_b1, clean_beff,
   snr: {"60": {"uniq": [...], "idx": [i_1..i_N]}, ...}}
where the transcript of draw j at that SNR is uniq[idx[j]].

Run:  python run_atc_v2.py <qn|ds2|wh|whatc> <atcosim|atco2> [N]
"""
import csv
import gzip
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
sys.path.insert(0, str(HERE))
import run_atc_split as RA                               # noqa: E402
import run_zeronoise as ZN                               # noqa: E402  raw decoders

RS = RA.RS
FS, DEV, BLBUDGET, SNRS = RS.FS, RS.DEV, RS.BLBUDGET, RS.SNRS

NO_TF32 = os.environ.get("V2_NO_TF32", "0") == "1"
if NO_TF32:
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _repair_gzip(path):
    """A run killed mid-append can leave a truncated last gzip member, which
    would hide every member appended after it.  Keep the complete members
    (records whose JSON parses), back up the original, rewrite."""
    import zlib
    raw = path.read_bytes()
    buf, data, clean = raw, b"", True
    while buf:
        d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            chunk = d.decompress(buf)
        except zlib.error:
            clean = False
            break
        if not d.eof:
            clean = False
            break
        data += chunk
        buf = d.unused_data
    lines = data.decode("utf-8", errors="replace").splitlines()
    good = []
    for ln in lines:
        try:
            json.loads(ln)
            good.append(ln)
        except json.JSONDecodeError:
            clean = False
    if clean:
        return
    bak = path.with_suffix(path.suffix + f".bak{int(time.time())}")
    bak.write_bytes(raw)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for ln in good:
            fh.write(ln + "\n")
    print(f"[repair] {path.name}: kept {len(good)} complete records "
          f"(backup {bak.name})", flush=True)


def main():
    which, dataset = sys.argv[1], sys.argv[2]
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    # Authors' decision (2026-10-08): DeepSpeech2 showed no TF32 zero-noise
    # effect, so its campaign certificates stand and its rerun is dropped.
    # The marker file lets the already-running queue skip that step.
    if (which == "ds2" and (HERE / "_skip_ds2_v2").exists()
            and not os.environ.get("V2_FILES")):          # targeted fixes still run
        print("[v2] DeepSpeech2 rerun skipped (_skip_ds2_v2 present)", flush=True)
        return
    if os.name == "nt":                       # block idle-sleep for the run
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    name, decode = ZN.make_raw_decoder(which)
    batch_sz = int(os.environ.get("BATCH", RA.BATCH_DEFAULT[name]))
    roles = {}
    for role in ("certificate", "ablation"):
        for f in RA.load_split(dataset, role):
            roles[f] = role
    files = list(roles)
    if int(os.environ.get("V2_LIMIT", 0)):                # smoke tests
        files = files[:int(os.environ["V2_LIMIT"])]
    if os.environ.get("V2_FILES"):                        # targeted reruns
        keep = set(os.environ["V2_FILES"].split(","))
        files = [f for f in files if f in keep]
    tag = os.environ.get("V2_TAG", "")
    base_out = HERE / f"results_v2_{name}_{dataset}{tag}.jsonl.gz"
    shard = os.environ.get("V2_SHARD", "")               # "i/n": parallel GPU shards
    if shard:
        si, sn = (int(v) for v in shard.split("/"))
        files = files[si::sn]
        out = HERE / f"results_v2_{name}_{dataset}{tag}_shard{si}of{sn}.jsonl.gz"
    else:
        out = base_out
    done = set()
    if shard and base_out.exists():                       # skip work done unsharded
        try:
            with gzip.open(base_out, "rt", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if len(rec.get("snr", {})) == len(SNRS):
                        done.add(rec["file"])
        except (EOFError, OSError):
            pass
    if out.exists():
        _repair_gzip(out)
        try:
            with gzip.open(out, "rt", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue                 # torn last line -> redo file
                    if len(rec.get("snr", {})) == len(SNRS):
                        done.add(rec["file"])
        except (EOFError, OSError):
            pass                                 # truncated gzip member
    todo = [f for f in files if f not in done]
    print(f"[v2 {name}/{dataset}] {len(files)} files ({len(todo)} to do), "
          f"N={N}, batch={batch_sz}, no_tf32={NO_TF32}, SNRS={SNRS}",
          flush=True)
    t_all, n_dec, consec_fail = time.time(), 0, 0
    for fi, fn in enumerate(todo):
        try:
            t0 = time.time()
            wav, sr = sf.read(str(ROOT / "external/stt/data" / dataset / fn),
                              dtype="float32")
            assert sr == FS
            wavg = torch.tensor(wav, dtype=torch.float32, device=DEV)
            clean_b1 = decode(wavg[None])[0]
            eff = max(8, min(batch_sz, int(BLBUDGET / max(len(wav), 1))))
            clean_beff = Counter(decode(wavg[None].expand(eff, -1).contiguous()))
            Lpad = RS._next_fast_len(len(wav))
            freqs = np.fft.rfftfreq(Lpad, 1 / FS)
            fmask = torch.tensor(~((freqs < RS.PASSBAND[0]) |
                                   (freqs > RS.PASSBAND[1])), device=DEV)
            rec = dict(model=name, dataset=dataset, role=roles[fn], file=fn,
                       eff_batch=eff, no_tf32=NO_TF32, N=N, clean_b1=clean_b1,
                       clean_beff=dict(clean_beff), snr={})
            for snr in SNRS:
                gen = torch.Generator(device=DEV)
                gen.manual_seed(zlib.crc32(f"{name}|{dataset}|{fn}|{snr}".encode()))
                texts, d = [], 0
                while d < N:
                    b = min(eff, N - d)
                    pb = RS.perturb_batch_gpu(wavg, snr, b, gen, fmask, Lpad)
                    texts.extend(decode(pb))
                    del pb
                    d += b
                n_dec += len(texts)
                uniq, idx, pos = [], [], {}
                for t in texts:
                    if t not in pos:
                        pos[t] = len(uniq); uniq.append(t)
                    idx.append(pos[t])
                rec["snr"][str(snr)] = dict(uniq=uniq, idx=idx)
            with gzip.open(out, "at", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            consec_fail = 0
            if DEV.type == "cuda":
                torch.cuda.empty_cache()
            el = time.time() - t_all
            k60 = sum(1 for i in rec["snr"][str(SNRS[0])]["idx"]
                      if rec["snr"][str(SNRS[0])]["uniq"][i] != clean_b1)
            print(f"  [{fi + 1}/{len(todo)}] {fn} eff={eff} "
                  f"k_raw@{SNRS[0]}={k60} beff_uniq={len(clean_beff)} "
                  f"({time.time() - t0:.0f}s; {n_dec / max(el, 1e-9):.0f} dec/s;"
                  f" {el / 3600:.2f}h; eta "
                  f"{el / (fi + 1) * (len(todo) - fi - 1) / 3600:.1f}h)",
                  flush=True)
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
                sys.exit(f"{consec_fail} consecutive failures -- aborting")
    print(f"wrote {out.name} ({n_dec:,} decodes, "
          f"{(time.time() - t_all) / 3600:.2f}h)", flush=True)


if __name__ == "__main__":
    main()
