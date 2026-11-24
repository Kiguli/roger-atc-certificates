"""Convert the revision campaign's raw-transcript dumps into the campaign's
CSV schemas, so the existing analysis scripts run unchanged on rev1/.

Input  (Stage-5-atc/, written by run_atc_v2.py, possibly still growing):
  results_v2_<model>_<ds>.jsonl.gz   one JSON record per utterance:
      {model, dataset, role, file, eff_batch, no_tf32, N, clean_b1 (RAW
       batch-1 clean decode), clean_beff ({raw: count} of the clean audio
       decoded in one campaign-size batch), snr: {"60": {uniq, idx}, ...}}
      draw j's RAW transcript at SNR s is uniq[idx[j]].  gzip members are
      appended per utterance; a torn last line / truncated member (job still
      writing) is skipped, and only records with all 8 SNRs are used
      (last record wins if a file appears twice).

Output (rev1/ by default; --out to change):
  results_atc_<model>_<ds>.csv   schema of run_atc_split.py
        model,dataset,role,file,snr,N,beta,k_msg,k_cs,clean_msg,clean_cs,
        bits_msg,bits_cs  + extra columns k_csid,bits_csid,clean_csid,clean_raw_json
      clean_msg = norm_text_digits(clean_b1); each draw's text is
      norm_text_digits(raw); msg bit = text != clean_msg; cs bit =
      RS.callsign(text) != RS.callsign(clean_msg); csid bit =
      callsign_grammar.identity(text) != identity(clean_msg).
  results_sem_<model>_<ds>.csv   schema of run_atc_sem.py
        model,dataset,role,file,snr,N,beta,clean_msg,kexact,ksem_90,ksem_95,
        n_decoded,n_uniq,min_cos,mode,bits_sem90,bits_sem95
      from the SAME pass: cosine (all-MiniLM-L6-v2, normalize_embeddings=True)
      between each changed normalized draw and the normalized clean text;
      kexact == k_msg; mode = "full" (every draw was decoded).
  texts_sem_<model>_<ds>.jsonl.gz  {"file","snr","uniq":{changed_text: count}}
  clean_batch_consistency.csv    per utterance: does the batch-1 clean decode
      differ from the mode of the campaign-batch clean decodes (normalized and
      raw), and how many SNRs change certified status (msg spec, k <= k*) if
      the draws are scored against the batch-mode reference instead.
  convert_manifest.json          sources, record counts, coverage vs the split
  Models WITHOUT a v2 dump (whisper_atc -- not rerun): the original
  results_atc_/results_sem_/texts_sem_ files are copied unchanged, and
  csid_<model>_<ds>.csv holds k_csid per (file, snr) computed from the
  SEMANTIC pass (texts_sem + its clean_msg): unchanged draws cannot change
  any readout, so k_csid = sum of the counts of stored changed texts whose
  identity differs from the clean identity.  Per-draw bits cannot be rebuilt
  from deduplicated counts, so there is no bits_csid for these models.

Embeddings are computed for unique strings only (deduplicated across all
SNRs of an utterance, batched across utterances), on CUDA if available with
batch size EMB_BATCH (default 256; the GPU is shared with the reruns).  A
per-utterance cache (<out>/_cache/sem_<model>_<ds>.pkl, keyed by a digest of
the record) makes re-running on a growing dump cheap.

Run:  python convert_v2.py [--out rev1] [--models whisper,quartznet,...]
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
import zlib
from collections import Counter
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "Stage-2-verification" / "scenario"))
import run_scenario_datasets as RS                      # noqa: E402
sys.path.append(str(HERE))
import callsign_grammar as CG                           # noqa: E402
from atc_text import norm_text_digits                   # noqa: E402

MODELS = ["quartznet", "deepspeech2", "whisper", "whisper_atc"]
CORPORA = ["atcosim", "atco2"]
SNRS = [60, 50, 45, 40, 35, 30, 25, 20]
FLOORS = (0.90, 0.95)
BETA = 1e-6
TAU = 0.05
EMBEDDER = "all-MiniLM-L6-v2"          # exactly as run_atc_sem.py
EMB_BATCH = int(os.environ.get("EMB_BATCH", 256))
CHUNK_STRINGS = int(os.environ.get("EMB_CHUNK", 20000))
CACHE_VERSION = "convert_v2/1|norm_text_digits|RS.callsign|CG.identity|" \
    + EMBEDDER

ATC_FIELDS = ["model", "dataset", "role", "file", "snr", "N", "beta",
              "k_msg", "k_cs", "clean_msg", "clean_cs", "bits_msg", "bits_cs",
              "k_csid", "bits_csid", "clean_csid", "clean_raw_json"]
SEM_FIELDS = ["model", "dataset", "role", "file", "snr", "N", "beta",
              "clean_msg", "kexact", "ksem_90", "ksem_95", "n_decoded",
              "n_uniq", "min_cos", "mode", "bits_sem90", "bits_sem95"]
CONS_FIELDS = ["model", "dataset", "role", "file", "eff_batch", "no_tf32",
               "beff_n", "beff_distinct_raw", "beff_distinct_norm",
               "clean_b1_norm", "beff_mode_norm", "beff_mode_share",
               "b1_in_beff", "raw_differs", "norm_differs",
               "k_msg_b1", "k_msg_beff", "cert_changes_msg"]

csv.field_size_limit(1 << 24)


@lru_cache(maxsize=None)
def kstar(n=1000, b=BETA, tau=TAU):
    """Largest k with Clopper-Pearson upper bound <= tau (20 at the paper's
    N=1000, beta=1e-6, tau=0.05)."""
    from scipy.stats import beta as _beta
    k = -1
    while k + 1 < n and _beta.ppf(1.0 - b, k + 2, n - k - 1) <= tau:
        k += 1
    return k


def csid_str(ident):
    if ident is None:
        return "<none>"
    return f"{ident[0]}|{' '.join(ident[1])}"


def _parse(line):
    """A complete record (all SNRs, N draws each) or None."""
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return None                       # torn last line (job writing)
    snr = rec.get("snr", {})
    N = rec.get("N")
    if (len(snr) != len(SNRS)
            or any(len(snr.get(str(s), {}).get("idx", ())) != N
                   for s in SNRS)):
        return None
    return rec


def scan_v2(path):
    """Pass 1: file -> line index of its LAST complete record (a resumed job
    may have rewritten a file).  Tolerates a truncated gzip member."""
    last, bad, torn = {}, 0, False
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                rec = _parse(line)
                if rec is None:
                    bad += 1
                    continue
                last[rec["file"]] = i
    except (EOFError, OSError, zlib.error):
        torn = True
    return last, bad, torn


def iter_v2(path, keep):
    """Pass 2: stream the records at the line indices `keep` (appends made
    after pass 1 are ignored; their indices are not in `keep`)."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i in keep:
                    yield json.loads(line)
    except (EOFError, OSError, zlib.error):
        return


def read_v2(path):
    """file -> complete record, all in memory (for small checks)."""
    last, bad, torn = scan_v2(path)
    recs = {r["file"]: r for r in iter_v2(path, set(last.values()))}
    return recs, bad, torn


def split_files():
    out = {}
    for r in csv.DictReader(open(HERE / "split_manifest.csv", newline="")):
        out.setdefault(r["dataset"], set()).add(r["file"])
    return out


def digest(rec):
    h = hashlib.sha1(CACHE_VERSION.encode())
    h.update(json.dumps([rec["N"], rec["clean_b1"], rec["snr"]],
                        sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()


class Embedder:
    def __init__(self):
        import torch
        from sentence_transformers import SentenceTransformer
        self.torch = torch
        self.dev = str(RS.DEV)
        self.em = SentenceTransformer(EMBEDDER, device=self.dev)
        self.em.eval()

    def __call__(self, texts):
        with self.torch.no_grad():
            return self.em.encode(texts, convert_to_tensor=True,
                                  normalize_embeddings=True,
                                  batch_size=EMB_BATCH,
                                  show_progress_bar=False, device=self.dev)


def prepare(rec):
    """CPU part of one utterance: normalized texts, spec bits, texts_sem."""
    N = rec["N"]
    clean = norm_text_digits(rec["clean_b1"])
    cs0, id0 = RS.callsign(clean), CG.identity(clean)
    nrm, flags = {}, {}                  # raw -> norm ; norm -> (cs, csid)

    def flag(t):
        if t not in flags:
            flags[t] = (RS.callsign(t) != cs0, CG.identity(t) != id0)
        return flags[t]

    per_snr, changed = {}, set()
    for snr in SNRS:
        e = rec["snr"][str(snr)]
        norms = []
        for u in e["uniq"]:
            if u not in nrm:
                nrm[u] = norm_text_digits(u)
            norms.append(nrm[u])
        texts = [norms[i] for i in e["idx"]]       # N normalized draws
        cnt = Counter(t for t in texts if t != clean)
        changed.update(cnt)
        bm = "".join("0" if t == clean else "1" for t in texts)
        bc = "".join("1" if (t != clean and flag(t)[0]) else "0"
                     for t in texts)
        bi = "".join("1" if (t != clean and flag(t)[1]) else "0"
                     for t in texts)
        per_snr[snr] = dict(texts=texts, cnt=cnt, bits_msg=bm, bits_cs=bc,
                            bits_csid=bi)
    # batch-consistency diagnostics
    beff = rec.get("clean_beff") or {}
    braw = Counter(beff)
    bnorm = Counter()
    for u, c in beff.items():
        bnorm[norm_text_digits(u)] += c
    b1raw = rec["clean_b1"]
    # mode with ties resolved in favour of the batch-1 decode (so "differs"
    # means the batch-1 decode is strictly NOT a modal batch decode)
    mode_raw = max(braw, key=lambda u: (braw[u], u == b1raw, u)) \
        if braw else b1raw
    mode_norm = max(bnorm, key=lambda u: (bnorm[u], u == clean, u)) \
        if bnorm else clean
    ks = kstar(N)
    k_b1, k_bf = [], []
    for snr in SNRS:
        texts = per_snr[snr]["texts"]
        k_b1.append(sum(1 for t in texts if t != clean))
        k_bf.append(sum(1 for t in texts if t != mode_norm))
    cons = dict(
        model=rec["model"], dataset=rec["dataset"], role=rec["role"],
        file=rec["file"], eff_batch=rec.get("eff_batch"),
        no_tf32=int(bool(rec.get("no_tf32"))),
        beff_n=sum(braw.values()), beff_distinct_raw=len(braw),
        beff_distinct_norm=len(bnorm), clean_b1_norm=clean,
        beff_mode_norm=mode_norm,
        beff_mode_share=round(bnorm[mode_norm] / max(sum(bnorm.values()), 1),
                              4) if bnorm else "",
        b1_in_beff=int(clean in bnorm), raw_differs=int(b1raw != mode_raw),
        norm_differs=int(clean != mode_norm),
        k_msg_b1="/".join(map(str, k_b1)), k_msg_beff="/".join(map(str, k_bf)),
        cert_changes_msg=sum(1 for a, b in zip(k_b1, k_bf)
                             if (a <= ks) != (b <= ks)))
    return dict(rec=rec, clean=clean, cs0=cs0, id0=id0, per_snr=per_snr,
                changed=changed, cons=cons, N=N)


def sem_rows(p, cos):
    """results_sem rows for one prepared utterance given cos[text]."""
    out = {}
    for snr in SNRS:
        d = p["per_snr"][snr]
        b90, b95, mn = [], [], 1.0
        for t in d["texts"]:
            if t == p["clean"]:
                b90.append("0"); b95.append("0")
                continue
            c = cos[t]
            mn = min(mn, c)
            b90.append("1" if c < FLOORS[0] else "0")
            b95.append("1" if c < FLOORS[1] else "0")
        b90, b95 = "".join(b90), "".join(b95)
        out[snr] = dict(ksem_90=b90.count("1"), ksem_95=b95.count("1"),
                        n_uniq=len(d["cnt"]), min_cos=round(mn, 4),
                        bits_sem90=b90, bits_sem95=b95)
    return out


def embed_pending(embed, pending):
    """pending: list of prepared utterances -> {file: sem result}."""
    import torch
    strings, pos = [], {}
    for p in pending:
        for t in [p["clean"]] + sorted(p["changed"]):
            if t not in pos:
                pos[t] = len(strings); strings.append(t)
    E = embed(strings) if strings else None
    res = {}
    for p in pending:
        ch = sorted(p["changed"])
        cos = {}
        if ch:
            ix = torch.tensor([pos[t] for t in ch], device=E.device)
            sims = (E.index_select(0, ix) @ E[pos[p["clean"]]]).tolist()
            cos = dict(zip(ch, sims))
        res[p["rec"]["file"]] = sem_rows(p, cos)
    return res


def convert_one(model, ds, src, out, embed, expected):
    """Stream one v2 dump into the three campaign-schema files.  Records are
    processed in order and written in chunks (bounded memory); embeddings are
    batched across the pending utterances of a chunk."""
    t0 = time.time()
    last, bad, torn = scan_v2(src)
    keep = set(last.values())
    cache_p = out / "_cache" / f"sem_{model}_{ds}.pkl"
    cache = {}
    if cache_p.exists():
        try:
            cache = pickle.loads(cache_p.read_bytes())
        except Exception:                                # noqa: BLE001
            cache = {}
    new_cache, cons, files, no_tf32 = {}, [], [], set()
    n_hit = 0
    pa = out / f"results_atc_{model}_{ds}.csv"
    ps = out / f"results_sem_{model}_{ds}.csv"
    pt = out / f"texts_sem_{model}_{ds}.jsonl.gz"
    ta, ts, tt = (q.with_name(q.name + ".tmp") for q in (pa, ps, pt))
    with open(ta, "w", newline="", encoding="utf-8") as fa, \
            open(ts, "w", newline="", encoding="utf-8") as fs, \
            gzip.open(tt, "wt", encoding="utf-8") as ft:
        wa = csv.DictWriter(fa, fieldnames=ATC_FIELDS)
        ws = csv.DictWriter(fs, fieldnames=SEM_FIELDS)
        wa.writeheader(); ws.writeheader()
        queue, pending, n_pend = [], [], 0      # queue keeps record order

        def flush():
            if pending:
                res = embed_pending(embed(), pending)
                for q in pending:
                    q["sem"] = res[q["rec"]["file"]]
            for p in queue:
                rec, fn = p["rec"], p["rec"]["file"]
                new_cache[fn] = (p["digest"], p["sem"])
                for snr in SNRS:
                    d, s = p["per_snr"][snr], p["sem"][snr]
                    base = dict(model=model, dataset=ds, role=rec["role"],
                                file=fn, snr=snr, N=p["N"], beta=BETA)
                    wa.writerow(dict(
                        base, k_msg=d["bits_msg"].count("1"),
                        k_cs=d["bits_cs"].count("1"), clean_msg=p["clean"],
                        clean_cs=p["cs0"], bits_msg=d["bits_msg"],
                        bits_cs=d["bits_cs"],
                        k_csid=d["bits_csid"].count("1"),
                        bits_csid=d["bits_csid"],
                        clean_csid=csid_str(p["id0"]),
                        clean_raw_json=json.dumps(rec["clean_b1"])))
                    ws.writerow(dict(
                        base, clean_msg=p["clean"],
                        kexact=d["bits_msg"].count("1"),
                        ksem_90=s["ksem_90"], ksem_95=s["ksem_95"],
                        n_decoded=p["N"], n_uniq=s["n_uniq"],
                        min_cos=s["min_cos"], mode="full",
                        bits_sem90=s["bits_sem90"],
                        bits_sem95=s["bits_sem95"]))
                    ft.write(json.dumps(
                        {"file": fn, "snr": snr,
                         "uniq": dict(sorted(d["cnt"].items()))},
                        ensure_ascii=False) + "\n")
                cons.append(p["cons"])
            queue.clear(); pending.clear()

        for rec in iter_v2(src, keep):
            p = prepare(rec)
            p["digest"] = digest(rec)
            files.append(rec["file"])
            no_tf32.add(bool(rec.get("no_tf32")))
            hit = cache.get(rec["file"])
            if hit and hit[0] == p["digest"]:
                p["sem"] = hit[1]; n_hit += 1
            else:
                pending.append(p); n_pend += len(p["changed"]) + 1
            queue.append(p)
            if n_pend >= CHUNK_STRINGS or len(queue) >= 64:
                flush(); n_pend = 0
        flush()
    for t, q in ((ta, pa), (ts, ps), (tt, pt)):
        os.replace(t, q)
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    tmpc = cache_p.with_name(cache_p.name + ".tmp")
    tmpc.write_bytes(pickle.dumps(new_cache))
    os.replace(tmpc, cache_p)
    exp = expected.get(ds, set())
    info = dict(source=src.name, source_bytes=src.stat().st_size,
                source_mtime=time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(src.stat().st_mtime)),
                records=len(files), skipped_lines=bad,
                torn_gzip_tail=torn, expected=len(exp),
                missing=len(exp - set(files)),
                complete=bool(exp) and exp <= set(files),
                no_tf32=sorted(no_tf32),
                cache_hits=n_hit, seconds=round(time.time() - t0, 1))
    print(f"  {model}/{ds}: {len(files)} utterances "
          f"({info['missing']} of {len(exp)} still missing; {bad} bad lines; "
          f"{n_hit} cached) in {info['seconds']}s", flush=True)
    return cons, info


def csid_from_sem(model, ds, out, override=None, tag=""):
    """k_csid for a model copied from the campaign, from its SEMANTIC pass
    (texts_sem + its clean_msg; an unchanged draw cannot change any readout).
    `override`: {(file, snr): rev results_atc row} of patched utterances,
    whose same-pass v2 k_csid replaces the semantic-pass count."""
    override = override or {}
    clean = {}
    for r in csv.DictReader(open(HERE / f"results_sem_{model}_{ds}.csv",
                                 newline="")):
        clean[(r["file"], int(r["snr"]))] = (r["clean_msg"], r["role"],
                                             int(r["N"]))
    ids = {}

    def ident(t):
        if t not in ids:
            ids[t] = CG.identity(t)
        return ids[t]

    rows = []
    with gzip.open(HERE / f"texts_sem_{model}_{ds}.jsonl.gz", "rt",
                   encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            fn, snr = rec["file"], int(rec["snr"])
            c, role, N = clean[(fn, snr)]
            o = override.get((fn, snr))
            if o is not None:
                rows.append(dict(model=model, dataset=ds, role=o["role"],
                                 file=fn, snr=snr, N=int(o["N"]),
                                 clean_msg=o["clean_msg"],
                                 clean_csid=o["clean_csid"],
                                 k_csid=int(o["k_csid"]),
                                 source=f"v2{tag}"))
                continue
            i0 = ident(c)
            k = sum(n for t, n in rec["uniq"].items() if ident(t) != i0)
            rows.append(dict(model=model, dataset=ds, role=role, file=fn,
                             snr=snr, N=N, clean_msg=c, clean_csid=csid_str(i0),
                             k_csid=k, source="semantic_pass"))
    p = out / f"csid_{model}_{ds}.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"  {model}/{ds}: copied campaign files; csid from semantic pass "
          f"({len(rows)} rows, {len(override)} from v2{tag})", flush=True)
    return len(rows)


def apply_patch(model, ds, out, pdir):
    """Replace, inside the copied campaign files in `out`, every utterance
    present in the v2-converted files in `pdir` (a targeted rerun such as
    _numerals): those utterances become same-pass, digit-preserving rows.
    -> ({(file, snr): new results_atc row}, sorted patched files)."""
    new_atc = {(r["file"], r["snr"]): r for r in csv.DictReader(
        open(pdir / f"results_atc_{model}_{ds}.csv", newline="",
             encoding="utf-8"))}
    new_sem = {(r["file"], r["snr"]): r for r in csv.DictReader(
        open(pdir / f"results_sem_{model}_{ds}.csv", newline="",
             encoding="utf-8"))}
    for name, new, extra in ((f"results_atc_{model}_{ds}.csv", new_atc,
                              ATC_FIELDS),
                             (f"results_sem_{model}_{ds}.csv", new_sem,
                              SEM_FIELDS)):
        with open(out / name, newline="") as fh:
            rd = csv.DictReader(fh)
            fields = list(rd.fieldnames) + [c for c in extra
                                            if c not in rd.fieldnames]
            rows = [new.get((r["file"], r["snr"]), r) for r in rd]
        tmp = out / (name + ".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, restval="",
                               extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        os.replace(tmp, out / name)
    new_txt = {}
    with gzip.open(pdir / f"texts_sem_{model}_{ds}.jsonl.gz", "rt",
                   encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            new_txt[(rec["file"], int(rec["snr"]))] = line
    name = f"texts_sem_{model}_{ds}.jsonl.gz"
    tmp = out / (name + ".tmp")
    with gzip.open(out / name, "rt", encoding="utf-8") as fi, \
            gzip.open(tmp, "wt", encoding="utf-8") as fo:
        for line in fi:
            rec = json.loads(line)
            fo.write(new_txt.get((rec["file"], int(rec["snr"])), line))
    os.replace(tmp, out / name)
    return ({(f, int(s)): r for (f, s), r in new_atc.items()},
            sorted({f for f, _ in new_atc}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "rev1"))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--original", default="whisper_atc",
                    help="comma list of models whose CAMPAIGN files are "
                         "copied (csid from the semantic pass) instead of "
                         "converting a v2 dump; add deepspeech2 if its "
                         "TF32-off rerun is dropped")
    ap.add_argument("--patch-tag", default="",
                    help="for --original models: replace the utterances of "
                         "results_v2_<m>_<ds><TAG>.jsonl.gz (a targeted v2 "
                         "rerun, e.g. _numerals) by same-pass "
                         "digit-preserving rows (default: off)")
    a = ap.parse_args()
    originals = {m for m in a.original.split(",") if m}
    out = Path(a.out).resolve()
    if out == HERE:
        sys.exit("--out must not be Stage-5-atc itself")
    out.mkdir(parents=True, exist_ok=True)
    expected = split_files()
    _emb = {}

    def embed():                         # lazy: no GPU unless needed
        if "e" not in _emb:
            _emb["e"] = Embedder()
        return _emb["e"]

    cons, manifest = [], dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),
                              embedder=EMBEDDER, emb_batch=EMB_BATCH,
                              originals=sorted(originals),
                              patch_tag=a.patch_tag, cells={})
    for model in a.models.split(","):
        for ds in CORPORA:
            src = HERE / f"results_v2_{model}_{ds}.jsonl.gz"
            key = f"{model}_{ds}"
            if model in originals:
                for name in (f"results_atc_{model}_{ds}.csv",
                             f"results_sem_{model}_{ds}.csv",
                             f"texts_sem_{model}_{ds}.jsonl.gz"):
                    shutil.copy2(HERE / name, out / name)
                info = dict(kind="original_copied",
                            csid_source="semantic_pass")
                override = None
                # comma list of tags (e.g. _numerals,_tf32fix): the first one
                # with a targeted dump for this model/dataset is applied
                ptag = next((t for t in a.patch_tag.split(",") if t and
                             (HERE / f"results_v2_{model}_{ds}{t}.jsonl.gz").exists()), "")
                psrc = HERE / f"results_v2_{model}_{ds}{ptag}.jsonl.gz"
                if ptag:
                    pdir = out / "_patch" / f"{model}_{ds}{ptag}"
                    pdir.mkdir(parents=True, exist_ok=True)
                    c, _pinfo = convert_one(model, ds, psrc, pdir, embed, {})
                    cons += c
                    override, pfiles = apply_patch(model, ds, out, pdir)
                    info.update(patch_source=psrc.name, patched_files=pfiles)
                    print(f"  {model}/{ds}: patched {len(pfiles)} utterances "
                          f"from {psrc.name}: {pfiles}", flush=True)
                info["csid_rows"] = csid_from_sem(model, ds, out, override,
                                                  ptag)
                if src.exists():
                    info["ignored_v2_dump"] = src.name
                manifest["cells"][key] = info
            elif src.exists():
                c, info = convert_one(model, ds, src, out, embed, expected)
                cons += c
                info["kind"] = "v2_rerun"
                manifest["cells"][key] = info
            else:
                print(f"  {model}/{ds}: no v2 dump yet -- skipped", flush=True)
                manifest["cells"][key] = dict(kind="missing")
    if cons:
        with open(out / "clean_batch_consistency.csv", "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=CONS_FIELDS)
            w.writeheader(); w.writerows(cons)
        for key in sorted({(c["model"], c["dataset"]) for c in cons}):
            cc = [c for c in cons if (c["model"], c["dataset"]) == key]
            print(f"  batch consistency {key[0]}/{key[1]}: "
                  f"{sum(c['norm_differs'] for c in cc)}/{len(cc)} utterances "
                  f"with batch-1 != batch-mode clean (normalized), "
                  f"{sum(c['cert_changes_msg'] > 0 for c in cc)} with a "
                  f"certificate change", flush=True)
    (out / "convert_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
