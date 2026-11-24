"""Multi-embedder audit of the cosine semantic oracle (TMLR revision).

The semantic spec and its call-sign audit (analyze_atc_semaudit.py) use one
sentence embedder, all-MiniLM-L6-v2.  This script recomputes the SAME audit
cosines -- every stored changed transcript against its clean transcript,
count-weighted by draws -- with three embedders:

    minilm  all-MiniLM-L6-v2                       (the paper's oracle)
    mpnet   sentence-transformers/all-mpnet-base-v2
    bge     BAAI/bge-base-en-v1.5                  (no query instruction:
                                                    symmetric similarity)

all with normalize_embeddings=True.  Cosine scales differ across embedders
(a floor of 0.95 is not the same strictness for each), so two views are
reported per embedder:

  (i)  fixed floors 0.90 / 0.95: share of ALL changed draws forgiven
       (cos >= floor), share of call-sign flips forgiven (cs_norm predicate
       of analyze_atc_semaudit.py) and share of SIGNATURE flips forgiven
       (ordered canonical digit+NATO signature of the call-sign span changes,
       scan_semaudit_witnesses.dn_sig);
  (ii) matched forgiveness: the floor f_e at which embedder e forgives the
       same share of ALL changed draws as MiniLM does at 0.95 (a draw-weighted
       quantile of e's cosines; ties make the achieved share >= the target,
       reported as matched_forgive), and the shares of call-sign flips and
       of signature flips that pass f_e.  A lower flip pass-rate at matched
       forgiveness means the embedder separates call-sign errors from benign
       edits better at equal leniency.

Pooled over every cell present in ATC_DATA_DIR (texts_sem_*.jsonl.gz +
results_sem_*.csv) and per (model, corpus).  Cosines are cached per
(embedder, cell) in <ATC_DATA_DIR>/_cache/ keyed by a digest of the strings.

Outputs (ATC_DATA_DIR): embedder_audit.json, embedder_audit_cells.csv;
macros are emitted by emit_semaudit_macros.py (\\SaEmb*).

Env: EMB_AUDIT_FILE_FRAC (default 1.0) audits a deterministic crc32 subset
of utterances if the full run is too slow; EMB_BATCH_SMALL / EMB_BATCH_BASE
(default 256 / 128) are the encode batch sizes.

Run:  ATC_DATA_DIR=rev1 ATC_REV=1 python audit_embedders.py
"""
import csv
import gzip
import hashlib
import json
import os
import time
import zlib

import numpy as np
import torch

from analyze_atc_semaudit import (MODELS, CORPORA, RS, DEV, cs_norm,
                                  clean_map)
from atc_paths import DATA, REV
from scan_semaudit_witnesses import dn_sig

EMBEDDERS = [
    ("minilm", "all-MiniLM-L6-v2", int(os.environ.get("EMB_BATCH_SMALL", 256))),
    ("mpnet", "sentence-transformers/all-mpnet-base-v2",
     int(os.environ.get("EMB_BATCH_BASE", 128))),
    ("bge", "BAAI/bge-base-en-v1.5", int(os.environ.get("EMB_BATCH_BASE", 128))),
]
REF_EMB, REF_FLOOR = "minilm", 0.95
FIXED = ((0.90, "90"), (0.95, "95"))
CHUNK = 20000
FILE_FRAC = float(os.environ.get("EMB_AUDIT_FILE_FRAC", 1.0))


def collect(model, ds):
    """All stored changed transcripts of one cell as flat arrays."""
    cm = clean_map(model, ds)
    groups, gi = [], {}               # (file, clean) -> [clean, texts...]
    w, csf, sgf, sidx = [], [], [], []
    with gzip.open(DATA / f"texts_sem_{model}_{ds}.jsonl.gz", "rt",
                   encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            fn, snr = rec["file"], int(rec["snr"])
            if FILE_FRAC < 1 and \
                    zlib.crc32(fn.encode()) % 10000 >= FILE_FRAC * 10000:
                continue
            clean, _N = cm[(fn, snr)]
            key = (fn, clean)
            if key not in gi:
                gi[key] = len(groups)
                groups.append(dict(clean=clean, texts={}))
            g = groups[gi[key]]
            cs0 = RS.callsign(clean)
            cs0n, s0 = cs_norm(cs0), dn_sig(cs0)
            for t, c in rec["uniq"].items():
                cst = RS.callsign(t)
                f1 = cs_norm(cst) != cs0n
                if t not in g["texts"]:
                    g["texts"][t] = len(g["texts"])
                w.append(c)
                csf.append(f1)
                sgf.append(f1 and dn_sig(cst) != s0)
                sidx.append((gi[key], g["texts"][t]))
    # flat string list: per group [clean, t_0, t_1, ...]
    strings, off = [], []
    for g in groups:
        off.append(len(strings))
        strings.append(g["clean"])
        strings.extend(g["texts"])          # insertion order == index order
    flat = np.array([off[a] + 1 + b for a, b in sidx], dtype=np.int64)
    return dict(w=np.array(w, dtype=np.int64), csf=np.array(csf, dtype=bool),
                sgf=np.array(sgf, dtype=bool), flat=flat, strings=strings,
                off=np.array(off + [len(strings)], dtype=np.int64))


def cosines(em, bs, cell):
    """cos(string, its group's clean) for every string of the cell."""
    strings, off = cell["strings"], cell["off"]
    out = np.ones(len(strings), dtype=np.float32)
    g0 = 0
    ng = len(off) - 1
    while g0 < ng:
        g1 = g0
        while g1 < ng and off[g1 + 1] - off[g0] <= CHUNK:
            g1 += 1
        g1 = max(g1, g0 + 1)
        a, b = int(off[g0]), int(off[g1])
        with torch.no_grad():
            E = em.encode(strings[a:b], convert_to_tensor=True,
                          normalize_embeddings=True, batch_size=bs,
                          show_progress_bar=False, device=str(DEV))
            # index of each string's clean row, relative to the chunk
            gid = np.repeat(np.arange(g0, g1), np.diff(off[g0:g1 + 1]))
            cix = torch.tensor(off[gid] - a, device=E.device)
            sims = (E * E.index_select(0, cix)).sum(-1)
        out[a:b] = sims.float().cpu().numpy()
        g0 = g1
    return out


def matched_floor(cos, w, target):
    """Largest floor f with draw-weighted share(cos >= f) >= target."""
    W = w.sum()
    if not W:
        return None, None
    order = np.argsort(-cos, kind="stable")
    cw = np.cumsum(w[order])
    k = int(np.searchsorted(cw, target * W, side="left"))
    k = min(k, len(order) - 1)
    f = float(cos[order[k]])
    return f, float(w[cos >= f].sum() / W)


def shares(cos, w, csf, sgf, floor):
    p = cos >= floor
    W, Wc, Ws = w.sum(), w[csf].sum(), w[sgf].sum()
    return (float(w[p].sum() / W) if W else None,
            float(w[p & csf].sum() / Wc) if Wc else None,
            float(w[p & sgf].sum() / Ws) if Ws else None)


def r6(x):
    return None if x is None else round(x, 6)


def main():
    if not REV:
        raise SystemExit("audit_embedders.py is a revision-mode script: "
                         "set ATC_DATA_DIR=<dir> ATC_REV=1")
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    cells = {}
    for m in MODELS:
        for ds in CORPORA:
            if (DATA / f"texts_sem_{m}_{ds}.jsonl.gz").exists():
                cells[(m, ds)] = collect(m, ds)
                c = cells[(m, ds)]
                print(f"  {m}/{ds}: {len(c['w']):,} (snr,text) entries, "
                      f"{len(c['strings']):,} strings, {c['w'].sum():,} "
                      f"changed draws", flush=True)
    cache = DATA / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    cos = {}                                   # (emb, m, ds) -> entry cosines
    for e, hf, bs in EMBEDDERS:
        em = None
        for (m, ds), c in cells.items():
            h = hashlib.sha1(f"{hf}|{m}|{ds}".encode())
            for s in c["strings"]:
                h.update(s.encode("utf-8")); h.update(b"\0")
            dg = h.hexdigest()
            cp = cache / f"audit_{e}_{m}_{ds}.npz"
            per_string = None
            if cp.exists():
                z = np.load(cp)
                if str(z["digest"]) == dg:
                    per_string = z["cos"]
            if per_string is None:
                if em is None:
                    em = SentenceTransformer(hf, device=str(DEV))
                    em.eval()
                t1 = time.time()
                per_string = cosines(em, bs, c)
                np.savez(cp, cos=per_string, digest=np.array(dg))
                print(f"    {e} {m}/{ds}: {len(c['strings']):,} strings "
                      f"embedded in {time.time() - t1:.0f}s", flush=True)
            cos[(e, m, ds)] = per_string[c["flat"]]
        del em
        if DEV.type == "cuda":
            torch.cuda.empty_cache()

    def pool(key):
        return np.concatenate([c[key] for c in cells.values()])
    W, CSF, SGF = pool("w"), pool("csf"), pool("sgf")
    PC = {e: np.concatenate([cos[(e, m, ds)] for (m, ds) in cells])
          for e, _, _ in EMBEDDERS}
    target = shares(PC[REF_EMB], W, CSF, SGF, REF_FLOOR)[0]
    res = dict(embedders={e: hf for e, hf, _ in EMBEDDERS},
               reference=dict(embedder=REF_EMB, floor=REF_FLOOR,
                              forgive_share=r6(target)),
               scope=dict(cells=[f"{m}_{ds}" for m, ds in cells],
                          file_frac=FILE_FRAC, draws_changed=int(W.sum()),
                          cs_flips=int(W[CSF].sum()),
                          sig_flips=int(W[SGF].sum()),
                          entries=int(len(W))),
               pooled={}, per_cell={})

    def block(cv, w, csf, sgf, tgt, mf_fixed=None):
        d = {}
        for f, tag in FIXED:
            fg, cp_, sp = shares(cv, w, csf, sgf, f)
            d[f"forgive{tag}"], d[f"cs_pass{tag}"], d[f"sig_pass{tag}"] = \
                r6(fg), r6(cp_), r6(sp)
        mf, ach = matched_floor(cv, w, tgt) if tgt is not None else (None,
                                                                    None)
        d["matched_floor"], d["matched_forgive"] = r6(mf), r6(ach)
        if mf is not None:
            _, cp_, sp = shares(cv, w, csf, sgf, mf)
            d["cs_pass_matched"], d["sig_pass_matched"] = r6(cp_), r6(sp)
        if mf_fixed is not None:          # pass rates at the POOLED floor
            fg, cp_, sp = shares(cv, w, csf, sgf, mf_fixed)
            d["forgive_at_pooled_floor"] = r6(fg)
            d["cs_pass_at_pooled_floor"] = r6(cp_)
            d["sig_pass_at_pooled_floor"] = r6(sp)
        return d

    for e, _, _ in EMBEDDERS:
        res["pooled"][e] = block(PC[e], W, CSF, SGF, target)
    crow = []
    sa = DATA / "numbers_semaudit.json"
    sa = json.loads(sa.read_text()) if sa.exists() else {}
    for (m, ds), c in cells.items():
        w, csf, sgf = c["w"], c["csf"], c["sgf"]
        tgt = shares(cos[(REF_EMB, m, ds)], w, csf, sgf, REF_FLOOR)[0]
        pc = dict(draws_changed=int(w.sum()), cs_flips=int(w[csf].sum()),
                  sig_flips=int(w[sgf].sum()), cell_target=r6(tgt),
                  semaudit_condpass95=sa.get(f"condpass95_{m}_{ds}"))
        for e, _, _ in EMBEDDERS:
            pc[e] = block(cos[(e, m, ds)], w, csf, sgf, tgt,
                          mf_fixed=res["pooled"][e]["matched_floor"])
            crow.append(dict(model=m, dataset=ds, embedder=e,
                             draws_changed=pc["draws_changed"],
                             cs_flips=pc["cs_flips"],
                             sig_flips=pc["sig_flips"], **pc[e]))
        res["per_cell"][f"{m}_{ds}"] = pc
    (DATA / "embedder_audit.json").write_text(json.dumps(res, indent=1))
    with open(DATA / "embedder_audit_cells.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(crow[0]))
        wr.writeheader(); wr.writerows(crow)
    print(f"  reference: MiniLM forgives {100*target:.2f}% of changed draws "
          f"at 0.95", flush=True)
    for e, _, _ in EMBEDDERS:
        d = res["pooled"][e]
        print(f"  {e:6s} cs_pass95={d['cs_pass95']} sig_pass95="
              f"{d['sig_pass95']} matched_floor={d['matched_floor']} "
              f"cs_pass@matched={d.get('cs_pass_matched')} "
              f"sig_pass@matched={d.get('sig_pass_matched')}", flush=True)
    print(f"wrote embedder_audit.json, embedder_audit_cells.csv "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
