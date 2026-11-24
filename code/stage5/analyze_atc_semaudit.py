"""Probabilistic audit of the cosine semantic oracle, ATC-native.

The semantic spec certifies V_sem = P[cos(perturbed, clean) < floor].  The
SAFETY question is about meaning.  For general meaning changes a human audit
is unavoidable (see the manuscript's assumption discussion), but for the ATC
safety core -- the call-sign -- the false-pass mass is EXACTLY measurable from
data this campaign already stored:

    eta_cs(x, floor) = P[ call-sign changed  AND  cos >= floor ]

i.e. the probability mass of draws whose decoded call-sign differs from the
clean one yet the cosine oracle declares the meaning preserved.  Everything
needed is in texts_sem_<model>_<ds>.jsonl.gz (deduped perturbed transcripts
with draw counts, per (file, snr), from the SAME decode pass as ksem/kexact)
plus the stored clean transcript; the call-sign extractor is the campaign's
own RS.callsign.  A draw with an unchanged transcript cannot flip the
call-sign, so restricting to stored changed-transcripts is exact.

Outputs:
  semaudit_perinput.csv   per (model,corpus,file,snr): n_draws_changed,
                          k_csflip, k_csflip_pass90, k_csflip_pass95,
                          cp_up on each eta at beta=1e-6
  semaudit_summary.csv    per (model,corpus,snr): totals + quantiles
  semaudit_examples.json  the highest-cosine call-sign-flipping edits
                          (concrete witnesses for the necessity proposition)
  numbers_semaudit.json   headline numbers

ATC_DATA_DIR=<dir> reads/writes everything in <dir> (atc_paths.py); with
ATC_REV=1 the embedding batch is 256 (the GPU is shared with reruns).  The
audit logic itself is identical in both modes.

Run:  python analyze_atc_semaudit.py
"""
import csv
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from scipy.stats import beta as _beta

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "Stage-2-verification" / "scenario"))

import run_scenario_datasets as RS                      # noqa: E402
sys.path.append(str(HERE))
from atc_paths import DATA, REV                         # noqa: E402

DEV = RS.DEV
BETA = 1e-6
FLOORS = (0.90, 0.95)
MODELS = ["quartznet", "deepspeech2", "whisper", "whisper_atc"]
CORPORA = ["atcosim", "atco2"]

csv.field_size_limit(1 << 24)


def cs_norm(cs):
    """Orthographic normalisation: the campaign's cs spec compares raw token
    strings, so re-tokenisation (e.g. "x's" -> "x 's") counts as a flip there.
    For the ORACLE AUDIT that would be a fake witness -- a call-sign counts as
    changed only if it differs after dropping all non-alphanumerics."""
    return re.sub(r"[^a-z0-9]", "", cs.lower())


def cp_upper(k, n, b):
    if n == 0:
        return None
    if k >= n:
        return 1.0
    return float(_beta.ppf(1.0 - b, k + 1, n - k))


def load_embedder():
    from sentence_transformers import SentenceTransformer
    em = SentenceTransformer("all-MiniLM-L6-v2", device=str(DEV))
    em.eval()

    @torch.no_grad()
    def embed(texts):
        return em.encode(texts, convert_to_tensor=True,
                         normalize_embeddings=True,
                         batch_size=256 if REV else 512,
                         show_progress_bar=False, device=str(DEV))
    return embed


def clean_map(model, ds):
    out = {}
    for r in csv.DictReader(open(DATA / f"results_sem_{model}_{ds}.csv",
                                 newline="")):
        out[(r["file"], int(r["snr"]))] = (r["clean_msg"], int(r["N"]))
    return out


def main():
    embed = load_embedder()
    rows, examples = [], []
    for model in MODELS:
        for ds in CORPORA:
            gzp = DATA / f"texts_sem_{model}_{ds}.jsonl.gz"
            if not gzp.exists():
                print(f"  (skip {model}/{ds}: no transcript dump)")
                continue
            cm = clean_map(model, ds)
            n_rows = 0
            with gzip.open(gzp, "rt", encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line)
                    fn, snr = rec["file"], int(rec["snr"])
                    uniq = rec["uniq"]
                    clean, N = cm[(fn, snr)]
                    cs0 = RS.callsign(clean)
                    cs0n = cs_norm(cs0)
                    texts = list(uniq.keys())
                    if texts:
                        E = embed([clean] + texts)
                        sims = (E[1:] @ E[0]).tolist()
                    else:
                        sims = []
                    n_changed = sum(uniq.values())
                    k_raw = k_flip = k_p90 = k_p95 = 0
                    for t, s in zip(texts, sims):
                        cst = RS.callsign(t)
                        c = uniq[t]
                        if cst != cs0:
                            k_raw += c          # spec-consistent raw count
                        if cs_norm(cst) == cs0n:
                            continue            # orthography-only: no witness
                        k_flip += c
                        if s >= FLOORS[0]:
                            k_p90 += c
                        if s >= FLOORS[1]:
                            k_p95 += c
                            examples.append(dict(
                                model=model, dataset=ds, file=fn, snr=snr,
                                cos=round(s, 4), count=c,
                                clean_cs=cs0, flip_cs=cst,
                                clean=clean[:90], perturbed=t[:90]))
                    rows.append(dict(
                        model=model, dataset=ds, file=fn, snr=snr, N=N,
                        n_changed=n_changed, k_csflip_raw=k_raw,
                        k_csflip=k_flip,
                        k_csflip_pass90=k_p90, k_csflip_pass95=k_p95,
                        eta90_cp_up=round(cp_upper(k_p90, N, BETA), 6),
                        eta95_cp_up=round(cp_upper(k_p95, N, BETA), 6)))
                    n_rows += 1
            print(f"  {model}/{ds}: {n_rows} rows audited", flush=True)

    with open(DATA / "semaudit_perinput.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # summaries
    cells = defaultdict(list)
    for r in rows:
        cells[(r["model"], r["dataset"], r["snr"])].append(r)
    srows, nums = [], {}
    for (m, ds, snr), rs_ in sorted(cells.items()):
        tot = sum(r["N"] for r in rs_)
        f90 = sum(r["k_csflip_pass90"] for r in rs_)
        f95 = sum(r["k_csflip_pass95"] for r in rs_)
        fl = sum(r["k_csflip"] for r in rs_)
        raw = sum(r["k_csflip_raw"] for r in rs_)
        srows.append(dict(
            model=m, dataset=ds, snr=snr, M=len(rs_), draws=tot,
            k_csflip_raw=raw, k_csflip=fl, k_pass90=f90, k_pass95=f95,
            rate_csflip=round(fl / tot, 6),
            rate_pass90=round(f90 / tot, 6),
            rate_pass95=round(f95 / tot, 6),
            cond_pass90=round(f90 / fl, 4) if fl else None,
            cond_pass95=round(f95 / fl, 4) if fl else None))
    with open(DATA / "semaudit_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(srows[0].keys()))
        w.writeheader(); w.writerows(srows)

    # headline: campaign-wide conditional false-pass rates
    for m in MODELS:
        for ds in CORPORA:
            if REV and not any(r["model"] == m and r["dataset"] == ds
                               for r in rows):
                continue                # rerun not converted yet: no keys
            fl = sum(r["k_csflip"] for r in rows
                     if r["model"] == m and r["dataset"] == ds)
            f90 = sum(r["k_csflip_pass90"] for r in rows
                      if r["model"] == m and r["dataset"] == ds)
            f95 = sum(r["k_csflip_pass95"] for r in rows
                      if r["model"] == m and r["dataset"] == ds)
            nums[f"csflips_{m}_{ds}"] = fl
            nums[f"csflips_raw_{m}_{ds}"] = sum(
                r["k_csflip_raw"] for r in rows
                if r["model"] == m and r["dataset"] == ds)
            nums[f"condpass90_{m}_{ds}"] = round(f90 / fl, 4) if fl else None
            nums[f"condpass95_{m}_{ds}"] = round(f95 / fl, 4) if fl else None
    nums["examples_pass95"] = len(examples)
    examples.sort(key=lambda e: -e["cos"])
    seen_ex, uniq_ex = set(), []
    for e in examples:                       # dedupe repeated (clean, pert)
        key = (e["model"], e["dataset"], e["clean"], e["perturbed"])
        if key not in seen_ex:
            seen_ex.add(key)
            uniq_ex.append(e)
    (DATA / "semaudit_examples.json").write_text(
        json.dumps(uniq_ex[:200], indent=1))
    (DATA / "numbers_semaudit.json").write_text(json.dumps(nums, indent=2))
    print(f"wrote semaudit_perinput.csv ({len(rows)} rows), "
          f"semaudit_summary.csv ({len(srows)}), "
          f"{len(examples)} floor-0.95-passing call-sign flips collected")


if __name__ == "__main__":
    main()
