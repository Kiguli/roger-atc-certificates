"""Mine SAFETY-GRADE witnesses for the cosine-oracle audit.

analyze_atc_semaudit.py measures eta_cs for call-sign flips at the
orthographically-normalised string level; sorting those by cosine surfaces
near-identical strings (articles, loop truncations) first.  The manuscript
needs the sharper class: draws whose call-sign DIGITS differ -- the decoded
message addresses a different aircraft -- yet the cosine oracle still calls
the meaning preserved.  This scan:

  1. string-filters every stored unique transcript for digit-signature or
     NATO-letter-signature changes in the extracted call-sign (CPU, cheap);
  2. embeds only those candidates (GPU, small fraction of the full audit);
  3. reports per (model, corpus): k_digitflip, k_digitflip_pass90/95 draw
     counts, and the top witnesses by cosine within the digit-flip class.

Flip-predicate hierarchy (paper section): raw-string >= normalised-string
>= digit/NATO-level; eta is exactly measurable for each.

Revision mode (ATC_DATA_DIR=<dir> ATC_REV=1, atc_paths.py).  The campaign
predicate above counts a flip only when the DIGIT signature changes (dflip);
NATO-only changes were candidates but never counted, raw spellings were
compared (alpha != alfa, niner != nine) and digit/letter interleaving was
ignored.  Revision mode additionally counts a flip when the ORDERED digit+NATO
signature of the extracted call-sign span changes, canonicalized with
callsign_grammar.canon_tokens/signature (alpha->alfa, juliet->juliett,
niner->nine, "x ray"->xray, double/triple expanded):
  sigflips_/sigpass90_/sigpass95_/sigcond95_<m>_<ds>   new predicate
  digflips_/digpass90_/digpass95_/digcond95_<m>_<ds>   campaign predicate
(both kept so they can be compared); the released witnesses are the
signature-flip class.  Default mode is unchanged.

Outputs: witnesses_semaudit.json, numbers_witness.json  (in ATC_DATA_DIR)
Run:  python scan_semaudit_witnesses.py
"""
import gzip
import json
from pathlib import Path

from analyze_atc_semaudit import (HERE, FLOORS, MODELS, CORPORA, RS,
                                  cs_norm, clean_map, load_embedder)
from atc_paths import DATA, REV

DIGITS = {"zero", "one", "two", "three", "four", "five", "six", "seven",
          "eight", "nine", "niner"}
NATO = {"alfa", "alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
        "golf", "hotel", "india", "juliett", "juliet", "kilo", "lima",
        "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra",
        "tango", "uniform", "victor", "whiskey", "xray", "yankee", "zulu"}


def digit_sig(cs):
    return tuple(w for w in cs.split() if w in DIGITS)


def nato_sig(cs):
    return tuple(w for w in cs.split() if w in NATO)


def dn_sig(cs):
    """Revision predicate: ordered, canonicalized digit+NATO signature of a
    call-sign span (callsign_grammar.canon_tokens + signature)."""
    import callsign_grammar as CG
    return CG.signature(CG.canon_tokens(cs))


def main():
    embed = load_embedder()
    wit = {}
    nums = {}
    for model in MODELS:
        for ds in CORPORA:
            gzp = DATA / f"texts_sem_{model}_{ds}.jsonl.gz"
            if not gzp.exists():
                continue
            cm = clean_map(model, ds)
            k_flip = k_p90 = k_p95 = 0
            s_p90 = s_p95 = 0
            cand = []                     # rows of (file,snr,clean,t,count,..)
            with gzip.open(gzp, "rt", encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line)
                    fn, snr = rec["file"], int(rec["snr"])
                    clean, _N = cm[(fn, snr)]
                    cs0 = RS.callsign(clean)
                    d0, n0 = digit_sig(cs0), nato_sig(cs0)
                    s0 = dn_sig(cs0) if REV else None
                    for t, c in rec["uniq"].items():
                        cst = RS.callsign(t)
                        if cs_norm(cst) == cs_norm(cs0):
                            continue
                        dflip = digit_sig(cst) != d0
                        nflip = nato_sig(cst) != n0
                        sflip = REV and dn_sig(cst) != s0
                        if dflip or nflip or sflip:
                            cand.append((fn, snr, clean, t, c, cs0, cst,
                                         dflip, nflip, sflip))
            # embed candidates in blocks, pairing each with its clean text
            k_flip = sum(c[4] for c in cand if c[7])
            s_flip = sum(c[4] for c in cand if c[9])
            rows = []
            B = 4096
            for i in range(0, len(cand), B):
                blk = cand[i:i + B]
                texts = []
                for fn, snr, clean, t, c, cs0, cst, dflip, nflip, _s in blk:
                    texts += [clean, t]
                E = embed(texts)
                for j, (fn, snr, clean, t, c, cs0, cst, dflip,
                        nflip, sflip) in enumerate(blk):
                    s = float(E[2 * j] @ E[2 * j + 1])
                    if dflip:
                        if s >= FLOORS[0]:
                            k_p90 += c
                        if s >= FLOORS[1]:
                            k_p95 += c
                    if sflip:
                        if s >= FLOORS[0]:
                            s_p90 += c
                        if s >= FLOORS[1]:
                            s_p95 += c
                    if s >= FLOORS[1]:
                        row = dict(
                            file=fn, snr=snr, cos=round(s, 4), count=c,
                            clean_cs=cs0, flip_cs=cst, digit_flip=dflip,
                            nato_flip=nflip, clean=clean[:100],
                            perturbed=t[:100])
                        if REV:
                            row["sig_flip"] = sflip
                        rows.append(row)
            rows.sort(key=lambda r: -r["cos"])
            key = "sig_flip" if REV else "digit_flip"
            dig = [r for r in rows if r[key]]
            wit[f"{model}_{ds}"] = dig[:60] + \
                [r for r in rows if not r[key]][:20]
            nums[f"digflips_{model}_{ds}"] = k_flip
            nums[f"digpass90_{model}_{ds}"] = k_p90
            nums[f"digpass95_{model}_{ds}"] = k_p95
            nums[f"digcond95_{model}_{ds}"] = (
                round(k_p95 / k_flip, 4) if k_flip else None)
            if REV:
                nums[f"sigflips_{model}_{ds}"] = s_flip
                nums[f"sigpass90_{model}_{ds}"] = s_p90
                nums[f"sigpass95_{model}_{ds}"] = s_p95
                nums[f"sigcond95_{model}_{ds}"] = (
                    round(s_p95 / s_flip, 4) if s_flip else None)
                print(f"  {model}/{ds}: signature flips {s_flip:,} draws, "
                      f"pass95 {s_p95:,} "
                      f"({(100*s_p95/s_flip) if s_flip else 0:.1f}%)",
                      flush=True)
            print(f"  {model}/{ds}: {len(cand)} candidate texts, "
                  f"{k_flip:,} digit-flip draws, "
                  f"pass95 {k_p95:,} "
                  f"({(100*k_p95/k_flip) if k_flip else 0:.1f}%), "
                  f"{len(dig)} floor-passing {key.split('_')[0]} witnesses",
                  flush=True)
    (DATA / "witnesses_semaudit.json").write_text(json.dumps(wit, indent=1))
    (DATA / "numbers_witness.json").write_text(json.dumps(nums, indent=2))
    print("wrote witnesses_semaudit.json, numbers_witness.json")


if __name__ == "__main__":
    main()
