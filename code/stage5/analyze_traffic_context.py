"""Is a call-sign flip a misattribution or a missed call?  (reviewer CGs6)

Each ATCO2 recording ships an .info file listing the call-signs of the
aircraft near the airport at recording time, from surveillance data
("callsigns nearby: DLH3NL : Lufthansa Three November Lima ...").  That is
the traffic context the paper previously lacked: a decoded call-sign that
matches ANOTHER aircraft on that list is a potential misattribution; one that
matches no listed aircraft is a garbled or missed call.

For every stored perturbed transcript of an ATCO2 utterance we compare the
grammar identity readout (callsign_grammar.identity: designator + ordered
digit/phonetic signature) of the clean and perturbed decodes and classify
the draw as

  same         identity unchanged
  lost         clean decode has a call-sign, perturbed decode has none
  other_exact  changed, and equal (designator + signature) to another
               listed aircraft
  other_sig    changed, signature equal to another listed aircraft's
               signature but designator differs/absent (abbreviated forms)
  unlisted     changed, matches no listed aircraft
  gained       clean decode has no call-sign, perturbed one does

Sources: results_v2_<model>_atco2.jsonl.gz when present (revision pass, raw
transcripts, digit-preserving normalization); otherwise the semantic pass's
texts_sem_<model>_atco2.jsonl.gz (changed draws only; unchanged draws cannot
flip).  Output: traffic_context.csv (per model x snr x class) and
traffic_context.json (pooled).

Run:  python analyze_traffic_context.py
"""
import csv
import glob
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "Stage-2-verification" / "scenario"))
sys.path.insert(0, str(HERE))
import run_scenario_datasets as RS                      # noqa: E402
import callsign_grammar as CG                           # noqa: E402
from atc_text import norm_text_digits                   # noqa: E402

MODELS = ["quartznet", "deepspeech2", "whisper", "whisper_atc"]
csv.field_size_limit(1 << 24)


def nearby(rec):
    """recording -> list of (icao, identity) for the listed aircraft."""
    p = CG.ATCO2_DIR / f"{rec}.info"
    out = []
    if not p.exists():
        return out
    on = False
    for line in open(p, encoding="utf-8", errors="replace"):
        if line.lower().startswith("callsigns nearby"):
            on = True
            continue
        if not on:
            continue
        m = re.match(r"^\s*([A-Z0-9]+)\s*:\s*(.+)$", line)
        if not m:
            continue
        spoken = m.group(2)
        # "Csa / Czech One Delta Zulu": one identity per designator variant
        head, _, tail = spoken.partition("/")
        variants = [spoken] if not tail else [
            head + " " + " ".join(tail.split()[1:]), tail]
        for v in variants:
            ident = CG.identity(norm_text_digits(v))
            if ident is not None:
                out.append((m.group(1), ident))
    return out


def consistent(dec, lst):
    """Could decoded identity `dec` designate the listed aircraft `lst`?
    Airline call-signs: same designator and signature.  Registrations
    (no designator) may be abbreviated (ICAO Annex 10 / Doc 4444: first
    character plus at least the last two), so a decoded registration is
    consistent with a listed one when it equals it, is a suffix of it of
    length >= 2, or is its first character followed by such a suffix."""
    dd, ds = dec
    ld, ls = lst
    if dd and ld:
        return dd == ld and ds == ls
    if dd and not ld:
        return False
    if not ds:
        return False
    if ds == ls:
        return True
    if len(ds) >= 2 and len(ds) < len(ls):
        if tuple(ls[-len(ds):]) == ds:
            return True
        if ds[0] == ls[0] and len(ds) >= 3 and                 tuple(ls[-(len(ds) - 1):]) == ds[1:]:
            return True
    return False


def classify(id0, idt, traffic):
    if idt == id0:
        return "same"
    if idt is None:
        return "lost"
    if id0 is None:
        return "gained"
    own = {icao for icao, i in traffic if consistent(id0, i)}
    others = [(icao, i) for icao, i in traffic if icao not in own]
    if any(consistent(idt, i) for _, i in others):
        return "other_listed"
    if idt[1] and any(i[1] == idt[1] for _, i in others):
        return "other_sig"
    return "unlisted"


def iter_draws(model):
    """Yield (file, snr, clean_text, {perturbed_text: count}, normalizer)."""
    v2 = HERE / f"results_v2_{model}_atco2.jsonl.gz"
    if v2.exists():
        with gzip.open(v2, "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                c0 = rec["clean_b1"]
                for snr, d in rec["snr"].items():
                    cnt = Counter(d["idx"])
                    yield (rec["file"], int(snr), c0,
                           {d["uniq"][i]: n for i, n in cnt.items()},
                           norm_text_digits, "v2")
        return
    clean = {}
    for r in csv.DictReader(open(HERE / f"results_sem_{model}_atco2.csv",
                                 newline="")):
        clean[(r["file"], int(r["snr"]))] = r["clean_msg"]
    with gzip.open(HERE / f"texts_sem_{model}_atco2.jsonl.gz", "rt",
                   encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            yield (rec["file"], int(rec["snr"]),
                   clean[(rec["file"], int(rec["snr"]))], rec["uniq"],
                   lambda s: s, "sem")


def main():
    seg = {r["file"]: r["recording"] for r in csv.DictReader(
        open(HERE / "atco2_segment_map.csv", newline="", encoding="utf-8"))}
    traffic_cache = {}
    rows, pooled = [], {}
    for model in MODELS:
        counts = defaultdict(Counter)
        counts_l = defaultdict(Counter)       # utterances whose own call-sign is listed
        heur = Counter()
        src = None
        clean_listed = Counter()
        seen_files = set()
        for fn, snr, c0, uniq, norm, src in iter_draws(model):
            rec = seg.get(fn)
            if rec is None:
                continue
            if rec not in traffic_cache:
                traffic_cache[rec] = nearby(rec)
            tr = traffic_cache[rec]
            n0 = norm(c0)
            id0 = CG.identity(n0)
            listed = id0 is not None and any(consistent(id0, i)
                                             for _, i in tr)
            if fn not in seen_files:
                seen_files.add(fn)
                clean_listed["has_id"] += id0 is not None
                clean_listed["id_listed"] += listed
                clean_listed["files"] += 1
            cs0 = RS.callsign(n0)
            for t, c in uniq.items():
                nt = norm(t)
                if nt == n0:
                    continue
                cls = classify(id0, CG.identity(nt), tr)
                counts[snr][cls] += c
                if listed:                    # traffic context covers x
                    counts_l[snr][cls] += c
                if RS.callsign(nt) != cs0:
                    heur["cs_flip"] += c
                    heur["cs_flip_identity_same"] += cls == "same"
                    heur["cs_flip_identity_same_draws"] += c * (cls == "same")
        tot = Counter()
        for snr, cn in sorted(counts.items()):
            tot.update(cn)
            for k, v in cn.items():
                rows.append(dict(model=model, snr=snr, cls=k, draws=v,
                                 source=src))
        totl = Counter()
        for snr, cn in counts_l.items():
            totl.update(cn)
        changed = sum(v for k, v in tot.items() if k != "same")
        flips = tot["other_listed"] + tot["other_sig"] + tot["unlisted"]
        flipsl = totl["other_listed"] + totl["other_sig"] + totl["unlisted"]
        idchl = sum(v for k, v in totl.items() if k != "same")
        pooled[model] = dict(
            source=src, changed_transcript_draws=sum(tot.values()),
            identity_changed_draws=changed, all=dict(tot),
            frac_flips_other_listed=round(tot["other_listed"] / flips, 4)
            if flips else None,
            listed_subset=dict(totl),
            listed_identity_changed=idchl,
            listed_frac_lost=round(totl["lost"] / idchl, 4) if idchl else None,
            listed_frac_flip=round(flipsl / idchl, 4) if idchl else None,
            listed_frac_flips_other_listed=round(
                totl["other_listed"] / flipsl, 4) if flipsl else None,
            listed_frac_flips_other_sig=round(
                totl["other_sig"] / flipsl, 4) if flipsl else None,
            heuristic_cs_flip_draws=heur["cs_flip"],
            heuristic_cs_flip_identity_same_draws=heur[
                "cs_flip_identity_same_draws"],
            clean=dict(clean_listed))
        print(model, json.dumps(pooled[model]), flush=True)
    with open(HERE / "traffic_context.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "snr", "cls", "draws",
                                           "source"])
        w.writeheader(); w.writerows(rows)
    n_listed = Counter(len(v) for v in traffic_cache.values())
    pooled["_traffic_list_sizes"] = dict(sorted(n_listed.items()))
    (HERE / "traffic_context.json").write_text(json.dumps(pooled, indent=2))


if __name__ == "__main__":
    main()
