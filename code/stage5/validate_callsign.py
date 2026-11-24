"""Validate the call-sign readouts against ATCO2's gold call-sign annotation.

The original ATCO2 1-hour release (ATCO2-ASRdataset-v1_beta, replaywell.com)
ships per-recording XML whose transcripts carry named-entity tags
([#callsign]..[/#callsign], [#command], [#value], [#unnamed]); 279 of its 877
segments are entity-tagged.  Each of our 857 ATCO2 utterances is matched to
its XML segment (recording-name prefix + segment start time), and on every
entity-tagged segment we compare, on the REFERENCE transcript:

  heuristic  RS.callsign(norm_text(ref))       -- the campaign's readout
  grammar    callsign_grammar.readout(norm_text_digits(ref))

against the gold span (the first maximal run of [#callsign] tokens).

Reported per readout: exact span match; span covers the gold call-sign;
signature match (ordered digit/phonetic content equal to the gold one -- the
identity content); gold call-sign non-initial; heuristic truncation; and, on
tagged segments WITHOUT a call-sign, whether the readout is (correctly) empty.

Output: callsign_validation.csv (per segment) + callsign_validation.json
Run:    python validate_callsign.py
"""
import csv
import glob
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "Stage-2-verification" / "scenario"))
sys.path.insert(0, str(HERE))
import run_scenario_datasets as RS                      # noqa: E402
import callsign_grammar as CG                           # noqa: E402
from atc_text import norm_text_digits                   # noqa: E402

DATA = CG.ATCO2_DIR
TAG = re.compile(r"\[(#\w+|/#\w+|NE[^\]]*|/NE)\]")


def labelled_tokens(xml_text):
    """XML transcript -> list of (canonical token, is_callsign)."""
    out, cur = [], None
    pos = 0
    for m in TAG.finditer(xml_text):
        chunk = xml_text[pos:m.start()]
        for t in CG.canon_tokens(norm_text_digits(chunk)):
            out.append((t, cur == "#callsign"))
        tag = m.group(1)
        if tag.startswith("#"):
            cur = tag
        elif tag.startswith("/#"):
            cur = None
        pos = m.end()
    for t in CG.canon_tokens(norm_text_digits(xml_text[pos:])):
        out.append((t, cur == "#callsign"))
    return out


def gold_span(lt):
    """First maximal run of call-sign tokens -> (start, end) or None."""
    s = None
    for i, (_, c) in enumerate(lt):
        if c and s is None:
            s = i
        if not c and s is not None:
            return s, i
    return (s, len(lt)) if s is not None else None


def contains(hay, needle):
    n = len(needle)
    return any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def main():
    segs = {}
    for f in sorted(glob.glob(str(DATA / "*.xml"))):
        rec = Path(f).stem
        for s in ET.parse(f).getroot().iter("segment"):
            segs.setdefault(rec, []).append(
                (float(s.find("start").text), s.find("text").text or ""))
    refs = list(csv.DictReader(open(ROOT / "external/stt/data/atco2/refs.csv",
                                    newline="", encoding="utf-8")))
    # The HF mirror truncates recording ids to 60 characters, so ids are
    # ambiguous; match on the transcript itself (tags stripped), preferring
    # segments with the same start time, and accept only near-identical text.
    from difflib import SequenceMatcher
    flat = [(rec, a, t, " ".join(tok for tok, _ in labelled_tokens(t)))
            for rec, ss in segs.items() for a, t in ss]
    rows, unmatched, used, seg_map = [], 0, Counter(), []
    for r in refs:
        st = float(r["segment_start_time"])
        rt = " ".join(CG.canon_tokens(norm_text_digits(r["text"])))
        best = max(flat, key=lambda z: (SequenceMatcher(None, rt, z[3]).ratio(),
                                        -abs(z[1] - st)))
        if SequenceMatcher(None, rt, best[3]).ratio() < 0.9:
            unmatched += 1
            continue
        rec, _, xt, _ = best
        used[(rec, best[1])] += 1
        seg_map.append(dict(file=r["file"], recording=rec, start=best[1]))
        if "[#" not in xt:
            continue                                 # not entity-tagged
        lt = labelled_tokens(xt)
        toks = [t for t, _ in lt]
        ref_old = RS.norm_text(r["text"])
        ref_new = norm_text_digits(r["text"])
        same_text = CG.canon_tokens(ref_new) == toks
        g = gold_span(lt)
        # both readouts read the SAME text (the XML transcript, tags stripped)
        heur = CG.canon_tokens(RS.callsign(" ".join(toks)))
        gp = CG.parse(" ".join(toks))
        gram = toks[gp[0]:gp[1]] if gp else []
        row = dict(file=r["file"], recording=rec, ref=r["text"],
                   ref_matches_xml=int(same_text), has_gold=int(g is not None),
                   gold="", gold_start=-1, heuristic=" ".join(heur),
                   grammar=" ".join(gram))
        if g is not None:
            gold = toks[g[0]:g[1]]
            gsig = CG.signature(gold)
            row.update(
                gold=" ".join(gold), gold_start=g[0],
                h_exact=int(heur == gold), h_covers=int(contains(heur, gold)),
                h_sig=int(CG.signature(heur) == gsig),
                h_truncated=int(g[0] == 0 and len(heur) < len(gold)
                                and gold[:len(heur)] == heur),
                h_superset=int(g[0] == 0 and len(heur) > len(gold)
                               and heur[:len(gold)] == gold),
                g_exact=int(gram == gold), g_covers=int(contains(gram, gold)),
                g_sig=int(CG.signature(gram) == gsig),
                g_found=int(bool(gram)))
        else:
            row.update(h_nonempty=int(bool(heur)), g_empty=int(not gram))
        rows.append(row)

    with open(HERE / "atco2_segment_map.csv", "w", newline="",
              encoding="utf-8") as fh:                # used by traffic analysis
        w = csv.DictWriter(fh, fieldnames=["file", "recording", "start"])
        w.writeheader(); w.writerows(seg_map)
    with open(HERE / "callsign_validation.csv", "w", newline="",
              encoding="utf-8") as fh:
        keys = sorted({k for r in rows for k in r},
                      key=lambda k: list(rows[0]).index(k)
                      if k in rows[0] else 99)
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(rows)

    G = [r for r in rows if r["has_gold"]]
    NG = [r for r in rows if not r["has_gold"]]

    def frac(rs, k):
        return round(sum(r[k] for r in rs) / len(rs), 4) if rs else None
    out = dict(
        n_atco2=len(refs), unmatched=unmatched, n_tagged=len(rows),
        duplicate_matches=sum(v > 1 for v in used.values()),
        n_gold=len(G), n_nogold=len(NG),
        ref_matches_xml=frac(rows, "ref_matches_xml"),
        gold_initial=sum(r["gold_start"] == 0 for r in G),
        gold_noninitial=sum(r["gold_start"] > 0 for r in G),
        gold_len_gt8=sum(len(r["gold"].split()) > 8 for r in G),
        heuristic=dict(exact=frac(G, "h_exact"), covers=frac(G, "h_covers"),
                       signature=frac(G, "h_sig"),
                       truncated=sum(r["h_truncated"] for r in G),
                       superset=sum(r["h_superset"] for r in G),
                       exact_initial=frac([r for r in G if r["gold_start"] == 0],
                                          "h_exact"),
                       sig_initial=frac([r for r in G if r["gold_start"] == 0],
                                        "h_sig"),
                       sig_noninitial=frac([r for r in G if r["gold_start"] > 0],
                                           "h_sig"),
                       nogold_nonempty=frac(NG, "h_nonempty")),
        grammar=dict(exact=frac(G, "g_exact"), covers=frac(G, "g_covers"),
                     signature=frac(G, "g_sig"), found=frac(G, "g_found"),
                     sig_initial=frac([r for r in G if r["gold_start"] == 0],
                                      "g_sig"),
                     sig_noninitial=frac([r for r in G if r["gold_start"] > 0],
                                         "g_sig"),
                     nogold_empty=frac(NG, "g_empty")),
        recordings=len({r["recording"] for r in rows}),
        airports=dict(Counter(r["recording"].split("_")[0] for r in rows)),
    )
    (HERE / "callsign_validation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
