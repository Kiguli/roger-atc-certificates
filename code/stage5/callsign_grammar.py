"""Grammar-based call-sign readout (TMLR revision, reviewer 3CgC).

The campaign's RS.callsign is a positional heuristic: the utterance-initial
token run before the first of 73 command words, capped at 8 tokens.  This
module adds a grammar readout that

  * searches the WHOLE utterance (non-initial call-signs, e.g. after a
    station name or greeting, are found);
  * recognizes the two ICAO call-sign forms
      (a) <telephony designator> <flight number>    "lufthansa four two two three"
      (b) <registration in phonetic letters/digits>  "oscar kilo papa mike bravo"
    where the flight number / registration is a run of digit words, NATO
    phonetic letters and the multipliers "double"/"triple";
  * returns an explicit EMPTY readout when no call-sign is present, instead
    of an arbitrary leading span; and
  * exposes the identity SIGNATURE -- the ordered digit and phonetic-letter
    content of the parsed call-sign, with multipliers expanded -- which is
    what distinguishes one aircraft from another.

Telephony designators come from two sources independent of every transcript:
the OpenFlights airline table (ICAO telephony column, 6,000+ airlines) and the
surveillance-derived "callsigns nearby" lists shipped with the ATCO2 release
(the spoken designators of the traffic near each airport at recording time --
in operation, the active traffic's call-signs are known from flight plans).

Input is a normalized transcript (atc_text.norm_text_digits or RS.norm_text).
"""
import csv
import glob
import re
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OPENFLIGHTS = ROOT / "external/stt/telephony/airlines.dat"
ATCO2_DIR = ROOT / "external/stt/atco2_v1_beta/ATCO2-ASRdataset-v1_beta/DATA"

DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
          "eight", "nine"]
DIGIT_SET = set(DIGITS) | {"niner", "hundred", "thousand"}
NATO = ["alfa", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
        "hotel", "india", "juliett", "kilo", "lima", "mike", "november",
        "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform",
        "victor", "whiskey", "xray", "yankee", "zulu"]
NATO_SET = set(NATO)
CANON = {"alpha": "alfa", "juliet": "juliett", "niner": "nine",
         "whisky": "whiskey", "x-ray": "xray"}
MULT = {"double": 2, "triple": 3}
# tokens after which a phonetic/digit run is a VALUE, not a registration
VALUE_KEYS = {
    "level", "heading", "runway", "squawk", "qnh", "information", "decimal",
    "speed", "altitude", "feet", "knots", "point", "taxiway", "via", "gate",
    "stand", "frequency", "flight", "climb", "descend", "maintain", "contact",
    "wind", "degrees", "time", "check", "minutes", "minute", "miles", "to",
    "holding", "approach", "on", "at", "of", "from", "for", "and", "track",
    "radial", "intersection", "direct", "route", "sid", "star", "transition",
    "departure", "arrival", "pressure", "temperature", "visibility", "cloud",
    "turn", "left", "right", "behind", "after", "number", "sector", "apron",
    "parking", "position", "code", "transponder", "reduce", "increase",
    "expect", "cleared", "proceed", "report", "passing", "leaving", "fl",
    # circuit / sequencing positions ("final two five", "number two")
    "final", "downwind", "base", "crosswind", "upwind", "next", "discretion"}
STOP_DESIG = (DIGIT_SET | NATO_SET | set(CANON) | set(MULT) | VALUE_KEYS | {
    "radar", "tower", "ground", "control", "good", "morning", "afternoon",
    "evening", "day", "hello", "bye", "goodbye", "roger", "thank", "thanks",
    "you", "the", "a", "an", "is", "are", "this", "that", "with", "in", "now",
    "please", "say", "again", "go", "ahead", "affirm", "negative", "correct",
    "standby", "wilco", "unable", "request", "confirm", "set", "line", "up",
    "wait", "hold", "short", "clear", "continue", "vacate", "taxi", "push",
    "start", "copy", "traffic", "service", "radio", "check", "loud", "air",
    "express", "jet", "charter", "cargo", "united", "international"})
SUFFIX = {"heavy"}
try:                                   # the campaign's 73 command words
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "Stage-2-verification" / "scenario"))
    from run_scenario_datasets import COMMAND_WORDS as COMMAND
except Exception:                      # noqa: BLE001
    COMMAND = set()


def canon_tokens(text):
    """Normalized text -> canonical tokens (alpha->alfa, 'x ray'->xray, ...)."""
    toks = text.split()
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t == "x" and i + 1 < len(toks) and toks[i + 1] == "ray":
            out.append("xray"); i += 2; continue
        out.append(CANON.get(t, t)); i += 1
    return out


def _desig_key(words):
    return re.sub(r"[^a-z]", "", "".join(words).lower())


@lru_cache(maxsize=1)
def designators():
    """Set of designator keys (spaces/punctuation removed), with max length
    in tokens.  'speedbird' matches 'speed bird'; 'usair' matches 'u s air'."""
    keys = set()
    if OPENFLIGHTS.exists():
        for row in csv.reader(open(OPENFLIGHTS, encoding="utf-8",
                                   errors="replace")):
            if len(row) > 5 and row[5] not in ("", "\\N"):
                keys.add(_desig_key(row[5].split()))
    for info in glob.glob(str(ATCO2_DIR / "*.info")):
        for line in open(info, encoding="utf-8", errors="replace"):
            m = re.match(r"^\s*[A-Z0-9]+\s*:\s*(.+)$", line)
            if not m:
                continue
            words = []
            for w in m.group(1).lower().replace("/", " / ").split():
                w = CANON.get(w, w)
                if w in DIGIT_SET or w in NATO_SET or w in MULT:
                    break
                words.append(w)
            # "Csa / Czech One Delta Zulu": alternatives separated by "/"
            alt = []
            for w in words + ["/"]:
                if w == "/":
                    if alt:
                        keys.add(_desig_key(alt))
                    alt = []
                else:
                    alt.append(w)
    return frozenset(k for k in keys if len(k) >= 3 and k not in STOP_DESIG)


def _run(toks, j):
    """Length of the identifier run starting at j (digits, NATO, multipliers)."""
    k = j
    while k < len(toks):
        t = toks[k]
        if t in DIGIT_SET or t in NATO_SET:
            k += 1
        elif t in MULT and k + 1 < len(toks) and (toks[k + 1] in DIGIT_SET
                                                  or toks[k + 1] in NATO_SET):
            k += 1
        else:
            break
    return k - j


def parse(text, max_desig=3):
    """-> (start, end, designator_len) token span of the first call-sign in
    the canonical token list of `text`, or None."""
    toks = canon_tokens(text)
    D = designators()
    for i in range(len(toks)):
        # (a) telephony designator + identifier run (longest designator first)
        for w in range(max_desig, 0, -1):
            if i + w > len(toks):
                continue
            key = _desig_key(toks[i:i + w])
            if key in D and key not in STOP_DESIG and \
                    (w > 1 or toks[i] not in STOP_DESIG):
                r = _run(toks, i + w)
                if r >= 1:
                    e = i + w + r
                    if e < len(toks) and toks[e] in SUFFIX:
                        e += 1
                    return i, e, w
        # (c) unlisted designator: one non-keyword word, not introduced by a
        #     value keyword, followed by an identifier run of >= 2 tokens
        #     containing a digit ("csa one delta zulu")
        if toks[i] not in STOP_DESIG and toks[i] not in COMMAND and \
                (i == 0 or toks[i - 1] not in VALUE_KEYS):
            r = _run(toks, i + 1)
            if r >= 2 and any(t in DIGIT_SET for t in toks[i + 1:i + 1 + r]):
                e = i + 1 + r
                if e < len(toks) and toks[e] in SUFFIX:
                    e += 1
                return i, e, 1
        # (b) registration: identifier run opening with a phonetic letter,
        #     >= 2 tokens, not introduced by a value keyword.  A run opening
        #     with >= 3 letters is a letters-only registration (HB-XXX, OK-XXX)
        #     and stops at the first digit, which begins a value
        #     ("hotel bravo golf kilo echo one four thousand feet")
        if toks[i] in NATO_SET and (i == 0 or toks[i - 1] not in VALUE_KEYS):
            r = _run(toks, i)
            nl = 0
            while nl < r and toks[i + nl] in NATO_SET:
                nl += 1
            if nl >= 3:
                r = nl
            if r >= 2:
                return i, i + r, 0
        # (d) abbreviated call-sign: an utterance-initial run of >= 3 digits
        #     ("three four two six joining downwind")
        if i == 0:
            r = _run(toks, 0)
            if r >= 3 and all(t in DIGIT_SET for t in toks[:r]):
                return 0, r, 0
    return None


def signature(tokens):
    """Ordered digit/phonetic content with multipliers expanded."""
    sig, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        if t in MULT and i + 1 < len(tokens):
            sig.extend([tokens[i + 1]] * MULT[t]); i += 2; continue
        if t in DIGIT_SET or t in NATO_SET:
            sig.append(t)
        i += 1
    return tuple(sig)


def readout(text):
    """Grammar call-sign readout: canonical span string ('' if none)."""
    toks = canon_tokens(text)
    p = parse(text)
    return "" if p is None else " ".join(toks[p[0]:p[1]])


def identity(text):
    """Identity readout used by the call-sign identity specification:
    (designator key, signature) of the first parsed call-sign, or None.
    The designator key drops spaces, so 'speed bird' == 'speedbird'."""
    toks = canon_tokens(text)
    p = parse(text)
    if p is None:
        return None
    s, e, w = p
    return (_desig_key(toks[s:s + w]), signature(toks[s + w:e]))


if __name__ == "__main__":
    print(len(designators()), "designators")
    for t in ["lufthansa four two two three descend to flight level two seven zero",
              "ruzyne tower hello again euro wings one tango kilo",
              "oscar kilo triple hotel please confirm one more holding",
              "taxi to holding point runway two seven via alfa charlie",
              "contact milan one three four five two good bye",
              "speed bird seven seven whiskey whiskey is identified",
              "roger thank you",
              "csa one delta zulu descend flight level one hundred"]:
        print(repr(t), "->", repr(readout(t)), identity(t))
