"""Digit-preserving transcript normalization (TMLR revision, reviewer 3CgC).

The campaign's RS.norm_text keeps only [a-z' ], so numerals in a raw decode
("Emirates 81", "flight level 320") are deleted.  ATC references and the CTC
models' letter vocabularies spell numbers as words, so the revision maps every
numeral to its digit-by-digit spoken form -- the ICAO radiotelephony
convention the references already follow -- before the letters-only filter:

    "Emirates 81, descend FL 320"  ->  "emirates eight one descend fl three two zero"
    "contact 121.5"                ->  "contact one two one decimal five"
    "climb 1,000 feet"             ->  "climb one zero zero zero feet"

On text that contains no digit the result is identical to RS.norm_text, so the
CTC models' readouts (and every certificate computed from them) are unchanged.
"""
import re

DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
               "eight", "nine"]
_KEEP = set("abcdefghijklmnopqrstuvwxyz' ")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_DECIMAL = re.compile(r"(?<=\d)\.(?=\d)")


def norm_text_digits(s):
    s = s.lower()
    s = _THOUSANDS.sub("", s)
    s = _DECIMAL.sub(" decimal ", s)
    s = "".join(f" {DIGIT_WORDS[ord(c) - 48]} " if "0" <= c <= "9" else c
                for c in s)
    return " ".join("".join(c if c in _KEEP else " " for c in s).split())


if __name__ == "__main__":
    for t in ["Emirates 81, descend FL 320.", "contact 121.5 bye",
              "climb 1,000 feet", "KLM 847 descend flight level one zero zero",
              "it's been climb"]:
        print(repr(t), "->", repr(norm_text_digits(t)))
