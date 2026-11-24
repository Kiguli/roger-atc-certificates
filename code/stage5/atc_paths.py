"""I/O locations for the ATC analysis scripts (TMLR revision, rev1).

Default (ATC_DATA_DIR unset): inputs and outputs live in Stage-5-atc/ and the
LaTeX macros / figures go to the sibling paper repo -- exactly the campaign's
original behaviour, byte for byte.

ATC_DATA_DIR=<dir>: every input is read from <dir> and every output is written
to <dir>; LaTeX macros and figures go to <dir>/figures.  The paper repo and the
campaign outputs in Stage-5-atc/ are never touched.

ATC_REV=1 (requires ATC_DATA_DIR): revision-mode extensions -- call-sign
identity spec (csid), ordered digit/NATO signature audit predicate,
digit-preserving WER references, multi-embedder audit.  See rev1/README_rev1.md.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER_FIGS = HERE.parents[1] / "Speech-to-Text-Verification" / "figures"

_dd = os.environ.get("ATC_DATA_DIR", "").strip()
OVERRIDE = bool(_dd)
DATA = Path(_dd).resolve() if OVERRIDE else HERE
FIGS = DATA / "figures" if OVERRIDE else PAPER_FIGS
REV = os.environ.get("ATC_REV", "0") == "1"

if OVERRIDE and DATA == HERE:
    sys.exit("ATC_DATA_DIR points at Stage-5-atc itself; unset it for the "
             "default (campaign) paths -- refusing to write to "
             "Stage-5-atc/figures")
if REV and not OVERRIDE:
    sys.exit("ATC_REV=1 requires ATC_DATA_DIR (revision outputs must never "
             "overwrite the campaign outputs or the paper's macros)")
if OVERRIDE:
    FIGS.mkdir(parents=True, exist_ok=True)
