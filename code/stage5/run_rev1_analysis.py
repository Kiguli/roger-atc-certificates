"""Regenerate every revision (rev1) analysis output in one go.

    python run_rev1_analysis.py                 # everything, partial data OK
    python run_rev1_analysis.py --require-complete   # refuse partial reruns
    python run_rev1_analysis.py --steps perinput,figs_pi   # a subset
    python run_rev1_analysis.py --skip convert,embedders
    python run_rev1_analysis.py --original whisper_atc,deepspeech2  # DS2 rerun dropped
    python run_rev1_analysis.py --patch-tag _numerals  # whisper_atc: 2 numeral msgs

Every step runs with ATC_DATA_DIR=<Stage-5-atc>/rev1 ATC_REV=1 (atc_paths.py),
so inputs/outputs live in rev1/ and macros/figures in rev1/figures/; the
campaign outputs in Stage-5-atc/ and the paper repo are never written.
Each step's stdout/stderr is also saved to rev1/logs/<step>.log.
See rev1/README_rev1.md for which files feed which macros.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REV1 = HERE / "rev1"
PY = sys.executable

STEPS = [
    ("convert", ["convert_v2.py"]),            # --out added in main
    ("wer", ["compute_wer.py"]),
    ("perinput", ["analyze_atc_perinput.py"]),
    ("semaudit", ["analyze_atc_semaudit.py"]),
    ("witnesses", ["scan_semaudit_witnesses.py"]),
    ("embedders", ["audit_embedders.py"]),
    ("emit_semaudit", ["emit_semaudit_macros.py"]),
    ("figs_pi", ["make_perinput_figs.py"]),
    ("stt", ["make_stt_paper_figs.py"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="",
                    help="comma list (default: all): "
                    + ",".join(s for s, _ in STEPS))
    ap.add_argument("--skip", default="", help="comma list of steps to skip")
    ap.add_argument("--require-complete", action="store_true",
                    help="abort after convert unless every v2 rerun "
                         "(quartznet, deepspeech2, whisper x 2 corpora) is "
                         "complete")
    ap.add_argument("--original", default="whisper_atc",
                    help="models whose campaign files are copied instead of "
                         "converting a v2 dump (convert_v2.py --original); "
                         "e.g. whisper_atc,deepspeech2 if the DS2 TF32-off "
                         "rerun is dropped")
    ap.add_argument("--patch-tag", default="",
                    help="convert_v2.py --patch-tag, e.g. _numerals: replace "
                         "the --original models' utterances found in a "
                         "targeted v2 rerun (default: off)")
    ap.add_argument("--out", default=str(REV1),
                    help="output directory (default Stage-5-atc/rev1)")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    STEPS[0] = ("convert", STEPS[0][1] + ["--out", str(out),
                                          "--original", a.original,
                                          "--patch-tag", a.patch_tag])
    want = [s for s in a.steps.split(",") if s] or [s for s, _ in STEPS]
    skip = {s for s in a.skip.split(",") if s}
    unknown = (set(want) | skip) - {s for s, _ in STEPS}
    if unknown:
        sys.exit(f"unknown step(s): {sorted(unknown)}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    env = dict(os.environ, ATC_DATA_DIR=str(out), ATC_REV="1",
               PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    for name, cmd in STEPS:
        if name not in want or name in skip:
            continue
        t0 = time.time()
        print(f"=== {name}: {' '.join(cmd)}", flush=True)
        with open(out / "logs" / f"{name}.log", "w", encoding="utf-8") as lg:
            p = subprocess.Popen([PY] + cmd, cwd=HERE, env=env,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace")
            for line in p.stdout:
                lg.write(line)
                if "Warning" not in line and "warnings.warn" not in line \
                        and "it/s]" not in line:
                    print("   " + line, end="", flush=True)
            rc = p.wait()
        print(f"=== {name}: exit {rc} ({time.time() - t0:.0f}s)", flush=True)
        if rc:
            sys.exit(f"step {name} failed (see {out / 'logs' / name}.log)")
        if name == "convert" and a.require_complete:
            man = json.loads((out / "convert_manifest.json").read_text())
            bad = [k for k, v in man["cells"].items()
                   if v.get("kind") != "original_copied"
                   and not v.get("complete")]
            if bad:
                sys.exit(f"incomplete v2 reruns: {bad} -- rerun without "
                         f"--require-complete for a partial analysis")
    print("rev1 analysis done ->", out)


if __name__ == "__main__":
    main()
