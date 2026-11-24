"""Regenerate every number and figure in the paper from the released data,
and check each one against the camera-ready version.

    python reproduce.py              # analysis tier (minutes to ~1 h, CPU is fine)
    python reproduce.py --from-raw   # also rebuild the revision per-draw CSVs from
                                     # the raw v2 decode dumps (convert step)
    python reproduce.py --keep       # keep build/ from a previous run, re-check only

What it does
------------
1. Builds ``build/`` with the directory layout the campaign used
   (``build/RadioStar/Stage-5-atc`` etc.).  It copies the code in ``code/``
   VERBATIM and decompresses the data in ``data/`` next to it, so the scripts
   run unmodified: these are the exact files that produced the paper.
2. Deletes, inside ``build/``, every file a step is about to produce, so a
   step that silently writes nothing cannot pass.
3. Runs the analysis steps in the order the paper used.
4. Compares every produced file with
     * ``expected/``  -- the macro files and figures the camera-ready
       ``\\input``s / ``\\includegraphics``-es, copied from the paper source;
     * ``data/``      -- the released intermediate files (summaries, per-input
       certificates, ...), which must also regenerate exactly.
   Text files are compared byte for byte after normalising line endings.
   PDFs are compared after removing the /CreationDate stamp; if that fails and
   PyMuPDF is installed, they are rasterised and compared pixel by pixel.

Exit status 0 means every compared file matched.
"""
import argparse
import gzip
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
BUILD = REPO / "build"
RS = BUILD / "RadioStar"
S5 = RS / "Stage-5-atc"
R1 = S5 / "rev1"
PAPER = BUILD / "Speech-to-Text-Verification"
EXP = REPO / "expected"

# The convert options used for the camera-ready (rev1/convert_manifest.json).
CONVERT_ARGS = ["--original", "whisper_atc,deepspeech2",
                "--patch-tag", "_numerals,_tf32fix"]


# ---------------------------------------------------------------------------
# build tree
# ---------------------------------------------------------------------------
def unpack(src_dir: Path, dst_dir: Path):
    for p in sorted(src_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(src_dir)
        if p.name.endswith(".gz") and not p.name.endswith(".jsonl.gz"):
            out = dst_dir / str(rel)[:-3]
            out.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(p, "rb") as fi, open(out, "wb") as fo:
                shutil.copyfileobj(fi, fo)
        else:
            out = dst_dir / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)


def build_tree():
    if BUILD.exists():
        shutil.rmtree(BUILD)
    print("building", BUILD.relative_to(REPO), "...", flush=True)
    for p in (REPO / "code/stage5").glob("*.py"):
        dst = S5 / p.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    for p in (REPO / "code/scenario").glob("*.py"):
        dst = RS / "Stage-2-verification/scenario" / p.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    for p in (REPO / "code/radiostar").glob("*.py"):
        dst = RS / "src/radiostar" / p.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    unpack(REPO / "data/campaign", S5)
    unpack(REPO / "data/rev1", R1)
    # June baseline run (independent noise stream) -> redraw ablation
    unpack(REPO / "data/scenario", RS / "Stage-2-verification/scenario")
    links = R1 / "SAME_AS_CAMPAIGN.txt"   # rev1 files identical to campaign ones
    if links.exists():
        for name in links.read_text().split():
            shutil.copy2(S5 / name, R1 / name)
        links.unlink()
    ext = RS / "external/stt/telephony"
    ext.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / "data/external/airlines.dat", ext / "airlines.dat")
    models = REPO / "models"          # from code/fetch_models.py (GPU campaign only)
    if models.exists():
        shutil.copytree(models, RS / "external/stt/models", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("_downloads"))
    atco2 = REPO / "data/external/atco2_info"
    if atco2.exists():
        dst = RS / "external/stt/atco2_v1_beta/ATCO2-ASRdataset-v1_beta/DATA"
        dst.mkdir(parents=True, exist_ok=True)
        for p in atco2.glob("*.info"):
            shutil.copy2(p, dst / p.name)
    # output directories the scripts write into
    for d in (S5 / "figures", RS / "Stage-4-final/figures",
              RS / "Stage-4-final/slides/figures", PAPER / "figures",
              PAPER / "figures_rev", R1 / "figures", R1 / "logs"):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# steps: (name, argv, env, files produced -> what to compare them with)
# ---------------------------------------------------------------------------
def steps(from_raw: bool):
    rev = {"ATC_DATA_DIR": str(R1), "ATC_REV": "1"}
    camp_atc = ["cert_summary.csv", "validation_oos.csv", "ablation_tau.csv",
                "ablation_beta.csv", "ablation_N.csv", "ablation_M.csv",
                "ablation_redraw.csv", "ablation_redraw_summary.csv",
                "atc_summary.json"]
    S = [
        # ---- original campaign (Sections 5-6, held-out table) -------------
        ("analyze_atc", ["analyze_atc.py"], {},
         {S5 / f: ("data", f) for f in camp_atc}),
        ("atc_figs", ["make_atc_split_figs.py"], {},
         {S5 / "figures/numbers_atc.tex": ("exp", "figures/numbers_atc.tex")}),
        ("heldout", ["emit_heldout_macros.py"], {},
         {PAPER / "figures/numbers_heldout.tex":
          ("exp", "figures/numbers_heldout.tex")}),
        # ---- revision analyses feeding numbers_revision.tex ---------------
        ("crosspass", ["crosspass_stats.py"], {},
         {S5 / "crosspass_stats.json": ("data", "crosspass_stats.json")}),
        ("zeronoise", ["analyze_zeronoise.py"], {},
         {S5 / "zeronoise_summary.json": ("data", "zeronoise_summary.json"),
          S5 / "zeronoise_summary.csv": ("data", "zeronoise_summary.csv")}),
        ("qn_impact", ["quick_qn_impact.py"], {},
         {S5 / "quick_qn_impact.json": ("data", "quick_qn_impact.json")}),
        ("revision_macros", ["emit_revision_macros.py"], {},
         {PAPER / "figures_rev/numbers_revision.tex":
          ("exp", "figures_rev/numbers_revision.tex")}),
    ]
    # ---- revision campaign (run_rev1_analysis.py, one step at a time) -----
    r = []
    if from_raw:
        r.append(("rev1:convert", ["convert_v2.py", "--out", str(R1)]
                  + CONVERT_ARGS, rev,
                  {R1 / f: ("rev1", f) for f in rev1_converted()}))
    r += [
        ("rev1:perinput", ["analyze_atc_perinput.py"], rev,
         {R1 / f: ("rev1", f) for f in ("per_input_cp.csv",
                                         "perinput_summary.csv",
                                         "snr_budget.csv", "numbers_pi.json")}
         | {R1 / "figures/numbers_pi.tex": ("exp", "figures_rev/numbers_pi.tex")}),
        ("rev1:semaudit", ["analyze_atc_semaudit.py"], rev,
         {R1 / f: ("rev1", f) for f in ("semaudit_perinput.csv",
                                         "semaudit_summary.csv",
                                         "semaudit_examples.json",
                                         "numbers_semaudit.json")}),
        ("rev1:witnesses", ["scan_semaudit_witnesses.py"], rev,
         {R1 / "numbers_witness.json": ("rev1", "numbers_witness.json")}),
        ("rev1:embedders", ["audit_embedders.py"], rev,
         {R1 / f: ("rev1", f) for f in ("embedder_audit.json",
                                         "embedder_audit_cells.csv")}),
        ("rev1:emit_semaudit", ["emit_semaudit_macros.py"], rev,
         {R1 / "figures/numbers_semaudit.tex":
          ("exp", "figures_rev/numbers_semaudit.tex")}),
        ("rev1:figs_pi", ["make_perinput_figs.py"], rev,
         {R1 / f"figures/{n}.pdf": ("exp", f"figures_rev/{n}.pdf")
          for n in ("fig_pi_bands", "fig_pi_bands_sem", "fig_pi_budget",
                    "fig_pi_certfrac", "fig_pi_certfrac_id",
                    "fig_pi_certfrac_sem", "fig_pi_semaudit", "fig_pi_wer")}),
        ("rev1:stt", ["make_stt_paper_figs.py"], rev,
         {R1 / "figures/numbers_stt.tex": ("exp", "figures_rev/numbers_stt.tex"),
          R1 / "figures/table_models.tex": ("exp", "figures_rev/table_models.tex")}),
    ]
    return S + r


def rev1_converted():
    out = []
    for m in ("quartznet", "deepspeech2", "whisper", "whisper_atc"):
        for d in ("atcosim", "atco2"):
            out += [f"results_atc_{m}_{d}.csv", f"results_sem_{m}_{d}.csv"]
    return out


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------
_CD = re.compile(rb"/CreationDate \(D:[^)]*\)")


def reference(kind, name):
    if kind == "exp":
        return EXP / name
    base = REPO / "data" / ("campaign" if kind == "data" else "rev1")
    for cand in (base / (name + ".gz"), base / name):
        if cand.exists():
            return cand
    links = REPO / "data/rev1/SAME_AS_CAMPAIGN.txt"
    if kind == "rev1" and links.exists() and name in links.read_text().split():
        return reference("data", name)
    return base / name


def read(p: Path) -> bytes:
    b = gzip.open(p).read() if p.suffix == ".gz" else p.read_bytes()
    return b


def same(prod: Path, ref: Path):
    a, b = read(prod), read(ref)
    if prod.suffix == ".pdf":
        if _CD.sub(b"", a) == _CD.sub(b"", b):
            return True, "identical (ignoring /CreationDate)"
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return False, "PDF bytes differ (install PyMuPDF for a pixel check)"
        da, db = fitz.open(stream=a, filetype="pdf"), fitz.open(stream=b, filetype="pdf")
        if len(da) != len(db):
            return False, "page count differs"
        for pa, pb in zip(da, db):
            ia, ib = pa.get_pixmap(dpi=200), pb.get_pixmap(dpi=200)
            if (ia.width, ia.height) != (ib.width, ib.height) or ia.samples != ib.samples:
                return False, "rendered pixels differ"
        return True, "pixel-identical at 200 dpi"
    a, b = a.replace(b"\r\n", b"\n"), b.replace(b"\r\n", b"\n")
    if a == b:
        return True, "identical"
    la, lb = a.decode("utf-8", "replace").splitlines(), b.decode("utf-8", "replace").splitlines()
    diff = [i for i in range(max(len(la), len(lb)))
            if (la[i] if i < len(la) else None) != (lb[i] if i < len(lb) else None)]
    i = diff[0]
    return False, (f"{len(diff)} line(s) differ; first at line {i + 1}:\n"
                   f"      produced: {la[i] if i < len(la) else '<missing>'}\n"
                   f"      expected: {lb[i] if i < len(lb) else '<missing>'}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-raw", action="store_true",
                    help="also rerun the convert step from the raw v2 dumps")
    ap.add_argument("--keep", action="store_true",
                    help="reuse build/ and only re-check (no steps are run)")
    ap.add_argument("--only", default="", help="comma list of step names to run")
    a = ap.parse_args()

    plan = steps(a.from_raw)
    only = {s for s in a.only.split(",") if s}
    if not a.keep:
        build_tree()
        # remove every file a step will produce, so nothing passes by default
        for name, _, _, outs in plan:
            if only and name not in only:
                continue
            for p in outs:
                if p.exists():
                    p.unlink()
        env0 = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
                    MPLBACKEND="Agg", SOURCE_DATE_EPOCH="0")
        logs = BUILD / "logs"
        logs.mkdir(exist_ok=True)
        for name, argv, env, _ in plan:
            if only and name not in only:
                continue
            t0 = time.time()
            print(f"=== {name}: python {' '.join(argv)}", flush=True)
            with open(logs / f"{name.replace(':', '_')}.log", "w",
                      encoding="utf-8") as lg:
                rc = subprocess.call([sys.executable] + argv, cwd=S5,
                                     env=dict(env0, **env), stdout=lg,
                                     stderr=subprocess.STDOUT)
            print(f"    exit {rc}  ({time.time() - t0:.0f}s)", flush=True)
            if rc:
                print(f"    see {logs / (name.replace(':', '_') + '.log')}")
                sys.exit(1)

    # ---- check -------------------------------------------------------------
    print("\n=== checking against the camera-ready and the released data")
    bad = n = 0
    for name, _, _, outs in plan:
        if only and name not in only:
            continue
        for prod, (kind, ref) in outs.items():
            n += 1
            refp = reference(kind, ref)
            if not prod.exists():
                ok, msg = False, "NOT PRODUCED"
            elif not refp.exists():
                ok, msg = False, f"reference missing: {refp}"
            else:
                ok, msg = same(prod, refp)
            bad += not ok
            tag = "paper" if kind == "exp" else "data "
            print(f"  [{'OK ' if ok else 'BAD'}] {tag} {ref:42s} {msg}")
    print(f"\n{n - bad}/{n} files reproduced exactly.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
