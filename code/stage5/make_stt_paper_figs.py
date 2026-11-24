"""Figures for the standalone Speech-to-Text-Verification paper.

Reads Stage-5-atc analysis outputs (cert_summary.csv, wer_summary.csv) and
writes into the sibling repo Speech-to-Text-Verification/figures/:

  fig_wer_vs_g.pdf   accuracy vs certified robustness: WER (clean decode vs
                     reference) against G@60dB (message spec), 4 models x
                     2 corpora -- the "separate axes" hero figure.
  fig_g_curves.pdf   certified G across the SNR grid, one panel per corpus,
                     color = model, solid = message spec, dashed = call-sign.
  fig_g_msg.pdf      exact-message spec only: 1x2 (ATCOSIM | ATCO2), 4 models.
  fig_g_cs.pdf       call-sign spec only: same layout.
  fig_g_by_spec.pdf  2x2 grid (rows = spec, cols = corpus) -- the combined
                     presentation with one spec per row, 4 lines per panel.

Only the two pre-registered ATC specs (msg, cs) exist in this campaign; the
semantic-equivalence oracle was run for LibriSpeech only and is not derivable
post hoc here (per-draw transcripts are not stored, only counts/bitstrings).

Style matches the RadioStar figure conventions (Okabe-Ito, serif, 9pt).
The four model hues pass pairwise CVD checks (worst-case OKLab dE>=8); the
per-model marker shapes are the secondary encoding so identity is not
color-alone in print. Run:  python make_stt_paper_figs.py [outdir]

ATC_DATA_DIR=<dir>: inputs from <dir>, outputs to <dir>/figures (an explicit
[outdir] still wins).  Revision mode (ATC_REV=1): the two-level G inputs
(cert_summary*.csv from analyze_atc.py) are optional -- without them only
numbers_stt.tex (the \\Wer macros) and table_models.tex (G columns "--") are
written; models without a rev CSV yet are left out of the table.
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atc_paths import DATA, FIGS, REV

HERE = DATA                    # inputs: Stage-5-atc/ unless ATC_DATA_DIR
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else FIGS

C_MODEL = {"quartznet": "#0072B2", "deepspeech2": "#D55E00",
           "whisper": "#009E73", "whisper_atc": "#E69F00"}
MODEL_NAME = {"quartznet": "QuartzNet", "deepspeech2": "DeepSpeech2",
              "whisper": "Whisper-base", "whisper_atc": "Whisper-ATC"}
MK_DS = {"atcosim": "o", "atco2": "s"}
SNRS = [60, 50, 45, 40, 35, 30, 25, 20]

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight"})


def load(name):
    return list(csv.DictReader(open(HERE / name, newline="")))


def fig_wer_vs_g(cert, wer):
    g60 = {(r["model"], r["dataset"]): float(r["frac_robust_lo"])
           for r in cert if r["snr"] == "60" and r["spec"] == "msg"}
    w = {(r["model"], r["dataset"]): float(r["wer"]) for r in wer}
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    for m in C_MODEL:
        for ds in MK_DS:
            x, y = w[(m, ds)], g60[(m, ds)]
            ax.plot(x, y, MK_DS[ds], ms=7, mfc=C_MODEL[m], mec=C_MODEL[m],
                    alpha=0.9)
    # connect the two corpora per model with a faint line
    for m in C_MODEL:
        ax.plot([w[(m, d)] for d in MK_DS], [g60[(m, d)] for d in MK_DS],
                "-", lw=0.8, color=C_MODEL[m], alpha=0.35)
    ax.set_xlabel("word error rate on clean audio (lower is better)")
    ax.set_ylabel("certified robust fraction $G$ at 60 dB")
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1)
    h = [plt.Line2D([], [], color=C_MODEL[m], marker="o", ls="", ms=6,
                    label=MODEL_NAME[m]) for m in C_MODEL]
    h += [plt.Line2D([], [], color="0.3", marker=MK_DS[d], ls="", ms=6,
                     mfc="none", label=d.upper()) for d in MK_DS]
    ax.legend(handles=h, frameon=False, fontsize=7, loc="upper right")
    fig.savefig(OUT / "fig_wer_vs_g.pdf")
    fig.savefig(OUT / "_fig_wer_vs_g.png")
    plt.close(fig)


def fig_g_curves(cert):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), sharey=True)
    for ax, ds in zip(axes, MK_DS):
        for m in C_MODEL:
            for spec, ls in (("msg", "-"), ("cs", "--")):
                ys = [float(r["frac_robust_lo"]) for s in SNRS
                      for r in cert
                      if (r["model"], r["dataset"], r["spec"],
                          int(r["snr"])) == (m, ds, spec, s)]
                ax.plot(SNRS, ys, ls, marker="o", ms=2.5, lw=1.2,
                        color=C_MODEL[m])
        ax.set_xlim(62, 18)                    # noise grows to the right
        ax.set_ylim(-0.02, 1)
        ax.set_xlabel("SNR (dB)")
        ax.set_title(ds.upper(), fontsize=9)
    axes[0].set_ylabel("certified robust fraction $G$")
    h = [plt.Line2D([], [], color=C_MODEL[m], lw=1.4, label=MODEL_NAME[m])
         for m in C_MODEL]
    h += [plt.Line2D([], [], color="0.3", lw=1.2, ls="-", label="message spec"),
          plt.Line2D([], [], color="0.3", lw=1.2, ls="--", label="call-sign spec")]
    axes[1].legend(handles=h, frameon=False, fontsize=7, loc="upper right")
    fig.savefig(OUT / "fig_g_curves.pdf")
    fig.savefig(OUT / "_fig_g_curves.png")
    plt.close(fig)


MK_MODEL = {"quartznet": "o", "deepspeech2": "s", "whisper": "^",
            "whisper_atc": "D"}
SPEC_NAME = {"msg": "exact-message spec", "cs": "call-sign spec"}


def _g_panel(ax, cert, ds, spec):
    """One panel: G vs SNR for the 4 models, fixed hue+marker per model."""
    for m in C_MODEL:
        ys = [float(r["frac_robust_lo"]) for s in SNRS for r in cert
              if (r["model"], r["dataset"], r["spec"], int(r["snr"]))
              == (m, ds, spec, s)]
        ax.plot(SNRS, ys, "-", marker=MK_MODEL[m], ms=3.4, lw=1.3,
                color=C_MODEL[m], mfc="white", mew=1.0, zorder=3)
    ax.axhline(0.9, color="0.75", lw=0.8, ls=":", zorder=1)
    ax.set_xlim(62, 18)                        # noise grows to the right
    ax.set_ylim(-0.02, 1.02)


def _model_handles():
    return [plt.Line2D([], [], color=C_MODEL[m], marker=MK_MODEL[m], ms=5,
                       mfc="white", mew=1.0, lw=1.3, label=MODEL_NAME[m])
            for m in C_MODEL]


def fig_g_per_spec(cert):
    """fig_g_msg.pdf / fig_g_cs.pdf: one spec, 1x2 panels (corpora)."""
    for spec, fname in (("msg", "fig_g_msg"), ("cs", "fig_g_cs")):
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), sharey=True)
        for ax, ds in zip(axes, MK_DS):
            _g_panel(ax, cert, ds, spec)
            ax.set_xlabel("SNR (dB)")
            ax.set_title(ds.upper(), fontsize=9)
        axes[0].set_ylabel("certified $\\tau$-robust fraction $G$")
        fig.legend(handles=_model_handles(), ncol=4, frameon=False,
                   fontsize=7.5, loc="lower center",
                   bbox_to_anchor=(0.5, 0.97))
        fig.savefig(OUT / f"{fname}.pdf")
        fig.savefig(OUT / f"_{fname}.png")
        plt.close(fig)


def fig_g_grid(cert):
    """fig_g_by_spec.pdf: 2x2 -- rows = spec, cols = corpus, 4 lines/panel."""
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.2), sharex=True,
                             sharey=True)
    for i, spec in enumerate(("msg", "cs")):
        for j, ds in enumerate(MK_DS):
            ax = axes[i][j]
            _g_panel(ax, cert, ds, spec)
            if i == 0:
                ax.set_title(ds.upper(), fontsize=9)
            if i == 1:
                ax.set_xlabel("SNR (dB)")
            if j == 0:
                ax.set_ylabel(f"{SPEC_NAME[spec]}\n"
                              "certified $\\tau$-robust fraction $G$",
                              fontsize=8.5)
    fig.legend(handles=_model_handles(), ncol=4, frameon=False,
               fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 0.965))
    fig.savefig(OUT / "fig_g_by_spec.pdf")
    fig.savefig(OUT / "_fig_g_by_spec.png")
    plt.close(fig)


FLOOR_NAME = {"sem90": "semantic floor $\\chi_{\\rm sem}=0.90$",
              "sem95": "semantic floor $\\chi_{\\rm sem}=0.95$"}


def fig_g_sem(certsem):
    """fig_g_sem.pdf: 2x2 -- rows = semantic floor, cols = corpus.  The
    exact-match curve from the SAME decode pass is drawn faint underneath, so
    the vertical gap is exactly what the semantic spec buys on those draws."""
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.2), sharex=True,
                             sharey=True)
    for i, spec in enumerate(("sem90", "sem95")):
        for j, ds in enumerate(MK_DS):
            ax = axes[i][j]
            for m in C_MODEL:
                ex = [float(r["frac_robust_lo"]) for s in SNRS for r in certsem
                      if (r["model"], r["dataset"], r["spec"], int(r["snr"]))
                      == (m, ds, "exact_repro", s)]
                sm = [float(r["frac_robust_lo"]) for s in SNRS for r in certsem
                      if (r["model"], r["dataset"], r["spec"], int(r["snr"]))
                      == (m, ds, spec, s)]
                ax.plot(SNRS, ex, ":", lw=0.9, color=C_MODEL[m], alpha=0.55,
                        zorder=2)
                ax.plot(SNRS, sm, "-", marker=MK_MODEL[m], ms=3.4, lw=1.3,
                        color=C_MODEL[m], mfc="white", mew=1.0, zorder=3)
            ax.axhline(0.9, color="0.75", lw=0.8, ls=":", zorder=1)
            ax.set_xlim(62, 18)
            ax.set_ylim(-0.02, 1.02)
            if i == 0:
                ax.set_title(ds.upper(), fontsize=9)
            if i == 1:
                ax.set_xlabel("SNR (dB)")
            if j == 0:
                ax.set_ylabel(f"{FLOOR_NAME[spec]}\n"
                              "certified $\\tau$-robust fraction $G$",
                              fontsize=8.5)
    h = _model_handles() + [
        plt.Line2D([], [], color="0.35", lw=1.3, ls="-", label="semantic"),
        plt.Line2D([], [], color="0.35", lw=0.9, ls=":", label="exact match")]
    fig.legend(handles=h, ncol=6, frameon=False, fontsize=7,
               loc="lower center", bbox_to_anchor=(0.5, 0.955))
    fig.savefig(OUT / "fig_g_sem.pdf")
    fig.savefig(OUT / "_fig_g_sem.png")
    plt.close(fig)


MODEL_LETTER = {"quartznet": "Q", "deepspeech2": "D", "whisper": "W",
                "whisper_atc": "F"}
DS_LETTER = {"atcosim": "A", "atco2": "O"}


def emit_tex_stt(cert, wer):
    """numbers_stt.tex (WER macros) + table_models.tex (the model-comparison
    table body: WER and G at 60/20 dB per model x corpus)."""
    L = ["% AUTO-GENERATED by make_stt_paper_figs.py -- do not edit"]
    w = {(r["model"], r["dataset"]): float(r["wer"]) for r in wer}
    for (m, ds), v in sorted(w.items()):
        L.append(f"\\newcommand{{\\Wer{MODEL_LETTER[m]}{DS_LETTER[ds]}}}"
                 f"{{{v:.3f}}}")
    for r in cert:                        # G at 20 dB (the worst-channel anchor)
        if r["snr"] == "20":
            stag = "" if r["spec"] == "msg" else "c"
            tag = MODEL_LETTER[r["model"]] + DS_LETTER[r["dataset"]]
            L.append(f"\\newcommand{{\\AtcG{stag}{tag}twenty}}"
                     f"{{{float(r['frac_robust_lo']):.3f}}}")
    (OUT / "numbers_stt.tex").write_text("\n".join(L) + "\n")

    def g(m, ds, spec, snr):
        for r in cert:
            if (r["model"], r["dataset"], r["spec"], r["snr"]) == \
                    (m, ds, spec, str(snr)):
                return f"{float(r['frac_robust_lo']):.2f}"
        return "--"
    T = ["\\begin{tabular}{llccccc}", "\\toprule",
         "model & corpus & WER & $G_{60}$ msg & $G_{60}$ cs & "
         "$G_{20}$ msg & $G_{20}$ cs \\\\", "\\midrule"]
    for m in C_MODEL:
        for ds in DS_LETTER:
            if REV and (m, ds) not in w:      # rerun still in progress
                continue
            T.append(f"{MODEL_NAME[m]} & {ds.upper()} & {w[(m, ds)]:.2f} & "
                     f"{g(m, ds, 'msg', 60)} & {g(m, ds, 'cs', 60)} & "
                     f"{g(m, ds, 'msg', 20)} & {g(m, ds, 'cs', 20)} \\\\")
    T += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "table_models.tex").write_text("\n".join(T) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if REV and not (HERE / "cert_summary.csv").exists():
        print("  (revision mode: no cert_summary.csv -- two-level G figures "
              "and \\AtcG macros skipped; WER macros only)")
        emit_tex_stt([], load("wer_summary.csv"))
        print("wrote numbers_stt.tex + table_models.tex ->", OUT)
        return
    cert, wer = load("cert_summary.csv"), load("wer_summary.csv")
    fig_wer_vs_g(cert, wer)
    fig_g_curves(cert)
    fig_g_per_spec(cert)
    fig_g_grid(cert)
    try:
        fig_g_sem(load("cert_summary_sem.csv"))
    except FileNotFoundError:
        print("  (no cert_summary_sem.csv yet -- run analyze_atc_sem.py)")
    emit_tex_stt(cert, wer)
    print("wrote figs + numbers_stt.tex + table_models.tex ->", OUT)


if __name__ == "__main__":
    main()
