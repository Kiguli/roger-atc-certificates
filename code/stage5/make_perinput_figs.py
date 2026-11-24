"""Figures for the PER-INPUT (Clopper-Pearson) ATC manuscript.

Reads Stage-5-atc analysis outputs (perinput_summary.csv, snr_budget.csv,
semaudit_summary.csv when present) and writes publication figures into the
Speech-to-Text-Verification repo.  Style matches make_stt_paper_figs.py:
Okabe-Ito hues fixed per model, marker shape as secondary encoding, serif 9pt.

Presentation conventions (2026-08 revision):
  * x-axis always runs 60 dB (left, clean) -> 20 dB (right, noisy);
  * the per-message bound figures show *robustness* 1 - CP+ (1 at the top),
    with the certified threshold tau=0.05 drawn at 1 - tau = 0.95;
  * every legend is placed ABOVE its axes (never overlapping the curves);
  * the general-purpose model is labelled "Whisper" (was "Whisper-base").

Figures:
  fig_pi_bands.pdf        robustness bands (median + IQR) vs SNR, exact-message
                          spec, cols = corpus                       [MAIN]
  fig_pi_bands_sem.pdf    same, semantic spec chi=0.95              [APPENDIX]
  fig_pi_certfrac.pdf     fraction of messages with CP+ <= tau,
                          rows = {msg, cs}, cols = corpus           [MAIN]
  fig_pi_certfrac_sem.pdf same for sem90/sem95 with same-pass exact [APPENDIX]
  fig_pi_budget.pdf       per-message SNR-budget survival curves    [MAIN]
  fig_pi_semaudit.pdf     cosine-oracle audit vs SNR                [APPENDIX]
  fig_pi_wer.pdf          WER vs certified fraction scatter         [APPENDIX]

ATC_DATA_DIR=<dir>: inputs from <dir>, figures to <dir>/figures (an explicit
[outdir] argument still wins).  Revision mode (ATC_REV=1) adds
  fig_pi_certfrac_id.pdf  certified fraction, rows = {cs, csid}  [REVISION]

All also emit _*.png previews.  Run:  python make_perinput_figs.py [outdir]
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

from atc_paths import DATA, FIGS, REV                 # noqa: E402

HERE = DATA                    # inputs: Stage-5-atc/ unless ATC_DATA_DIR
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else FIGS

C_MODEL = {"quartznet": "#0072B2", "deepspeech2": "#D55E00",
           "whisper": "#009E73", "whisper_atc": "#E69F00"}
MODEL_NAME = {"quartznet": "QuartzNet", "deepspeech2": "DeepSpeech2",
              "whisper": "Whisper", "whisper_atc": "Whisper-ATC"}
MK_MODEL = {"quartznet": "o", "deepspeech2": "s", "whisper": "^",
            "whisper_atc": "D"}
DS_NAME = {"atcosim": "ATCOSIM", "atco2": "ATCO2"}
SNRS = [60, 50, 45, 40, 35, 30, 25, 20]
TAU = 0.05
CEIL = 0.9864          # 1 - CP+(0,N,beta): best achievable robustness at N,beta

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight",
                     # trim the border the PDF is cropped to; the default 0.1in
                     # on every side is visible whitespace once scaled in LaTeX
                     "savefig.pad_inches": 0.01,
                     "axes.labelpad": 2.0, "axes.titlepad": 3.0,
                     "xtick.major.pad": 2.0, "ytick.major.pad": 2.0})


def load(name):
    p = HERE / name
    if not p.exists():
        return None
    return list(csv.DictReader(open(p, newline="")))


def _model_handles():
    return [plt.Line2D([], [], color=C_MODEL[m], marker=MK_MODEL[m], ms=5,
                       mfc="white", mew=1.0, lw=1.3, label=MODEL_NAME[m])
            for m in C_MODEL]


def _snr_axis(ax):
    """60 dB left -> 20 dB right.  Idempotent (absolute limits, not a toggle),
    so it is safe under sharex where invert_xaxis() would cancel itself out."""
    ax.set_xlim(62, 18)
    ax.set_xticks([20, 30, 40, 50, 60])


def _legend_above(fig, handles, ncol, top=0.97):
    """Place a single shared legend above the axes (never over the curves).

    The legend sits as close to the axes as it can without touching them: the
    anchor is just above the axes rectangle and the reserved strip (1-top) is
    only as deep as the legend row itself, so no blank band is left behind."""
    fig.legend(handles=handles, ncol=ncol, frameon=False, fontsize=7.5,
               loc="lower center", bbox_to_anchor=(0.5, 0.985),
               columnspacing=1.1, handlelength=1.5, handletextpad=0.5,
               borderaxespad=0.0, borderpad=0.0)
    fig.tight_layout(rect=(0, 0, 1, top), pad=0.25, w_pad=0.6, h_pad=0.4)


def _save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"_{name}.png")
    plt.close(fig)


def cellmap(rows):
    return {(r["model"], r["dataset"], r["spec"], int(r["snr"])): r
            for r in rows}


SPEC_TITLE = {"msg": "exact-message spec",
              "cs": "call-sign spec",
              "sem90": "semantic spec, $\\chi_{\\rm sem}=0.90$",
              "sem95": "semantic spec, $\\chi_{\\rm sem}=0.95$",
              "csid": "call-sign identity spec"}


def _bands_grid(cm, spec, fname):
    """Median + IQR band of the per-message *robustness* 1 - CP+, one row of
    corpus panels.  Robustness is 1 at the top; the certified threshold
    tau=0.05 is drawn at 1 - tau = 0.95 and the k=0 ceiling at CEIL."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.35), sharex=True,
                             sharey=True)
    for j, ds in enumerate(DS_NAME):
        ax = axes[j]
        for m in C_MODEL:
            xs, med, lo, hi = [], [], [], []
            for snr in SNRS:
                r = cm.get((m, ds, spec, snr))
                if not r:
                    continue
                xs.append(snr)
                med.append(1.0 - float(r["q50"]))
                lo.append(1.0 - float(r["q75"]))   # lower robustness edge
                hi.append(1.0 - float(r["q25"]))   # upper robustness edge
            if not xs:
                continue
            ax.plot(xs, med, color=C_MODEL[m], marker=MK_MODEL[m],
                    ms=3.5, mfc="white", mew=0.9, lw=1.3, zorder=3)
            ax.fill_between(xs, lo, hi, color=C_MODEL[m], alpha=0.15,
                            lw=0, zorder=1)
        ax.axhline(1 - TAU, color="0.35", lw=0.9, ls=":")
        ax.axhline(CEIL, color="0.7", lw=0.6, ls=(0, (1, 2)))
        ax.set_ylim(-0.03, 1.03)
        _snr_axis(ax)                          # 60 dB left -> 20 dB right
        ax.set_title(DS_NAME[ds], fontsize=9)
        ax.set_xlabel("SNR (dB)")
    axes[0].text(0.02, 1 - TAU, r"certified: $1-\tau=0.95$", fontsize=7,
                 color="0.35", va="top", ha="left",
                 transform=axes[0].get_yaxis_transform())
    axes[0].text(0.02, CEIL, r"$k{=}0$ ceiling $0.986$", fontsize=6.5,
                 color="0.55", va="bottom", ha="left",
                 transform=axes[0].get_yaxis_transform())
    axes[0].set_ylabel(f"certified robustness $1-\\mathrm{{CP}}^+$\n"
                       f"({SPEC_TITLE[spec]})", fontsize=8)
    _legend_above(fig, _model_handles(), ncol=4, top=0.97)
    _save(fig, fname)


def _certfrac_panel(ax, cm, ds, spec, ls="-", faint=False):
    for m in C_MODEL:
        xs, ys = [], []
        for snr in SNRS:
            r = cm.get((m, ds, spec, snr))
            if r:
                xs.append(snr)
                ys.append(float(r["certified_frac"]))
        if not xs:
            continue
        if faint:
            ax.plot(xs, ys, color=C_MODEL[m], lw=0.9, ls=":", alpha=0.55,
                    zorder=1)
        else:
            ax.plot(xs, ys, color=C_MODEL[m], marker=MK_MODEL[m], ms=3.5,
                    mfc="white", mew=0.9, lw=1.3, ls=ls, zorder=3)


def _certfrac_grid(cm, specs, fname, extra_faint=None):
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.0), sharex=True,
                             sharey=True)
    for i, spec in enumerate(specs):
        for j, ds in enumerate(DS_NAME):
            ax = axes[i][j]
            if extra_faint:
                _certfrac_panel(ax, cm, ds, extra_faint, faint=True)
            _certfrac_panel(ax, cm, ds, spec)
            ax.set_ylim(-0.03, 1.03)
            _snr_axis(ax)                      # 60 dB left -> 20 dB right
            if i == 0:
                ax.set_title(DS_NAME[ds], fontsize=9)
            if i == 1:
                ax.set_xlabel("SNR (dB)")
            if j == 0:
                ax.set_ylabel("fraction of messages with\n"
                              f"CP$^+ \\leq \\tau$  ({SPEC_TITLE[spec]})",
                              fontsize=8)
    h = _model_handles()
    if extra_faint:
        h.append(plt.Line2D([], [], color="0.4", lw=0.9, ls=":",
                            label="exact match (same pass)"))
    _legend_above(fig, h, ncol=5 if extra_faint else 4, top=0.97)
    _save(fig, fname)


def fig_pi_budget(budget_rows):
    """Survival curve: fraction of messages whose SNR budget reaches s."""
    per = defaultdict(list)
    for r in budget_rows:
        per[(r["model"], r["dataset"], r["spec"])].append(
            int(r["budget_snr"]))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.35), sharey=True)
    for ax, ds in zip(axes, DS_NAME):
        for m in C_MODEL:
            for spec, ls in (("msg", "-"), ("cs", "--")):
                bs = per.get((m, ds, spec))
                if not bs:
                    continue
                n = len(bs)
                xs = SNRS
                ys = [sum(1 for b in bs if b != 999 and b <= s) / n
                      for s in xs]
                ax.plot(xs, ys, color=C_MODEL[m], lw=1.3, ls=ls,
                        marker=MK_MODEL[m] if spec == "msg" else None,
                        ms=3.0, mfc="white", mew=0.8)
        ax.set_ylim(-0.03, 1.03)
        _snr_axis(ax)                          # 60 dB left -> 20 dB right
        ax.set_title(DS_NAME[ds], fontsize=9)
        # same axis, same label as every other SNR figure: the "budget" reading
        # lives in the y-label and the caption, not in a second name for x
        ax.set_xlabel("SNR (dB)")
    axes[0].set_ylabel("fraction of messages\ncertified down to this SNR",
                       fontsize=8)
    h = _model_handles() + [
        plt.Line2D([], [], color="0.3", lw=1.2, label="exact-message spec"),
        plt.Line2D([], [], color="0.3", lw=1.2, ls="--",
                   label="call-sign spec")]
    _legend_above(fig, h, ncol=6, top=0.97)
    _save(fig, "fig_pi_budget")


def fig_pi_semaudit(srows):
    """Oracle audit: of the draws that flip the call-sign, what fraction does
    each cosine floor forgive?  Exactly measured (no human audit)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.35), sharey=True)
    per = {(r["model"], r["dataset"], int(r["snr"])): r for r in srows}
    for ax, ds in zip(axes, DS_NAME):
        for m in C_MODEL:
            for key, ls in (("cond_pass95", "-"), ("cond_pass90", "--")):
                xs, ys = [], []
                for snr in SNRS:
                    r = per.get((m, ds, snr))
                    if r and r[key] not in ("", "None", None):
                        xs.append(snr)
                        ys.append(float(r[key]))
                if not xs:
                    continue
                ax.plot(xs, ys, color=C_MODEL[m], lw=1.3, ls=ls,
                        marker=MK_MODEL[m] if ls == "-" else None,
                        ms=3.0, mfc="white", mew=0.8)
        ax.set_ylim(-0.03, 1.03)
        _snr_axis(ax)                          # 60 dB left -> 20 dB right
        ax.set_title(DS_NAME[ds], fontsize=9)
        ax.set_xlabel("SNR (dB)")
    axes[0].set_ylabel("share of call-sign flips\nwith cosine $\\geq$ floor",
                       fontsize=8)
    h = _model_handles() + [
        plt.Line2D([], [], color="0.3", lw=1.2, label="floor 0.95"),
        plt.Line2D([], [], color="0.3", lw=1.2, ls="--", label="floor 0.90")]
    _legend_above(fig, h, ncol=6, top=0.97)
    _save(fig, "fig_pi_semaudit")


def fig_pi_wer(cm, wer_rows):
    """WER vs certified-message fraction (exact spec): does accuracy predict
    per-message certifiability?"""
    wer = {(r["model"], r["dataset"]): float(r["wer"]) for r in wer_rows}
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for (m, ds), w in wer.items():
        for snr, mfc in ((60, C_MODEL[m]), (20, "white")):
            r = cm.get((m, ds, "msg", snr))
            if not r:
                continue
            ax.scatter([w], [float(r["certified_frac"])], s=28,
                       marker=MK_MODEL[m], facecolor=mfc,
                       edgecolor=C_MODEL[m], linewidth=1.1,
                       zorder=3)
    ax.set_xlabel("clean word error rate")
    ax.set_ylabel("fraction of messages with CP$^+ \\leq \\tau$\n"
                  "(exact-message spec)", fontsize=8)
    ax.set_ylim(-0.03, 1.03)
    h = _model_handles() + [
        plt.Line2D([], [], color="0.3", marker="o", ls="", ms=5,
                   label="60 dB (filled)"),
        plt.Line2D([], [], color="0.3", marker="o", ls="", ms=5,
                   mfc="white", label="20 dB (open)")]
    _legend_above(fig, h, ncol=3, top=0.94)
    _save(fig, "fig_pi_wer")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cm = cellmap(load("perinput_summary.csv"))
    _bands_grid(cm, "msg", "fig_pi_bands")            # MAIN: exact-message
    _bands_grid(cm, "sem95", "fig_pi_bands_sem")      # APPENDIX: semantic 0.95
    _certfrac_grid(cm, ("msg", "cs"), "fig_pi_certfrac")
    _certfrac_grid(cm, ("sem90", "sem95"), "fig_pi_certfrac_sem",
                   extra_faint="exact_repro")
    if REV:                                       # REVISION: cs vs csid
        _certfrac_grid(cm, ("cs", "csid"), "fig_pi_certfrac_id")
    fig_pi_budget(load("snr_budget.csv"))
    wr = load("wer_summary.csv")
    if wr:
        fig_pi_wer(cm, wr)
    sa = load("semaudit_summary.csv")
    if sa:
        fig_pi_semaudit(sa)
    else:
        print("  (semaudit_summary.csv absent -- fig_pi_semaudit skipped)")
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
