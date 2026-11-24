"""Figures + headline numbers for the ATC certificate/ablation split study.

Reads Stage-5-atc/{cert_summary,validation_oos,ablation_*}.csv (produced by
analyze_atc.py) and writes -- canonically into Stage-5-atc/figures/, mirrored
into Stage-4-final/figures/ and Stage-4-final/slides/figures/ so the master
document and the deck pick them up unchanged:

  figures/fig_atc_validation.pdf   out-of-sample diagonal check (G vs R_test),
                                   one panel per spec, style-matched to
                                   validation/analyze_validation.py.
  figures/fig_atc_ablations.pdf    2x3 ablation collage: N-sensitivity (per
                                   model), tau grid, M-scaling,
                                   fresh-vs-June redraw agreement.
  figures/numbers_atc.{json,tex}   headline numbers for the .tex.
  figures/table_atc_cert.tex       generated certificate table.

Colors follow the paper's Okabe-Ito convention (model identity is also
encoded by marker/linestyle, never by hue alone).

Run:  python make_atc_split_figs.py
"""
import csv
import json
import os
from pathlib import Path

TAG = os.environ.get("ATC_TAG", "")   # analyse a tagged (smoke) run instead

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent      # <repo>/Stage-5-atc
ROOT = HERE.parent
ATC = HERE
FIG = HERE / "figures"                       # canonical
M4FIG = ROOT / "Stage-4-final" / "figures"   # mirror: master document
SFIG = ROOT / "Stage-4-final" / "slides" / "figures"   # mirror: deck

C_MODEL = {"quartznet": "#0072B2", "deepspeech2": "#D55E00",
           "whisper": "#009E73", "whisper_atc": "#E69F00"}
MK_DS = {"atcosim": "o", "atco2": "s"}
LS_DS = {"atcosim": "-", "atco2": "--"}
SPEC_NAME = {"msg": "full-message spec", "cs": "call-sign spec"}
MODEL_NAME = {"quartznet": "QuartzNet", "deepspeech2": "DeepSpeech2",
              "whisper": "Whisper-base", "whisper_atc": "Whisper-ATC"}

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight"})


def rows(name, **filt):
    path = ATC / name.replace(".csv", f"{TAG}.csv")
    if not path.exists():
        return []
    out = []
    for r in csv.DictReader(open(path, newline="")):
        if all(r[k] == str(v) for k, v in filt.items()):
            out.append(r)
    return out


def f(r, k):
    return float(r[k])


def seq_blues(n):
    """Single-hue light->dark ramp (sequential: SNR magnitude)."""
    return [plt.cm.Blues(0.35 + 0.6 * i / max(n - 1, 1)) for i in range(n)]


# ------------------------------------------------------ fig 1: OOS validation
def fig_validation():
    oos = rows("validation_oos.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2), sharey=True)
    for ax, spec in zip(axes, ["msg", "cs"]):
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color="0.5")
        ax.fill_between([0, 1], [0, 1], [1, 1], color="#009E73", alpha=0.06)
        for r in [r for r in oos if r["spec"] == spec]:
            c = C_MODEL[r["model"]]
            ax.errorbar(f(r, "G"), f(r, "R_test"),
                        yerr=[[max(0.0, f(r, "R_test") - f(r, "ci_lo"))],
                              [max(0.0, f(r, "ci_hi") - f(r, "R_test"))]],
                        fmt=MK_DS[r["dataset"]], ms=4, mfc=c, mec=c, ecolor=c,
                        elinewidth=0.7, capsize=1.5, alpha=0.85)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("certified lower bound $G$")
        ax.set_title(SPEC_NAME[spec], fontsize=9)
    axes[0].set_ylabel("empirical robust fraction $R_{\\rm test}$ (held out)")
    h = [plt.Line2D([], [], color=C_MODEL[m], marker="o", ls="", ms=4,
                    label=MODEL_NAME[m]) for m in C_MODEL]
    h += [plt.Line2D([], [], color="0.3", marker=MK_DS[d], ls="", ms=4,
                     mfc="none", label=d.upper()) for d in MK_DS]
    h += [plt.Line2D([], [], color="0.5", ls="--", label="$G=R_{\\rm test}$")]
    axes[1].legend(handles=h, frameon=False, fontsize=7, loc="lower right")
    for p in (FIG, M4FIG, SFIG):
        fig.savefig(p / "fig_atc_validation.pdf")
    plt.close(fig)
    n_hold = sum(int(r["holds"]) for r in oos)
    gaps = [f(r, "gap") for r in oos]
    return dict(n_cells=len(oos), n_holds=n_hold,
                mean_gap=round(float(np.mean(gaps)), 4),
                min_gap=round(float(np.min(gaps)), 4))


# ---------------------------------------------------- fig 2: ablation collage
def fig_ablations():
    fig, axes = plt.subplots(2, 4, figsize=(12.2, 5.6))
    snr_show = [60, 50, 40, 30]

    # (a-c) N-sensitivity, ATCOSIM, msg spec, one panel per model
    for ax, model in zip(axes[0], C_MODEL):
        cols = seq_blues(len(snr_show))
        for c, snr in zip(cols, snr_show):
            rr = sorted(rows("ablation_N.csv", model=model, dataset="atcosim",
                             spec="msg", snr=snr), key=lambda r: int(r["N_prime"]))
            ax.plot([int(r["N_prime"]) for r in rr], [f(r, "G") for r in rr],
                    "o-", ms=3, lw=1.2, color=c, label=f"{snr} dB")
        ax.set_xscale("log"); ax.set_ylim(-0.02, 1)
        ax.set_xlabel("perturbations per message $N'$")
        ax.set_title(f"{MODEL_NAME[model]}: $G$ vs $N'$ (ATCOSIM)", fontsize=8.5)
        if model == "quartznet":
            ax.set_ylabel("certified robust fraction $G$")
            ax.legend(frameon=False, fontsize=6.5, title="SNR", title_fontsize=6.5)

    # (d) tau grid: G (lines) vs R_test (open markers), ATCOSIM, msg, 60/45 dB
    ax = axes[1][0]
    for model in C_MODEL:
        for snr, lsty in ((60, "-"), (45, "--")):
            rr = sorted(rows("ablation_tau.csv", model=model, dataset="atcosim",
                             spec="msg", snr=snr), key=lambda r: float(r["tau"]))
            taus = [f(r, "tau") for r in rr]
            ax.plot(taus, [f(r, "G") for r in rr], lsty, lw=1.2,
                    color=C_MODEL[model])
            ax.plot(taus, [f(r, "R_test") for r in rr], "o", ms=3.5, mfc="none",
                    color=C_MODEL[model])
    ax.set_xscale("log"); ax.set_ylim(-0.02, 1)
    ax.set_xlabel("per-message risk threshold $\\tau$")
    ax.set_ylabel("certified robust fraction $G$")
    ax.set_title("$G(\\tau)$ lines vs held-out $R_{\\rm test}(\\tau)$ circles\n"
                 "(ATCOSIM; solid 60 dB, dashed 45 dB)", fontsize=8.5)

    # (e) M-scaling at 60 dB, msg spec, both corpora
    ax = axes[1][1]
    any_m = False
    for model in C_MODEL:
        for ds in MK_DS:
            rr = sorted(rows("ablation_M.csv", model=model, dataset=ds,
                             spec="msg", snr=60), key=lambda r: int(r["M_sub"]))
            if not rr:
                continue
            any_m = True
            ms = [int(r["M_sub"]) for r in rr]
            ax.plot(ms, [f(r, "G_mean") for r in rr], LS_DS[ds], lw=1.2,
                    marker=MK_DS[ds], ms=3, color=C_MODEL[model])
            ax.fill_between(ms, [f(r, "G_min") for r in rr],
                            [f(r, "G_max") for r in rr],
                            color=C_MODEL[model], alpha=0.12, lw=0)
    if any_m:
        ax.set_xscale("log")
    ax.set_ylim(-0.02, 1)
    ax.set_xlabel("certificate messages $M'$")
    ax.set_title("$G$ vs $M'$ at 60 dB (band: min-max of 25 draws;\n"
                 "solid ATCOSIM, dashed ATCO2)", fontsize=8.5)
    ax.legend(handles=[plt.Line2D([], [], color=C_MODEL[m], lw=1.2,
                                  label=MODEL_NAME[m]) for m in C_MODEL],
              frameon=False, fontsize=6.5)

    # (f) redraw agreement (fresh vs June independent noise), msg spec
    ax = axes[1][2]
    stats = {}
    for model in C_MODEL:
        rr = rows("ablation_redraw.csv", model=model)
        if not rr:
            continue
        x = np.array([int(r["k_msg_june"]) for r in rr]) / 1000.0
        y = np.array([int(r["k_msg_fresh"]) for r in rr]) / 1000.0
        idx = np.random.default_rng(0).choice(len(x), min(len(x), 2500),
                                              replace=False)
        ax.plot(x[idx], y[idx], MK_DS["atcosim"], ms=1.5, alpha=0.15,
                color=C_MODEL[model], mec="none")
        stats[model] = round(float(np.corrcoef(x, y)[0, 1]), 4)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="0.5")
    ax.set_xlabel("June run $k/N$ (independent draws)")
    ax.set_ylabel("this run $k/N$")
    ax.set_title("redraw agreement, message spec\n"
                 + ", ".join(f"{MODEL_NAME[m]} $r$={stats[m]:.3f}"
                             for m in stats), fontsize=8.5)

    # (g) beta grid at 60 dB, ATCOSIM, msg spec
    ax = axes[1][3]
    any_b = False
    for model in C_MODEL:
        rr = sorted(rows("ablation_beta.csv", model=model, dataset="atcosim",
                         spec="msg", snr=60), key=lambda r: float(r["beta"]))
        if not rr:
            continue
        any_b = True
        ax.plot([f(r, "beta") for r in rr], [f(r, "G") for r in rr], "o-",
                ms=3.5, lw=1.2, color=C_MODEL[model])
    if any_b:
        ax.set_xscale("log")
    ax.set_ylim(-0.02, 1)
    ax.set_xlabel("confidence budget $\\beta$")
    ax.set_title("$G$ vs $\\beta$ at 60 dB (ATCOSIM)", fontsize=8.5)
    fig.tight_layout()
    for p in (FIG, M4FIG, SFIG):
        fig.savefig(p / "fig_atc_ablations.pdf")
    plt.close(fig)
    return dict(redraw_pearson=stats)


MODEL_LETTER = {"quartznet": "Q", "deepspeech2": "D", "whisper": "W",
                "whisper_atc": "F"}
DS_LETTER = {"atcosim": "A", "atco2": "O"}


def emit_tex(nums, summ):
    """figures/numbers_atc.tex (macros) + figures/table_atc_cert.tex (booktabs
    body) so the .tex never hardcodes result numbers."""
    oos = rows("validation_oos.csv")
    holds_strict = sum(int(r.get("holds_strict", 0)) for r in oos)
    L = [
        "% AUTO-GENERATED by make_atc_split_figs.py -- do not edit",
        f"\\newcommand{{\\AtcCells}}{{{nums['oos']['n_cells']}}}",
        f"\\newcommand{{\\AtcHolds}}{{{nums['oos']['n_holds']}}}",
        f"\\newcommand{{\\AtcHoldsStrict}}{{{holds_strict}}}",
        f"\\newcommand{{\\AtcMeanGap}}{{{nums['oos']['mean_gap']:+.3f}}}",
        f"\\newcommand{{\\AtcMinGap}}{{{nums['oos']['min_gap']:+.3f}}}",
    ]
    for key in (k for k in summ if "/" in k):
        model, ds = key.split("/")
        tag = MODEL_LETTER[model] + DS_LETTER[ds]
        L.append(f"\\newcommand{{\\AtcMcert{tag}}}"
                 f"{{{summ[key]['M_cert']}}}")
        L.append(f"\\newcommand{{\\AtcMabl{tag}}}{{{summ[key]['M_abl']}}}")
        for spec, stag in (("msg", ""), ("cs", "c")):
            g = summ[key]["per_spec"][spec]["curve"]["60"]["frac_robust"]
            L.append(f"\\newcommand{{\\AtcG{stag}{tag}}}{{{g:.3f}}}")
    for m, r_p in nums["ablations"].get("redraw_pearson", {}).items():
        L.append(f"\\newcommand{{\\AtcRedraw{MODEL_LETTER[m]}}}{{{r_p:.3f}}}")
    for d in (FIG, M4FIG):
        (d / "numbers_atc.tex").write_text("\n".join(L) + "\n")

    # certificate table: G(msg)/G(cs) per SNR, one row per model x corpus
    cert = rows("cert_summary.csv")
    snrs = sorted({int(r["snr"]) for r in cert}, reverse=True)

    def g(model, ds, spec, snr):
        for r in cert:
            if (r["model"], r["dataset"], r["spec"], int(r["snr"])) == \
                    (model, ds, spec, snr):
                return f"{float(r['frac_robust_lo']):.2f}"[1:] \
                    if float(r["frac_robust_lo"]) < 1 else "1.0"
        return "--"
    T = ["\\begin{tabular}{ll" + "c" * len(snrs) + "}", "\\toprule",
         "model & corpus & " + " & ".join(str(s) for s in snrs) + " \\\\",
         "\\midrule"]
    for model in MODEL_NAME:
        for ds in DS_LETTER:
            if not any(r["model"] == model and r["dataset"] == ds
                       for r in cert):
                continue
            cells = " & ".join(f"{g(model, ds, 'msg', s)}/"
                               f"{g(model, ds, 'cs', s)}" for s in snrs)
            T.append(f"{MODEL_NAME[model]} & {ds.upper()} & {cells} \\\\")
    T += ["\\bottomrule", "\\end{tabular}"]
    for d in (FIG, M4FIG):
        (d / "table_atc_cert.tex").write_text("\n".join(T) + "\n")
    print("wrote numbers_atc.tex + table_atc_cert.tex (canonical + mirror)")


def main():
    for d in (FIG, M4FIG, SFIG):
        d.mkdir(parents=True, exist_ok=True)
    nums = {"oos": fig_validation(), "ablations": fig_ablations()}
    summ = json.load(open(ATC / f"atc_summary{TAG}.json"))
    nums["meta"] = summ["meta"]
    for key in (k for k in summ if "/" in k):
        nums[key] = {
            "M_cert": summ[key]["M_cert"], "M_abl": summ[key]["M_abl"],
            "G60_msg": summ[key]["per_spec"]["msg"]["curve"]["60"]["frac_robust"],
            "G60_cs": summ[key]["per_spec"]["cs"]["curve"]["60"]["frac_robust"],
        }
    for d in (FIG, M4FIG):
        with open(d / "numbers_atc.json", "w") as fh:
            json.dump(nums, fh, indent=1)
    emit_tex(nums, summ)
    print("wrote", FIG / "numbers_atc.json")
    print(json.dumps(nums["oos"], indent=1))


if __name__ == "__main__":
    main()
