# Roger? Probabilistic Certificates for Safety-Critical ATC Speech Recognition

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24513539.svg)](https://doi.org/10.5281/zenodo.24513539)

Code, data and proofs for the paper

> Ben Wooding and Taylor T. Johnson. **Roger? Probabilistic Certificates for
> Safety-Critical ATC Speech Recognition.** *Transactions on Machine Learning
> Research*, 2026. [OpenReview](https://openreview.net/forum?id=9Pj3FcSfgz)

Every number and figure in the paper is generated from the data in this
repository by the scripts in this repository. No number is typed by hand. One
command regenerates all of them and checks each one against the published
paper:

```bash
pip install -r requirements.txt
python reproduce.py
```

It needs no audio and no GPU. The last line of its output is the verdict, for
`39/39 files reproduced exactly.` (about 25 minutes on one machine; most of it is the per-message Clopper–Pearson table and the semantic audit)

---

## What `reproduce.py` checks

The paper's LaTeX source reads its numbers from macro files and its plots from
PDF figures. Copies of exactly those files, taken from the camera-ready source,
are in [`expected/`](expected):

| file in the paper | made by | from |
|---|---|---|
| `figures/numbers_atc.tex` | `make_atc_split_figs.py` (after `analyze_atc.py`) | original campaign, per-draw CSVs |
| `figures/numbers_heldout.tex` | `emit_heldout_macros.py` | held-out campaign CSVs |
| `figures_rev/numbers_pi.tex` | `analyze_atc_perinput.py` | revision campaign, per-draw CSVs |
| `figures_rev/numbers_semaudit.tex` | `analyze_atc_semaudit.py` → `scan_semaudit_witnesses.py` → `audit_embedders.py` → `emit_semaudit_macros.py` | stored transcripts |
| `figures_rev/numbers_stt.tex`, `table_models.tex` | `make_stt_paper_figs.py` | `wer_summary.csv` |
| `figures_rev/numbers_revision.tex` | `emit_revision_macros.py` (after `crosspass_stats.py`, `analyze_zeronoise.py`, `quick_qn_impact.py`) | zero-noise control, cross-pass, call-sign and traffic analyses |
| `figures_rev/fig_pi_*.pdf` | `make_perinput_figs.py` | the summaries above |

`reproduce.py`:

1. builds `build/` with the directory layout the experiments were run in, copies
   the scripts from `code/` into it **unchanged**, and decompresses `data/` next
   to them;
2. deletes every file a step is about to write, so a step that writes nothing
   cannot pass;
3. runs the steps in the order the paper used;
4. compares every file it produced with the camera-ready copy in `expected/`,
   and every intermediate file (certificate tables, summaries, audit results)
   with the released copy in `data/`.

Text files must match byte for byte (line endings aside). Figures must match
byte for byte apart from the PDF creation date; if they do not, and PyMuPDF is
installed, they are rendered at 200 dpi and must match pixel for pixel.

Options: `--from-raw` also rebuilds the revision per-draw CSVs from the raw
decode dumps (`convert_v2.py`); `--only a,b` runs selected steps; `--keep`
re-checks an existing `build/` without rerunning anything.

---

## Repository layout

```
reproduce.py          regenerate and check everything (start here)
code/stage5/          every analysis, campaign and figure script, verbatim
code/scenario/        shared threat model: band-limited Gaussian noise at a
                      target SNR, normalisation, call-sign readout, seeding
code/radiostar/       QuartzNet15x5 / DeepSpeech2 wrappers used by the campaign
code/campaign_commands/  the exact command files and supervisors that ran the
                      GPU campaigns (local paths replaced by placeholders)
data/campaign/        original campaign: per-draw outcome vectors for every
                      (model, dataset, message, SNR) cell, the semantic pass,
                      held-out sets, zero-noise control, all summaries
data/rev1/            revision campaign (the numbers in the final paper):
                      converted per-draw CSVs, audit outputs, embedding caches
data/external/        OpenFlights airline list and the airline designators
                      derived from ATCO2 (used by the call-sign grammar)
expected/             the macro files and figures the camera-ready reads
lean/                 Lean 4 / Mathlib proofs of the paper's formal results
```

Files ending in `.csv.gz` / `.json.gz` are plain gzip; `reproduce.py`
decompresses them. `*.jsonl.gz` files are read compressed by the scripts.

### The data

Each row of a `results_atc_<model>_<dataset>.csv` is one (message, SNR) cell. It
stores the count of violating draws for every specification and the complete
per-draw outcome vector (`bits_msg`, `bits_cs`, ...; N = 1000 bits), plus the
clean decode the certificate is defined against. Every certificate in the paper
is one Clopper–Pearson evaluation on those counts:

```
CP+(k, N, beta) = BetaInv(1 - beta; k + 1, N - k)
certified      <=>  CP+(k, N, beta) <= tau      # k <= 20 at N=1000, beta=1e-6, tau=0.05
```

`texts_sem_*.jsonl.gz` hold the distinct perturbed transcripts and their
multiplicities, which the semantic-equivalence audit embeds.
`results_v2_*.jsonl.gz` are the raw decode dumps of the revision reruns, from
which `convert_v2.py` builds `data/rev1/`.

---

## Rerunning the experiments themselves (GPU)

`reproduce.py` starts from the stored decodes. Producing those decodes needs the
audio and a GPU:

* **ATCOSIM** and **ATCO2** (1 h test set) are public research corpora; obtain
  them from their distributors and place the 16 kHz WAVs and `refs.csv` under
  `build/RadioStar/external/stt/data/<corpus>/`.
* **UWB-ATCC** (CC BY-NC-SA 4.0) is fetched by `code/stage5/fetch_uwb.py`.
* The six **LiveATC** held-out sets are not redistributable.
  `data/campaign/heldout_manifest.csv.gz` gives, for every certified clip, the
  SHA-256 of the exact 16 kHz PCM-16 clip, its public source recording and its
  cut offsets, so the identical audio can be rebuilt. Decode the source, mix to
  mono, resample to 16 kHz with `scipy.signal.resample_poly`, peak-normalise
  over the **whole** recording, slice `[start_sample_16k:end_sample_16k]`,
  write PCM-16 and check the hash.
* Model weights (QuartzNet15x5, DeepSpeech2, Whisper-base, Whisper-ATC) are
  downloaded at run time under their own licences.

The command files in `code/campaign_commands/` list the exact invocations
(SNR grids, N, batch sizes, flags). The main scripts are `run_atc_split.py` and
`run_atc_sem.py` (main campaign), `run_atc_v2.py` (revision reruns),
`run_heldout.py` (held-out sets), `run_zeronoise.py` (zero-noise control) and
`probe_digits.py`.

**Determinism.** Noise draws are seeded per (model, dataset, message, SNR) by
CRC32 of the identifying tuple. Given the same batch schedule (the GPU random
stream is drawn batch by batch), the noise behind any certificate can be
regenerated in isolation. Decoding is greedy, but reduced-precision GPU
inference is not bitwise reproducible across passes, hardware or library
versions. The paper measures this residual in its zero-noise control and
reproducibility appendices. A fresh GPU rerun will therefore agree with the
paper statistically, not bit for bit. That is why this repository releases the
per-draw outcomes, from which every number in the paper is reproduced exactly.
The QuartzNet revision reruns were made with TF32 disabled; the command files
record this.

A few inputs are not redistributable, so the steps that read them are not
rerun by `reproduce.py`. Their outputs are in `data/` and are used as inputs:

| step | needs | released output |
|---|---|---|
| `compute_wer.py` | ATCOSIM / ATCO2 reference transcripts | `rev1/wer_summary.csv` |
| `validate_callsign.py` | ATCO2 XML call-sign annotations | `callsign_validation.{csv,json}` |
| `analyze_traffic_context.py` | ATCO2 `.info` surveillance lists | `traffic_context.{csv,json}` |
| `estimate_inherent_snr.py` | ATCOSIM / ATCO2 audio | `inherent_snr.{csv,json}` |
| `run_zeronoise.py`, `probe_digits.py` | audio + GPU | `zeronoise_*.csv`, `probe_digits_*.csv` |

With the corpora in place these scripts run unchanged.

---

## Proofs

`lean/Roger.lean` formalises the paper's results in Lean 4 against Mathlib
`v4.29.0`. Every declaration is `sorry`-free; two standard facts about the
binomial tail are stated as labelled axioms. See [`lean/README.md`](lean/README.md).

```bash
cd lean && lake exe cache get && lake build
```

---

## Environment used for the paper

One NVIDIA GeForce RTX 5090, Windows 11, Python 3.10.11, PyTorch 2.11.0 +
CUDA 12.8, transformers 5.12.1, sentence-transformers 5.6.0, NumPy 1.26.4,
SciPy 1.15.3, matplotlib 3.10.9. The main campaign took about 47 GPU-hours and
the held-out campaign about 38.

## Licence

Code and proofs: MIT ([`LICENSE`](LICENSE)). Data in `data/` and `expected/`:
CC BY 4.0 ([`LICENSE-DATA.md`](LICENSE-DATA.md)), except
`data/external/airlines.dat` (OpenFlights, ODbL 1.0). No third-party audio is
included.

## Citation

```bibtex
@article{wooding2026roger,
  title   = {Roger? Probabilistic Certificates for Safety-Critical {ATC} Speech Recognition},
  author  = {Wooding, Ben and Johnson, Taylor T.},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=9Pj3FcSfgz}
}

@software{wooding2026roger_code,
  title   = {Code, data and proofs for ``{Roger? Probabilistic Certificates for Safety-Critical ATC Speech Recognition}''},
  author  = {Wooding, Ben and Johnson, Taylor T.},
  year    = {2026},
  doi     = {10.5281/zenodo.24513539},
  url     = {https://github.com/Kiguli/roger-atc-certificates}
}
```

The DOI above is Zenodo's concept DOI, which always resolves to the latest
release; v1.0.0 alone is [10.5281/zenodo.24513540](https://doi.org/10.5281/zenodo.24513540).
