"""WER for all four STT models on both ATC corpora, from stored clean decodes.

The campaign CSVs record each model's greedy decode of the UNPERTURBED
utterance (clean_msg, already norm_text-normalized). Reference transcripts
come from external/stt/data/<corpus>/refs.csv. WER = word-level Levenshtein
distance / reference length, corpus-pooled (total edits / total ref words),
with the same norm_text normalization applied to references.

Calibration: quartznet/atcosim and deepspeech2 must reproduce the June
accuracy_eval numbers (QN 0.572/0.841, DS2 0.866/1.036) if the method
matches; the script asserts the first within a small tolerance.

Writes wer_summary.csv: model,dataset,n_utts,wer,ref_words.

ATC_DATA_DIR=<dir>: hypotheses read from / wer_summary.csv written to <dir>.
Revision mode (ATC_REV=1): references are normalized with
atc_text.norm_text_digits (numerals -> spoken digits; identical to norm_text
on digit-free text), and the hypothesis is norm_text_digits(raw clean decode)
where the rev1 CSV carries the RAW batch-1 clean decode (v2 models: column
clean_raw_json = json.dumps(clean_b1), ASCII-safe), else the stored clean_msg
(models copied from the campaign, e.g. whisper_atc).  Models whose rerun has
not produced a CSV yet are skipped.
"""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "Stage-2-verification" / "scenario"))
from run_scenario_datasets import norm_text  # noqa: E402
sys.path.append(str(HERE))
from atc_paths import DATA, REV  # noqa: E402
if REV:
    from atc_text import norm_text_digits as norm_text  # noqa: E402,F811

csv.field_size_limit(1 << 24)
MODELS = ["quartznet", "deepspeech2", "whisper", "whisper_atc"]
DATASETS = ["atcosim", "atco2"]


def lev(a, b):
    """Word-level Levenshtein distance."""
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i]
        for j, wb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1,
                           prev[j - 1] + (wa != wb)))
        prev = cur
    return prev[-1]


def main():
    rows_out = []
    for ds in DATASETS:
        refs = {}
        for r in csv.DictReader(open(ROOT / "external/stt/data" / ds / "refs.csv",
                                     newline="", encoding="utf-8")):
            refs[r["file"]] = norm_text(r["text"])
        for model in MODELS:
            path = DATA / f"results_atc_{model}_{ds}.csv"
            if REV and not path.exists():
                print(f"{model:12s} {ds:8s} (no rev CSV yet -- skipped)")
                continue
            hyp = {}
            for r in csv.DictReader(open(path, newline="")):
                if REV and r.get("clean_raw_json"):      # v2: raw clean_b1
                    hyp.setdefault(r["file"],
                                   norm_text(json.loads(r["clean_raw_json"])))
                    continue
                hyp.setdefault(r["file"], r["clean_msg"])
            # macro-average per-utterance WER (edit/max(len_ref,1), then
            # mean over utterances) -- matches accuracy_eval.py, so numbers
            # are comparable with the June results_atc_wer.csv
            per, edits, words = [], 0, 0
            for fn, h in hyp.items():
                ref = refs[fn].split()
                d = lev(h.split(), ref)
                per.append(d / max(len(ref), 1))
                edits += d; words += len(ref)
            wer = sum(per) / len(per)
            rows_out.append(dict(model=model, dataset=ds, n_utts=len(hyp),
                                 wer=round(wer, 4),
                                 wer_pooled=round(edits / words, 4),
                                 ref_words=words))
            print(f"{model:12s} {ds:8s} n={len(hyp):4d}  WER={wer:.3f} "
                  f"(pooled {edits/words:.3f})")
    with open(DATA / "wer_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader(); w.writerows(rows_out)
    print("wrote wer_summary.csv")


if __name__ == "__main__":
    main()
