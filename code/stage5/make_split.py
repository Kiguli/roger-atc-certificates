"""Certificate/ablation split for the ATC corpora (ATCOSIM, ATCO2).

Each corpus is split ONCE, deterministically, into
  certificate  -- utterances whose perturbation outcomes build the scenario
                  certificates (joint single-level and two-level),
  ablation     -- held-out utterances reserved for the ablation studies and
                  the out-of-sample validation (G <= R_test diagonal check).

The split is a fixed seeded shuffle of the sorted wav basenames -- the same
fixed-shuffle convention as the LibriSpeech certificate/validation split --
so it is independent of any speaker/session ordering in the directories and
of all perturbation outcomes. Certificate sizes follow the M=1000 convention
used by the other benchmarks (ATCO2 only has 857 utterances, so it gets a
600/257 split at roughly the same ratio).

Writes split_manifest.csv: dataset,file,idx_sorted,idx_shuffled,role
"""
import csv
import glob
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "external/stt/data"

SEED = 20260709                      # fixed; recorded here and in the README
M_CERT = {"atcosim": 1000, "atco2": 600}


def main():
    rows = []
    for ds in ("atcosim", "atco2"):
        wavs = sorted(Path(p).name for p in glob.glob(str(DATA / ds / "*.wav")))
        rng = np.random.default_rng([SEED, zlib.crc32(ds.encode())])
        perm = rng.permutation(len(wavs))
        mc = M_CERT[ds]
        for shuf_pos, sorted_idx in enumerate(perm):
            rows.append(dict(dataset=ds, file=wavs[sorted_idx],
                             idx_sorted=int(sorted_idx), idx_shuffled=shuf_pos,
                             role="certificate" if shuf_pos < mc else "ablation"))
        n_cert = sum(1 for r in rows if r["dataset"] == ds and r["role"] == "certificate")
        n_abl = sum(1 for r in rows if r["dataset"] == ds and r["role"] == "ablation")
        print(f"{ds}: {len(wavs)} wavs -> {n_cert} certificate + {n_abl} ablation")
    rows.sort(key=lambda r: (r["dataset"], r["idx_shuffled"]))
    out = HERE / "split_manifest.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "file", "idx_sorted",
                                           "idx_shuffled", "role"])
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
