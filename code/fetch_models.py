"""Download the four speech-to-text models used in the paper and check that
they are the exact weights the campaigns ran with.

    python code/fetch_models.py            # all four
    python code/fetch_models.py qn ds2     # a subset: qn, ds2, wh, whatc

No model weights are redistributed in this repository; each one comes from
its original publisher under its own licence:

  qn     QuartzNet15x5     NVIDIA NeMo stt_en_quartznet15x5      (CC BY 4.0)
         -> models/qn/{model_weights.ckpt,model_config.yaml}
  ds2    DeepSpeech2       SeanNaren/deepspeech.pytorch V3.0      (MIT)
         librispeech_pretrained_v3.ckpt, converted to a plain state_dict
         -> models/ds2_state_dict.pt
  wh     Whisper-base      openai/whisper-base                    (Apache-2.0)
  whatc  Whisper-ATC       jlvdoorn/whisper-large-v3-atco2-asr-atcosim
         (whisper-large-v3 fine-tuned on ATCO2 + ATCOSIM; see its model card)
         -> Hugging Face cache, at the pinned revision below

QuartzNet and DeepSpeech2 are checked against the SHA-256 / tensor checksums
of the files the campaign loaded. The Whisper models are pinned to the
Hugging Face commits that were in the campaign's cache, and the script prints
the revision so that run_atc_split.py's from_pretrained(...) loads the same
files (set HF_HUB_OFFLINE=1 afterwards to be sure nothing newer is fetched).
"""
import hashlib
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"          # git-ignored; reproduce.py copies it into build/
DL = MODELS / "_downloads"

QN_SHA = {
    "model_weights.ckpt": "489bcd4036716a1573391744cd0790ce613822407a106ba3209269e7b1617f2c",
    "model_config.yaml": "1b986f096f5fc08435cce51c001968c9776bfa3ba538bb230f2bd080453d0fc7",
}
DS2_URL = ("https://github.com/SeanNaren/deepspeech.pytorch/releases/download/"
           "V3.0/librispeech_pretrained_v3.ckpt")
# tensor_digest() of the ds2_state_dict.pt the campaign loaded (80 tensors)
DS2_TENSOR_SHA = "ad88607e3e078e80a2161f3617c8cdeddb9ef2e5415b5a6b231cf468636a4d6f"
WHISPER = {
    "wh": ("openai/whisper-base", "e37978b90ca9030d5170a5c07aadb050351a65bb"),
    "whatc": ("jlvdoorn/whisper-large-v3-atco2-asr-atcosim",
              "efd96bd7862709d2d6aeb55621934a9965db8ed7"),
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tensor_digest(sd) -> str:
    """Order-independent checksum of a state_dict's names, dtypes, shapes, values."""
    import torch
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous().reshape(-1)
        h.update(k.encode()); h.update(str(t.dtype).encode())
        h.update(str(tuple(sd[k].shape)).encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def fetch_qn():
    from huggingface_hub import hf_hub_download
    nemo = hf_hub_download("nvidia/stt_en_quartznet15x5", "stt_en_quartznet15x5.nemo")
    out = MODELS / "qn"
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(nemo) as t:
        for n in QN_SHA:
            m = next(x for x in t.getmembers() if x.name.endswith(n))
            with t.extractfile(m) as fi, open(out / n, "wb") as fo:
                shutil.copyfileobj(fi, fo)
    for n, want in QN_SHA.items():
        got = sha256(out / n)
        if got != want:
            sys.exit(f"qn: {n} sha256 {got} != expected {want}")
    print("qn    QuartzNet15x5: OK (SHA-256 matches the campaign's weights)")


def fetch_ds2():
    import torch
    DL.mkdir(parents=True, exist_ok=True)
    ck_p = DL / "librispeech_pretrained_v3.ckpt"
    if not ck_p.exists():
        print("ds2   downloading", DS2_URL)
        urllib.request.urlretrieve(DS2_URL, ck_p)
    ck = torch.load(ck_p, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    sd = {k.split("model.", 1)[-1]: v for k, v in sd.items()}
    got = tensor_digest(sd)
    if got != DS2_TENSOR_SHA:
        sys.exit(f"ds2: tensor checksum {got} != expected {DS2_TENSOR_SHA}")
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(sd, MODELS / "ds2_state_dict.pt")
    print("ds2   DeepSpeech2: OK (every tensor identical to the campaign's state_dict)")


def fetch_whisper(tag):
    from huggingface_hub import snapshot_download
    repo, rev = WHISPER[tag]
    snapshot_download(repo, revision=rev)
    print(f"{tag:5s} {repo} @ {rev}: OK (in the Hugging Face cache)")


def main():
    want = sys.argv[1:] or ["qn", "ds2", "wh", "whatc"]
    for t in want:
        {"qn": fetch_qn, "ds2": fetch_ds2}.get(t, lambda: fetch_whisper(t))()
    print("\nDone. Before running the GPU campaign scripts, set HF_HUB_OFFLINE=1 so "
          "transformers uses exactly these cached Whisper revisions.")


if __name__ == "__main__":
    main()
