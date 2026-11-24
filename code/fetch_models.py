"""Download the four speech-to-text models used in the paper and check that
they are the exact weights the campaigns ran with.

    python code/fetch_models.py            # all four
    python code/fetch_models.py qn ds2     # a subset: qn, ds2, wh, whatc

No model weights are redistributed in this repository; each one comes from
its original publisher under its own licence:

  qn     QuartzNet15x5     NVIDIA NGC nemo/stt_en_quartznet15x5 1.0.0rc1 (CC BY 4.0)
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

QN_URL = ("https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_quartznet15x5/"
          "versions/1.0.0rc1/files/stt_en_quartznet15x5.nemo")
QN_SHA = {
    "model_weights.ckpt": "489bcd4036716a1573391744cd0790ce613822407a106ba3209269e7b1617f2c",
    "model_config.yaml": "1b986f096f5fc08435cce51c001968c9776bfa3ba538bb230f2bd080453d0fc7",
}
DS2_URL = ("https://github.com/SeanNaren/deepspeech.pytorch/releases/download/"
           "V3.0/librispeech_pretrained_v3.ckpt")
# SHA-256 of the release file (checked before loading it)
DS2_FILE_SHA = "762701a432e68cc9ed6f8baa74d911b7d942729e6c3740c8d78febcee8a496fd"
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


def download(url, dst: Path):
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        print("      downloading", url)
        tmp = dst.with_suffix(dst.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dst)
    return dst


def fetch_qn():
    nemo = download(QN_URL, DL / "stt_en_quartznet15x5.nemo")
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


class _Stub:
    """Inert stand-in for any non-tensor class in a checkpoint: accepts and
    stores whatever the pickle passes, runs no code from the file."""
    def __init__(self, *a, **k):
        self._args = a
    def __setstate__(self, state):
        self._state = state
    def __setitem__(self, k, v):
        pass
    def append(self, v):
        pass
    def extend(self, v):
        pass


def _stub_pickle_module():
    """A pickle module for torch.load whose Unpickler resolves only what is
    needed to rebuild tensors; every other global becomes an inert _Stub."""
    import pickle
    import types
    allow = {("collections", "OrderedDict")}
    for n in ("dict", "list", "tuple", "set", "frozenset", "int", "float",
              "str", "bool", "bytes", "bytearray", "complex", "slice"):
        allow.add(("builtins", n))

    class Unpickler(pickle.Unpickler):
        def find_class(self, mod, name):
            if (mod, name) in allow or mod == "torch._utils" or (
                    mod == "torch" and name.endswith("Storage")) or (
                    mod == "torch" and name in ("Size", "device", "dtype")):
                return super().find_class(mod, name)
            if mod == "collections" and name == "defaultdict":
                return lambda *a: {}
            return type(name, (_Stub,), {})

    m = types.ModuleType("stub_pickle")
    m.Unpickler, m.load = Unpickler, pickle.load
    m.__name__ = "stub_pickle"
    return m


def fetch_ds2():
    import torch
    ck_p = download(DS2_URL, DL / "librispeech_pretrained_v3.ckpt")
    got = sha256(ck_p)               # checked before anything is unpickled
    if got != DS2_FILE_SHA:
        sys.exit(f"ds2: download sha256 {got} != expected {DS2_FILE_SHA}")
    ck = torch.load(ck_p, map_location="cpu", weights_only=False,
                    pickle_module=_stub_pickle_module())
    sd = ck["state_dict"]
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
