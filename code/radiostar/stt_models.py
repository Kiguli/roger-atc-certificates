"""Plain-PyTorch inference for the two open-weight STT models (plan-STT P0).

No framework dependencies: both models are reconstructed directly from
their released weight files, which doubles as the start of the golden
reimplementation the verification pipeline needs (cf. pipeline.py for
FSDD). Greedy CTC decoding only -- no language model, by design: the
verification target is the acoustic model, and LM rescoring would mask
its raw behavior.

Models:

  DeepSpeech2 (LSTM lineage)  -- deepspeech.pytorch V3.0 release,
    librispeech_pretrained_v3.ckpt (SeanNaren), 86.6M params:
    log1p-magnitude STFT (20ms/10ms, hamming, 161 bins) -> 2x masked
    Conv2d+BN+HardTanh(0,20) -> 5x bidirectional LSTM(1024) with
    sum-merged directions and inter-layer BatchNorm -> BN+Linear(29).
    Labels: '_' (blank, index 0), apostrophe, A-Z, space.

  QuartzNet15x5 (TCN lineage) -- NVIDIA NeMo stt_en_quartznet15x5
    (NGC 1.0.0rc1), 19.0M params, fully convolutional:
    pre-emphasis 0.97 -> 64 log-mel (20ms/10ms, hann, n_fft 512,
    per-feature normalized) -> C1 (stride-2 separable conv k=33) ->
    15 residual separable blocks (5 groups, kernels 33..75ish, repeat 5)
    -> C2 (k=87, dilation 2) -> C3 (1x1, 1024) -> 1x1 conv to 29.
    Labels: space, a-z, apostrophe, blank LAST (index 28).
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.signal import get_window

_MODELS = Path(__file__).resolve().parents[2] / "external" / "stt" / "models"


def ctc_greedy(logits, labels, blank):
    """logits: (T, C) -> collapsed text."""
    ids = logits.argmax(-1).tolist()
    out, prev = [], None
    for i in ids:
        if i != prev and i != blank:
            out.append(labels[i])
        prev = i
    return "".join(out)


# ---------------------------------------------------------------------------
# DeepSpeech2 (deepspeech.pytorch V3.0)
# ---------------------------------------------------------------------------

class DeepSpeech2:
    LABELS = ["_", "'", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
              "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
              "W", "X", "Y", "Z", " "]
    BLANK = 0

    def __init__(self, sd_path=_MODELS / "ds2_state_dict.pt"):
        self.sd = torch.load(sd_path, map_location="cpu", weights_only=True)
        self.window = torch.tensor(
            get_window("hamming", 320, fftbins=True), dtype=torch.float32)

    def features(self, wav):
        """log1p |STFT|, per-utterance standardized. wav: float32 16 kHz."""
        x = torch.as_tensor(wav, dtype=torch.float32)
        spec = torch.stft(x, n_fft=320, hop_length=160, win_length=320,
                          window=self.window, center=True,
                          pad_mode="reflect", return_complex=True).abs()
        spec = torch.log1p(spec)
        return (spec - spec.mean()) / (spec.std() + 1e-10)   # (161, T)

    def _bn(self, x, prefix, eps=1e-5):
        sd = self.sd
        w, b = sd[f"{prefix}.weight"], sd[f"{prefix}.bias"]
        m, v = sd[f"{prefix}.running_mean"], sd[f"{prefix}.running_var"]
        shape = [1, -1] + [1] * (x.dim() - 2)
        return (x - m.view(shape)) / torch.sqrt(v.view(shape) + eps) \
            * w.view(shape) + b.view(shape)

    def _bilstm(self, x, prefix):
        """x: (T, D) -> (T, 1024), directions summed (BatchRNN semantics)."""
        sd = self.sd
        outs = []
        for suffix, rev in (("", False), ("_reverse", True)):
            wih = sd[f"{prefix}.weight_ih_l0{suffix}"]
            whh = sd[f"{prefix}.weight_hh_l0{suffix}"]
            bias = sd[f"{prefix}.bias_ih_l0{suffix}"] + \
                sd[f"{prefix}.bias_hh_l0{suffix}"]
            seq = torch.flip(x, [0]) if rev else x
            H = whh.shape[1]
            h = torch.zeros(H)
            c = torch.zeros(H)
            pre = seq @ wih.T + bias
            res = []
            for t in range(seq.shape[0]):
                g = pre[t] + h @ whh.T
                i, f_, gg, o = g.split(H)
                c = torch.sigmoid(f_) * c + torch.sigmoid(i) * torch.tanh(gg)
                h = torch.sigmoid(o) * torch.tanh(c)
                res.append(h)
            out = torch.stack(res)
            outs.append(torch.flip(out, [0]) if rev else out)
        return outs[0] + outs[1]

    @torch.no_grad()
    def transcribe(self, wav):
        sd = self.sd
        x = self.features(wav)[None, None]                  # (1,1,161,T)
        x = F.conv2d(x, sd["conv.seq_module.0.weight"],
                     sd["conv.seq_module.0.bias"], stride=(2, 2),
                     padding=(20, 5))
        x = torch.clamp(self._bn(x, "conv.seq_module.1"), 0, 20)
        x = F.conv2d(x, sd["conv.seq_module.3.weight"],
                     sd["conv.seq_module.3.bias"], stride=(2, 1),
                     padding=(10, 5))
        x = torch.clamp(self._bn(x, "conv.seq_module.4"), 0, 20)
        B, C, Fr, T = x.shape
        x = x.view(C * Fr, T).T                              # (T, 1312)

        for layer in range(5):
            if layer > 0:
                x = self._bn(x[None].transpose(1, 2),
                             f"rnns.{layer}.batch_norm.module"
                             ).transpose(1, 2)[0]
            x = self._bilstm(x, f"rnns.{layer}.rnn")

        x = self._bn(x[None].transpose(1, 2), "fc.0.module.0"
                     ).transpose(1, 2)[0]
        logits = x @ sd["fc.0.module.1.weight"].T            # (T, 29)
        return ctc_greedy(logits, self.LABELS, self.BLANK).lower().strip(), \
            logits


# ---------------------------------------------------------------------------
# QuartzNet15x5 (NeMo NGC checkpoint, manually reconstructed)
# ---------------------------------------------------------------------------

class QuartzNet15x5:
    def __init__(self, nemo_dir=_MODELS / "qn"):
        nemo_dir = Path(nemo_dir)
        self.sd = torch.load(nemo_dir / "model_weights.ckpt",
                             map_location="cpu", weights_only=True)
        cfg = yaml.safe_load(open(nemo_dir / "model_config.yaml"))
        enc = cfg["encoder"]
        self.blocks = (enc.get("params") or enc)["jasper"]
        self.labels = cfg["decoder"].get("params", cfg["decoder"])["vocabulary"]
        self.blank = len(self.labels)                        # 28
        self.window = self.sd["preprocessor.featurizer.window"]
        self.fb = self.sd["preprocessor.featurizer.fb"][0]   # (64, 257)

    def features(self, wav):
        """64 log-mel, per-feature normalized. wav: float32 16 kHz."""
        x = torch.as_tensor(wav, dtype=torch.float32)
        x = torch.cat([x[:1], x[1:] - 0.97 * x[:-1]])        # pre-emphasis
        spec = torch.stft(x, n_fft=512, hop_length=160, win_length=320,
                          window=self.window, center=True,
                          pad_mode="reflect", return_complex=True)
        power = spec.real ** 2 + spec.imag ** 2              # (257, T)
        mel = self.fb @ power                                # (64, T)
        mel = torch.log(mel + 2.0 ** -24)
        mean = mel.mean(dim=1, keepdim=True)
        std = mel.std(dim=1, keepdim=True) + 1e-5
        return (mel - mean) / std                            # (64, T)

    def _bn(self, x, prefix, eps=1e-3):
        # NeMo JasperBlock BatchNorm1d uses eps=1e-3
        sd = self.sd
        w, b = sd[f"{prefix}.weight"], sd[f"{prefix}.bias"]
        m, v = sd[f"{prefix}.running_mean"], sd[f"{prefix}.running_var"]
        return (x - m[:, None]) / torch.sqrt(v[:, None] + eps) \
            * w[:, None] + b[:, None]

    def _conv_unit(self, x, prefix_dw, prefix_pw, stride=1, dilation=1):
        sd = self.sd
        wd = sd[f"{prefix_dw}.conv.weight"]                  # (C,1,k) dw
        k = wd.shape[-1]
        pad = (dilation * (k - 1)) // 2
        x = F.conv1d(x, wd, stride=(stride,), padding=(pad,),
                     dilation=(dilation,), groups=wd.shape[0])
        wp = sd[f"{prefix_pw}.conv.weight"]                  # (Cout,Cin,1)
        return F.conv1d(x, wp)

    @torch.no_grad()
    def transcribe(self, wav):
        sd = self.sd
        x = self.features(wav)[None]                         # (1, 64, T)

        for bi, cfg in enumerate(self.blocks):
            base = f"encoder.encoder.{bi}"
            stride = cfg.get("stride", [1])[0]
            dil = cfg.get("dilation", [1])[0]
            repeat = cfg.get("repeat", 1)
            separable = cfg.get("separable", False)
            inp = x
            if separable:
                for r in range(repeat):
                    i0 = 5 * r
                    x = self._conv_unit(x, f"{base}.mconv.{i0}",
                                        f"{base}.mconv.{i0+1}",
                                        stride=stride if r == 0 else 1,
                                        dilation=dil)
                    x = self._bn(x[0], f"{base}.mconv.{i0+2}")[None]
                    if r < repeat - 1:
                        x = F.relu(x)
            else:
                w = sd[f"{base}.mconv.0.conv.weight"]
                pad = (dil * (w.shape[-1] - 1)) // 2
                x = F.conv1d(x, w, stride=(stride,), padding=(pad,),
                             dilation=(dil,))
                x = self._bn(x[0], f"{base}.mconv.1")[None]
            if cfg.get("residual", False):
                r = F.conv1d(inp, sd[f"{base}.res.0.0.conv.weight"])
                r = self._bn(r[0], f"{base}.res.0.1")[None]
                x = x + r
            x = F.relu(x)

        logits = F.conv1d(x, sd["decoder.decoder_layers.0.weight"],
                          sd["decoder.decoder_layers.0.bias"])[0].T  # (T,29)
        return ctc_greedy(logits, self.labels + ["_"], self.blank).strip(), \
            logits
