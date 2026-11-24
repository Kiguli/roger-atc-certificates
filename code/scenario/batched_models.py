"""Batched forward passes for QuartzNet & DeepSpeech2 so the scenario
sampler can score thousands of perturbations of one utterance at once.

Mirrors radiostar.stt_models exactly (validated: batched logits[0] matches
the single-utterance transcribe), but keeps a leading batch dim B = number
of perturbations. All perturbations of one utterance share its length, so
they batch cleanly.
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from radiostar.stt_models import ctc_greedy   # noqa: E402

torch.set_grad_enabled(False)


# ----------------------------- QuartzNet -----------------------------

def qn_logits_batch(mdl, wavs):
    """wavs: (B, L) float tensor -> logits (B, T, 29)."""
    sd = mdl.sd; win = mdl.window
    x = wavs
    x = torch.cat([x[:, :1], x[:, 1:] - 0.97 * x[:, :-1]], dim=1)   # pre-emphasis
    spec = torch.stft(x, n_fft=512, hop_length=160, win_length=320, window=win,
                      center=True, pad_mode="reflect", return_complex=True)
    power = spec.real ** 2 + spec.imag ** 2                          # (B,257,T)
    mel = torch.einsum("fj,bjt->bft", mdl.fb, power)                 # (B,64,T)
    mel = torch.log(mel + 2.0 ** -24)
    mean = mel.mean(dim=2, keepdim=True); std = mel.std(dim=2, keepdim=True) + 1e-5
    x = (mel - mean) / std                                           # (B,64,T)
    for bi, cfg in enumerate(mdl.blocks):
        base = f"encoder.encoder.{bi}"
        stride = cfg.get("stride", [1])[0]; dil = cfg.get("dilation", [1])[0]
        repeat = cfg.get("repeat", 1); sep = cfg.get("separable", False)
        inp = x
        if sep:
            for r in range(repeat):
                i0 = 5 * r
                wd = sd[f"{base}.mconv.{i0}.conv.weight"]; k = wd.shape[-1]
                pad = (dil * (k - 1)) // 2
                x = F.conv1d(x, wd, stride=(stride if r == 0 else 1),
                             padding=(pad,), dilation=(dil,), groups=wd.shape[0])
                x = F.conv1d(x, sd[f"{base}.mconv.{i0+1}.conv.weight"])
                x = _bn1d(x, sd, f"{base}.mconv.{i0+2}")
                if r < repeat - 1:
                    x = F.relu(x)
        else:
            w = sd[f"{base}.mconv.0.conv.weight"]; pad = (dil * (w.shape[-1] - 1)) // 2
            x = F.conv1d(x, w, stride=(stride,), padding=(pad,), dilation=(dil,))
            x = _bn1d(x, sd, f"{base}.mconv.1")
        if cfg.get("residual", False):
            r_ = F.conv1d(inp, sd[f"{base}.res.0.0.conv.weight"])
            r_ = _bn1d(r_, sd, f"{base}.res.0.1")
            x = x + r_
        x = F.relu(x)
    logits = F.conv1d(x, sd["decoder.decoder_layers.0.weight"],
                      sd["decoder.decoder_layers.0.bias"])      # (B,29,T)
    return logits.transpose(1, 2)                              # (B,T,29)


def _bn1d(x, sd, prefix, eps=1e-3):
    w = sd[f"{prefix}.weight"]; b = sd[f"{prefix}.bias"]
    m = sd[f"{prefix}.running_mean"]; v = sd[f"{prefix}.running_var"]
    return (x - m[None, :, None]) / torch.sqrt(v[None, :, None] + eps) \
        * w[None, :, None] + b[None, :, None]


# ----------------------------- DeepSpeech2 -----------------------------

def ds2_logits_batch(mdl, wavs):
    """wavs: (B, L) -> logits (B, T, 29)."""
    sd = mdl.sd
    spec = torch.stft(wavs, n_fft=320, hop_length=160, win_length=320,
                      window=mdl.window, center=True, pad_mode="reflect",
                      return_complex=True).abs()
    spec = torch.log1p(spec)                                   # (B,161,T)
    spec = (spec - spec.mean(dim=(1, 2), keepdim=True)) / (spec.std(dim=(1, 2), keepdim=True) + 1e-10)
    x = spec[:, None]                                          # (B,1,161,T)
    x = F.conv2d(x, sd["conv.seq_module.0.weight"], sd["conv.seq_module.0.bias"],
                 stride=(2, 2), padding=(20, 5))
    x = torch.clamp(_bn2d(x, sd, "conv.seq_module.1"), 0, 20)
    x = F.conv2d(x, sd["conv.seq_module.3.weight"], sd["conv.seq_module.3.bias"],
                 stride=(2, 1), padding=(10, 5))
    x = torch.clamp(_bn2d(x, sd, "conv.seq_module.4"), 0, 20)
    B, C, Fr, T = x.shape
    x = x.reshape(B, C * Fr, T).transpose(1, 2)               # (B,T,1312)
    for layer in range(5):
        if layer > 0:
            x = _bn_seq(x, sd, f"rnns.{layer}.batch_norm.module")
        x = _bilstm_batch(x, sd, f"rnns.{layer}.rnn")
    x = _bn_seq(x, sd, "fc.0.module.0")
    return x @ sd["fc.0.module.1.weight"].T                   # (B,T,29)


def _bn2d(x, sd, prefix, eps=1e-5):
    w = sd[f"{prefix}.weight"]; b = sd[f"{prefix}.bias"]
    m = sd[f"{prefix}.running_mean"]; v = sd[f"{prefix}.running_var"]
    sh = [1, -1, 1, 1]
    return (x - m.view(sh)) / torch.sqrt(v.view(sh) + eps) * w.view(sh) + b.view(sh)


def _bn_seq(x, sd, prefix, eps=1e-5):
    # x: (B,T,D); BN over D
    w = sd[f"{prefix}.weight"]; b = sd[f"{prefix}.bias"]
    m = sd[f"{prefix}.running_mean"]; v = sd[f"{prefix}.running_var"]
    return (x - m) / torch.sqrt(v + eps) * w + b


def _bilstm_batch(x, sd, prefix):
    """x: (B,T,D) -> (B,T,1024), directions summed."""
    B, T, _ = x.shape
    outs = []
    for suffix, rev in (("", False), ("_reverse", True)):
        wih = sd[f"{prefix}.weight_ih_l0{suffix}"]
        whh = sd[f"{prefix}.weight_hh_l0{suffix}"]
        bias = sd[f"{prefix}.bias_ih_l0{suffix}"] + sd[f"{prefix}.bias_hh_l0{suffix}"]
        H = whh.shape[1]
        seq = torch.flip(x, [1]) if rev else x
        pre = seq @ wih.T + bias                               # (B,T,4H)
        h = x.new_zeros(B, H); c = x.new_zeros(B, H); res = []
        for t in range(T):
            g = pre[:, t] + h @ whh.T
            i, f_, gg, o = g.split(H, dim=1)
            c = torch.sigmoid(f_) * c + torch.sigmoid(i) * torch.tanh(gg)
            h = torch.sigmoid(o) * torch.tanh(c)
            res.append(h)
        out = torch.stack(res, dim=1)
        outs.append(torch.flip(out, [1]) if rev else out)
    return outs[0] + outs[1]


def decode_batch(logits, labels, blank):
    """logits: (B,T,C) -> list of (text, argmax_path)."""
    paths = logits.argmax(-1)
    return [(ctc_greedy(logits[b], labels, blank), paths[b]) for b in range(logits.shape[0])]
