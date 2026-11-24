"""RadioStar: star-set reachability verification of neural networks on sound inputs.

First target: Prover's (CAV 2021) FSDD speech classifier, certified with star
sets end-to-end -- waveform dB-box input star, exact affine DFT/mel maps,
approx-star Square/Log transformers, star LSTM (ported from NNV), halfspace
output check. Builds on n2v (Python NNV translation, sibling repo).
"""

__version__ = "0.0.1"
