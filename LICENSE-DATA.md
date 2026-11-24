# Data licence

## Our data: CC BY 4.0

Everything under `data/campaign/`, `data/rev1/` and `expected/`, and the derived
designator list `data/external/atco2_info/designators_derived.info`, is
released under the [Creative Commons Attribution 4.0 International
licence](https://creativecommons.org/licenses/by/4.0/). This covers the per-draw
violation vectors, counts, stored transcripts, embedding caches, summaries, the
held-out clip manifest and the generated paper numbers and figures.
Copyright (c) 2026 Ben Wooding and Taylor T. Johnson.

The stored transcripts are outputs of speech recognition models on air-traffic
control recordings. They are included because each certificate is defined
relative to them.

## Third-party data

* `data/external/airlines.dat` comes from [OpenFlights](https://openflights.org/data)
  and is licensed under the [Open Database Licence 1.0](https://opendatacommons.org/licenses/odbl/1-0/).
* No audio is included. ATCOSIM, ATCO2, UWB-ATCC and the LiveATC archives keep
  their own terms. The LiveATC recordings may be used for personal,
  non-commercial purposes but must not be redistributed, so we release only the
  manifest needed to rebuild identical clips from the public source.
* Model weights (QuartzNet15x5, DeepSpeech2, Whisper, Whisper-ATC) are not
  included and keep their own licences.
