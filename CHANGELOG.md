# Releases

## v1.1.0 — October 27, 2026

Adds `v2_analysis/`, the scripts, analysis outputs and figure source data of manuscript and Supporting Information V2.06: 26 analysis scripts, the 2 local modules they import and the Drought Monitor download script; 42 analysis outputs (JSON and log); the copies of v1.0.0 plotted values read by the Supporting Information figures, with their checksums (`03_analysis/author_release_v1_0_0/`); and 41 figure source-data files. Every file is byte-identical to the Supporting Information data-and-scripts archive, except README.txt and MANIFEST.sha256, which are rewritten without the reconstruction record (part of the Supporting Information). No raw data are added; their sources and checksums are listed in `v2_analysis/03_analysis/input_checksums.json`. The v1.0.0 materials are unchanged.

## v1.0.0 — October 1, 2026

First public computational release corresponding to paper/SI V1.59. Includes final figure/table data, scientific implementation and numerical tests, analysis-ready monitoring records, frozen probability distributions and selected auxiliary checkpoints, and paired inference records. Core frozen-result replay, EPA raw-transition rebuilding, and 45 unit tests passed before publication.

No new models were fitted for this release. Frozen-score reproduction, optional retraining, and exploratory model evaluation are kept distinct.
