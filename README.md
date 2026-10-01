# PFAS reporting rules and mixture reconstruction

Code, analysis-ready data, frozen predictions, and numerical results for:

**Reporting Limits and Analyte Coverage Shape PFAS Mixture Comparisons in United States Drinking Water**

This repository corresponds to manuscript and Supporting Information **V1.59**. The public computational release is **v1.0.0**. Manuscript drafts are not distributed here.

## Scientific scope

The study asks how reporting limits and analyte coverage change the PFAS mixtures described by drinking-water monitoring, and which environmental or chemical summaries can be reconstructed from restricted records.

The observational analyses compare reporting rules within the same modern measurements and examine administratively matched UCMR 3–5 entry points. Controlled masking hides modern measurements below older reporting cutoffs, retains them as evaluation outcomes, and tests reconstruction of **six shared targets**. It does not reconstruct all unmeasured PFAS or establish historical low-concentration composition.

The six reconstruction targets, in bit order, are **PFBS, PFHpA, PFHxS, PFNA, PFOA, and PFOS**. Additional measurements change the input information; fixed-margin comparisons change the joint probability structure without changing the six marginal probabilities. Counts, chemical composition, and supply-system contrasts are evaluated separately.

## Start here

For the numerical results used to draw the seven main figures and eight supplementary figures, open [`results/figures`](results/figures). For Tables 1–2 and S1–S35, see [`TABLE_INDEX.csv`](results/tables/TABLE_INDEX.csv). Table cells were exported from the final Word files without changing displayed numbers; statistical scope is documented in the companion notes and figure provenance.

For a computational check:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/fetch_assets.py
python scripts/reproduce.py
```

The four data ZIPs are attached to the [v1.0.0 GitHub Release](https://github.com/SiqiLi77/pfas-reporting-rules/releases/tag/v1.0.0), not stored in Git history. The download script verifies ZIP and expanded-file SHA-256 checksums, preserves relative paths, and refuses to overwrite changed data. Approximately 0.64 GB must be downloaded; allow several GB of working space for extraction and numerical replay.

The replay recomputes the main fixed-prediction scores, the same-record panel comparison, and aligned environmental/chemical endpoint estimates. It checks them against the frozen unrounded outputs and writes a local `replay/REPLAY_REPORT.json`. It does **not** refit models or create new validation evidence.

## What is included

| Material | Location | Role |
|---|---|---|
| Final figure data, intervals, and provenance | `results/figures/` | Plotting and numerical inspection |
| Final table blocks and index | `results/tables/` | Displayed values and table-level evidence |
| Detailed computational records | `results/computational_records/` | Strong baselines, null results, sensitivity grids, and supporting extensions |
| Analysis/training code and imported dependencies | `src/` | Scientific implementation |
| Numerical unit tests | `tests/` | Score decomposition, state encoding, resampling, and information constraints |
| Frozen records, probabilities, selected checkpoints, and draws | Release ZIPs | Result replay and selected inference/training inputs |
| Input and asset checksums | `reproducibility/` | Version and integrity checks |

Some original module and folder names contain development dates or legacy model identifiers. Those names are preserved to keep array keys and relative paths consistent; they do not indicate additional active predictors. In particular, the final industrial-background block is zeroed, as described in the study information set.

## Reproducibility boundaries

The portable replay scripts are tested with the published inputs. Original `src/` modules also expose training and analysis functions, but some legacy entry points refer to development locks or intermediate files outside the curated release; these are not the recommended quick-start route. See [reproduction instructions](docs/REPRODUCING.md).

Frozen probability distributions support direct score reconstruction. The release includes selected auxiliary-model checkpoints and fitted joint-model decisions, but **not the original marginal Transformer checkpoints**. Full marginal retraining is therefore a computational rerun, not a promised bitwise restoration of the original neural weights. The publication checks distinguish frozen-score replay from retraining.

Joint-model evaluation and later supporting analyses retain their **exploratory, post-development** status. Resampling intervals condition on the recorded predictions and their specified populations/weights; they do not incorporate retraining or model-selection uncertainty. Washington attributes are current administrative proxies, not verified historical attributes. West Virginia supports descriptive threshold checks, not a completed blind predictive test.

## Data sources and citation

Public monitoring sources are [EPA UCMR occurrence archives](https://www.epa.gov/dwucmr/occurrence-data-unregulated-contaminant-monitoring-rule), [Washington Sentry](https://doh.wa.gov/data-and-statistical-reports/environmental-health/drinking-water-system-data/sentry-internet), and the [USGS West Virginia finished-water release](https://doi.org/10.5066/P13ZNTD3). Release versions and cohort definitions are recorded in [data sources](docs/DATA_SOURCES.md) and [data dictionary](docs/DATA_DICTIONARY.md).

Please identify **release v1.0.0** and its commit when citing these computational materials. No publication DOI is asserted here. GitHub provides versioned access, not a DOI-backed preservation guarantee.

Third-party data retain their original source terms. No new blanket software/data license has been assigned in this release; see [reuse information](NOTICE.md).
