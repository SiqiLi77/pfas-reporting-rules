# Reproduction routes

## 1. Inspect final numerical evidence

No model software is needed to read `results/figures/` or the final Word-exported table CSVs. The figure provenance identifies panel-specific cohorts, weights, interval sources, and units. Formal SI numbering is S1–S35; detailed computational records use their historical **C identifiers**, not current S numbering. The mapping is retained under `results/computational_records/`.

## 2. Recompute frozen point results

```bash
python -m pip install -r requirements.txt
python scripts/fetch_assets.py
python scripts/reproduce.py --section all
```

Sections can also be run separately: `observational`, `joint`, or `environmental`. Replay writes only to `replay/` by default. It checks the same-record source-standardized panel rates; fixed-joint pattern/count/identity, CRPS, and Brier losses under both principal and record-equal weights; aligned native chemical scores; and supported source-group rates and contrasts. This is not a new fit or new independent test.

For ZIPs downloaded through the browser:

```bash
python scripts/fetch_assets.py --from-directory /path/to/downloads
python scripts/fetch_assets.py --verify-only
```

The source and data SHA-256 inventories are in `reproducibility/`. Checksum integrity is not, by itself, evidence that a statistical analysis is valid.

## 3. Rebuild same-record cross-cycle transitions from EPA archives

```bash
python src/analyze_ucmr5_threshold_pattern_transitions.py \
  --ucmr3 data/ucmr_raw/ucmr3-occurrence-data.zip \
  --ucmr5 data/ucmr_raw/ucmr5-occurrence-data_final_20260828.zip \
  --output-dir replay/cross_cycle_raw
```

This rebuilds strict same-event matched records and cutoff-pattern transition summaries. It does not replace the separately weighted inferential contrast procedures.

## 4. Resampling and analysis implementation

The corrected independent-inner nested bootstrap is implemented in `src/audit_ucmr5_information_nested_bootstrap.py`: every repeated sampled jurisdiction receives an independent PWS draw. The corresponding frozen paired contrasts are in the fixed-joint asset. Use that method for the reported count–identity intervals, not the inherited cached-inner bootstrap function in the original scoring module.

`src/environmental_reconstruction_20260927.py` supplies native endpoint summation, count-conditioned chemical scoring, source support, standardization, and paired PWS/county resampling. Stored paired draws are supplied so interval provenance remains inspectable without refitting. These intervals condition on frozen predictions and exclude retraining/model-selection uncertainty.

## 5. Model code and optional retraining

The marginal interface is implemented in `run_ucmr5_censored_distribution_screen.py`; nested training margins are implemented in `run_ucmr5_nested_joint_training_margins.py`. Joint structures are in `analyze_ucmr5_conditional_joint_models.py` and `analyze_ucmr5_conditional_gaussian_factor.py`. Optional compiled monotonicity checks are not needed for frozen-score replay.

The auxiliary cohort and model families are implemented in `prepare_auxiliary_information_20260926.py` and `run_auxiliary_information_20260926.py`. After downloading the assets and installing `requirements-training.txt`, the original training entry point accepts:

```bash
python src/run_auxiliary_information_20260926.py --help
# Choose a fresh destination; never use the published frozen data directory.
# Provide the same cohort/schema and the documented training-selected settings.
```

The release retains baseline and three-auxiliary checkpoints, settings, feature specifications, and held-out predictions. Other comparator predictions/results remain available, without uploading every development checkpoint. Original marginal Transformer checkpoints are unavailable in this curated local archive. Consequently, exact frozen marginal inference is not advertised; the supplied out-of-fold margins permit downstream joint replay, and marginal training code permits a separate retraining experiment.

Original modules may include broader development locks or optional secondary entry points. Their filenames and numerical functions are retained for provenance, but the tested supported entry points are the download, replay, raw-transition, and unit-test commands listed here. No remote-server access is required for these checks.

## Unit tests

```bash
python -m unittest discover -s tests -p 'test_ucmr5_joint_information_decomposition.py'
python -m unittest discover -s tests -p 'test_ucmr5_information_nested_bootstrap.py'
python -m unittest discover -s tests -p 'test_environmental_reconstruction_20260927.py'
```

Additional tests use training libraries from `requirements-training.txt`. The final release validation report records exactly what was run; passing unit tests does not convert post-development evaluation into confirmatory validation.
