# V1.58 figure-data companion

This is a figure-table companion, not the complete computational archive or a
new validation exercise. It supplies the frozen plotting tables for the seven
main result figures and eight supporting result figures, with a panel-level
guide to cohorts, endpoints, weights, intervals and provenance. No models were
refitted, no new bootstrap draws generated, and no numerical rows changed.

## Contents

- `Fig1`–`Fig7` and `S1`–`S8`: all **51 existing plotting CSV files**, copied byte-for-byte
  from `outputs/est_main_figure_balance_v1_46_20260928/plot_data`. This includes
  48 inherited plotting tables and three existing presentation-derived tables (3a panel
  expansion, 4d pattern membership, 6b bias plane). All original rows remain,
  including null/negative findings and rows not selected in the current art.
- `PANEL_MANIFEST.csv`: current panel → file → endpoint, population, weighting,
  interval/resampling definition and upstream source path. Paths beginning
  `outputs/` or `src/` are repository-relative provenance; the files themselves
  are not all included in this small companion.
- `FILE_MANIFEST.csv`: exact original path, row count, columns and SHA-256 of
  each unchanged plotting CSV, plus associations with current panels.
- `support_tables`: separately labelled, source-verified existing interval
  rows for Figures 5a/c, Figure 6a count-prior rates, and Figure S6b endpoints,
  plus supporting evidence for curated SI Tables S2, S6, S9, S18, and S35.
  These are not original plot CSVs, newly generated inferential results, or
  evidence of newly fitted models. `SUPPORT_SOURCE.csv` records their sources.
- `provenance`: inherited source-map lineage and the verified interval/baseline
  audit. Exact Figure 5 absolute intervals, Figure 6 count-prior rate intervals,
  and Figure S6 bound endpoint intervals are documented there. The audit used
  existing stored bootstrap draws only to replay reported percentile endpoints.

## Interpretation / plotting

1. Cohorts, weights and thresholds differ across figures. Do not compare their
   percentage gains as if they were one evaluation. Figure 5 informative identity
   loss is conditional on observed K6=1–5; Figure 4 identity and Figure 7 chemical
   losses are all-event conditional scores with determined states contributing
   zero. Actual counts enter scoring, not prediction input.
2. A probability is multiplied by 100 for a rate in percent. A difference in
   probabilities is multiplied by 100 for percentage points. Losses use natural
   logarithms (nats). Figure 7 displays the original nats multiplied by 1,000.
   Follow explicit `plot_*` / unit columns instead of guessing scale from values.
3. Interval endpoints are not error-bar lengths. For asymmetric errors use
   estimate−lower and upper−estimate. Read the manifest's interval columns;
   Figure 7 and Figure S7 display simultaneous intervals, not their also-retained
   pointwise columns. Do not add invented joint ellipses to the Figure 6 bias plane.
4. Figure 2b and Figure 4d describe **exact** shared-six patterns: every other
   target is unreported. Figure 7 designated pairs instead allow other targets.
   The common matrix order is PFBS, PFHpA, PFHxS, PFNA, PFOA, PFOS.
5. Figure S2 bands are randomized-reference envelopes, not sampling CIs. Figure
   S6b's broad identification region is not a CI; narrow intervals separately
   quantify sampling uncertainty of each bound endpoint.
6. “Empirical composition” in Figures 6–7 is the query-wise location-equal
   training reference conditioned on the old-visible mask (Table S19, Query-wise
   empirical row), not a coherent variant and not the auxiliary conditional-
   identity reference in Table S29. Detailed construction is in the audit.

## Legacy names and deliberate preservation

Some filenames, `panel` fields, label strings and embedded table-number strings
predate current artwork. They are retained because this companion promises
unchanged CSV bytes. Use `PANEL_MANIFEST.csv` for current mapping:

- `Fig6/Fig6b_marginal_bias.csv` belongs to current Figure **6c**; the separate
  `Fig6b_two_group_bias.csv` belongs to current **6b**.
- Figure 7 source files may carry former Figure 8 panel names.
- `Fig3a_source_rates.csv` retains older-cutoff rows not displayed in current 3a.
- `Fig4b_component_gains.csv` retains pattern-gain rows beyond the two displayed
  decomposition components; `Fig4a_absolute_losses.csv` retains its legacy S23
  source label (current curated SI S16).
- Original Origin formatting tables retain old wording; scientific definitions
  in current captions / manifest take precedence over historical display labels.

## Scope of provenance checks

Copy integrity is exhaustive for all 51 included plot CSVs (52 inherited CSVs
when the source-map CSV is counted separately). The separate interval
audit directly traced every Figure 5a/c absolute interval, Figure 6a count-prior
group-rate interval, empirical-reference identity, and all Figure S6b endpoint
intervals to frozen analysis sources. It is **not** an independent replay of
every scientific result. Remaining metadata are taken from current SI methods
and established figure-builder lineage. Figure S7's 32 rows additionally match
the canonical simultaneous-band table exactly; its builder uses 20,000 single-
level PWS draws. The original Figure S1 plot CSV lacks row support fields, which
are supplied separately here rather than inserted into unchanged plotting data.

The package contains summary data, not the full raw monitoring tables, model
weights, saved prediction arrays or full computational supplement. Public data
availability for the article requires a separately accessible archive.
