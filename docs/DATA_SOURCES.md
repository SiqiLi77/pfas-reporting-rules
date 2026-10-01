# Versioned monitoring sources

## EPA occurrence archives

- Source: https://www.epa.gov/dwucmr/occurrence-data-unregulated-contaminant-monitoring-rule
- UCMR 5 version used for the final analysis: archive retrieved **September 13, 2026**, with a recorded Last-Modified date of **August 28, 2026**.
- Filename: `data/ucmr_raw/ucmr5-occurrence-data_final_20260828.zip`
- SHA-256: `ff8a6bf937823cdd75295d02f662966e8473eececb47190c406b9de53b2b6389`
- The UCMR 3 archive and January 15, 2026 UCMR 5 snapshot are included for matched comparisons and release-vintage checks. They are not pooled to create the final modern cohort.

All released input files are individually identified in `reproducibility/ASSET_FILES.csv`. The final study uses distinct eligibility rules for cross-cycle matching, complete Method 533 panels, and model evaluation. A location identifier is not by itself a sample-event identifier: panel comparisons require the same sampling date, sample identifier, and method panel.

## Washington

- Agency: Washington State Department of Health.
- Source landing page: https://doh.wa.gov/data-and-statistical-reports/environmental-health/drinking-water-system-data/sentry-internet
- Sample portal: https://fortress.wa.gov/doh/eh/portal/odw/si/downloadsreports.aspx
- The release includes analytic result/input/outcome records and current system/source proxy variables needed for the reported eligibility and prediction analyses. Owner addresses, contacts, and bulk administrative exports are excluded.
- The fixed-margin 16-query evaluation uses **1,923 events**; the native-target auxiliary evaluation uses **1,819 events**. Their populations, thresholds, and weights differ. The supported native source comparison has **1,818 events**.
- Current administrative attributes are proxies. They must not be represented as verified attributes at the historical sampling date. These post-development evaluations are not a new prospectively untouched blind test.

## West Virginia

- USGS release: McAdoo, M. A. *Per- and Polyfluoroalkyl Substances in Drinking Water at Select Public Water Systems in West Virginia, 2024*, version 2.0 (September 2025).
- Persistent source identifier: https://doi.org/10.5066/P13ZNTD3
- Included records support the finished-water eligibility and within-record threshold reclassification analyses. Reliable historical system attributes were not obtained for a transported model evaluation; the release makes no claim that this constitutes independent predictive validation.

## Files not included

Manuscript drafts, author/account credentials, private network configuration, server launch scripts, machine identities, personal contact exports, and unrelated projects are not part of this release. Public analytical identifiers are retained where necessary for same-record matching and cluster-aware evaluation.
