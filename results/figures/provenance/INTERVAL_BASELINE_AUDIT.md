# V1.58 interval and empirical-reference audit

Read-only audit completed 2026-10-01. No fitting or bootstrap simulation was performed. Frozen point estimates and percentile endpoints were compared directly; existing stored bootstrap draws were used only to replay endpoint quantiles.

## Figure 5: absolute complete-pattern loss intervals

All eight Figure 5a plotted estimates and their absolute-loss intervals match `outputs/auxiliary_followup_20260926/national/metrics.csv` with maximum absolute difference below 2e-15. Select `weighting=jurisdiction_pws_equal`, `metric=pattern_nll`, and the model names below. These are absolute loss intervals, not paired-gain intervals.

| Plot model | Auxiliary condition | Source name | Estimate | Lower 95% | Upper 95% |
|---|---|---|---:|---:|---:|
| Linear | None | baseline__linear | 0.6444256677460296 | 0.5836235071815286 | 0.7051704243983605 |
| Linear | Count | decompose_count_linear | 0.5165934700375461 | 0.4720332718450616 | 0.5616268310544071 |
| Linear | Flags | decompose_flags_linear | 0.5159415681347942 | 0.4710701391349935 | 0.5615652737483497 |
| Linear | Flags + concentrations | selected3__linear | 0.5005352283456705 | 0.4633958788766718 | 0.5385808664150517 |
| Nonlinear | None | baseline__nonlinear | 0.6351204025868232 | 0.5728627184747009 | 0.6977177280398523 |
| Nonlinear | Count | decompose_count_nonlinear | 0.508449148047492 | 0.4610224926504409 | 0.5561501400568118 |
| Nonlinear | Flags | decompose_flags_nonlinear | 0.5093875832954535 | 0.4603467006681385 | 0.5587442457236502 |
| Nonlinear | Flags + concentrations | selected3__nonlinear | 0.4978233316479969 | 0.4550776225580967 | 0.5410050059831449 |

Absolute scores use all 26,423 events. There are 3,491 events with observed target count one through five; these support the separate informative conditional-identity evaluation. The primary absolute-score intervals use 5,000 paired PWS resampling draws within jurisdiction, preserving jurisdiction/PWS-equal evaluation weights. Jurisdiction resampling is a separate sensitivity used in the paired identity panel, not the plotted absolute-loss interval source. The quantiles are 2.5% and 97.5%; all 5,000 draws are valid.

All six Figure 5c estimates and intervals match `outputs/auxiliary_followup_20260926/washington/current_metadata_proxy/metrics.csv` with maximum absolute difference below 2e-15. Select the same metric and weighting label; because all events are in Washington, `jurisdiction_pws_equal` here is PWS-equal. The corresponding file is copied into the historical publication source as `outputs/est_environmental_reconstruction_v1_34_20260927/source_tables/wa_proxy_metrics.csv`.

| Plot model | Auxiliary condition | Source name | Estimate | Lower 95% | Upper 95% |
|---|---|---|---:|---:|---:|
| Logistic | None | baseline__independent | 0.6618291791868157 | 0.540859182409959 | 0.7885540346699494 |
| Linear | None | baseline__linear | 0.5070012134085898 | 0.4219377418044561 | 0.5932017000203283 |
| Nonlinear | None | baseline__nonlinear | 0.478977415275306 | 0.4002227134126904 | 0.560454274813759 |
| Logistic | Three auxiliaries | selected3__independent | 0.4713808701478008 | 0.3950634302112819 | 0.5520152542113433 |
| Linear | Three auxiliaries | selected3__linear | 0.4222943870118146 | 0.3521182083863268 | 0.4957704838060545 |
| Nonlinear | Three auxiliaries | selected3__nonlinear | 0.3955099319625447 | 0.3312909249860974 | 0.4625541834842537 |

Washington pattern scores use 1,819 events; 133 have observed target count one through five. Figure 5c absolute-loss intervals use 5,000 PWS resampling draws. County intervals are a separate sensitivity in panel d. Both schemes retain the same PWS-equal estimand.

Code lineage: `src/report_auxiliary_followup_20260926.py` lines 99–106 exports the national pattern ladder from `national/metrics.csv`; `src/summarize_auxiliary_followup_20260926.py` lines 16–49 computes absolute-score percentile intervals from `cluster_bootstrap`; `src/summarize_auxiliary_information_20260926.py` lines 12–31 defines the paired PWS-within-jurisdiction draw and separate jurisdiction draw. `src/figures_est_v1_35.py` lines 150–165 exports the historical figure ladder and Washington absolute pattern table. `src/reorganize_result_figures_v1_39.py` lines 131–148 maps them to current Figure 5a/c.

## Figure 6a: count-prior group-rate intervals

The intervals exist and are based on the same frozen source-comparison support and draws as the other fixed-margin rows. The statement in old Table S30 that the two group-rate intervals were not tabulated is outdated relative to the archived source table and should be replaced with the verified intervals.

Source: `outputs/marginal_source_diagnostic_20260927/national/source_rates_and_contrasts.csv`, selecting `condition=count_prior`, `endpoint=multiple`, `quantity=GW_rate` or `SW_rate`. The copied publication source is `outputs/est_target_specific_reconstruction_v1_36_20260927/marginal_diagnostic/national/source_rates_and_contrasts.csv`, CSV lines 368–369. The V1.46 figure file adds plot units as percentages without altering original probability columns.

| Group | Estimate (%) | Lower 95% (%) | Upper 95% (%) |
|---|---:|---:|---:|
| Groundwater systems | 9.505259683751808 | 9.327589947381455 | 9.727721920751756 |
| Surface-water systems | 7.098905345832993 | 6.92654587543929 | 7.305226001620703 |

Support: 15,723 events, 51 jurisdiction-by-size strata, 29 jurisdictions. Source-group means use the common pooled-location stratum weights. PWS clusters are resampled within jurisdiction with all their records retained, and the same draws are used across groups/models. Of 5,000 attempted draws, 4,998 had valid common support. Intervals are pointwise percentile intervals, conditional on saved predictions. Frozen-draw replay directly reproduced both group-rate quantiles from `outputs/marginal_source_diagnostic_20260927/national/source_bootstrap.npz`.

The count prior comes from the native-limit anchor of `outputs/significance_extension_20260924/count_baseline/training_count_priors.csv`, spreads mass equally across patterns of each count, and is projected to the same six frozen marginal probabilities. Source construction and replay: `src/marginal_source_diagnostic_20260927.py` lines 28–65 and `src/environmental_reconstruction_20260927.py` lines 200–294. The verification file confirms original model draws reproduced and count-prior maximum marginal error below 1e-11.

## Figures 6–7: empirical composition identity

The plotted `training_composition` / `Empirical composition` is the **query-wise empirical reference**, corresponding to the row `Query-wise empirical` in Table S19 Panels A and B. It is not either coherent reference at lambda=1 and not the empirical conditional-identity control in Table S29.

Training locations have equal weight within the permitted training folds. Outer test and designated validation folds are excluded. Input is only the exact six-bit old-visible mask m. For each queried threshold, the pooled 64-state training histogram receives 1/64 pseudocount per state, totaling one. Its compatible support is restricted to states s containing every visible bit in m, and renormalized. The same-mask histogram then receives one virtual observation from that pooled prior:

`q_m(s) = [n_ms + pi_ms] / [n_m + 1]`.

Unseen masks use the compatible pooled prior. The code constants are `PRIOR_TOTAL=1.0` and `SHRINKAGE=1.0`; no lambda grid or coherent-coupling parameter is used for this reference. Each cutoff distribution is valid, but cross-threshold coherence is not guaranteed. Single-analyte marginal probabilities can change.

Exact sources: `src/build_ucmr5_training_composition_baseline.py` lines 26–85; `outputs/joint_information_decomposition_20260916/baseline/PROTOCOL.json`; `src/analyze_four_extensions_selective_v1_30.py` lines 72–80 directly invokes this construction at alpha=0 and saves the native predictions later consumed by environmental reconstruction. `src/environmental_reconstruction_20260927.py` lines 24 and 51–81 includes those native `training_composition` predictions. Figure 6 evaluates the native endpoint on the 15,723-event source support, and Figure 7 evaluates the native endpoint on 21,134 aligned events. Table S19 reports 16-threshold-average scores from the same comparator family; its numerical losses should not be substituted for native-endpoint figure values.

Suggested concise formal description: “Empirical composition denotes the query-wise, location-equal training reference conditioned on the old-visible mask, with a total pooled pseudocount of one and one same-mask prior observation (Text S13; Table S19, Query-wise empirical row). It can change the single-PFAS margins.”

## Figure S6: identification endpoints and endpoint sampling intervals

All four plotted identification intervals and all eight endpoint sampling intervals match `outputs/est_four_extensions_v1_30_20260925/environment/targeted_sum_contrast_identification_bounds.csv` and replay directly from the archived `environment_bootstrap.npz` burden means. Maximum absolute endpoint/quantile difference is below 2e-14.

| Panel | Lower identification endpoint | Upper endpoint | Lower endpoint 95% sampling interval | Upper endpoint 95% sampling interval |
|---|---:|---:|---|---|
| Shared six | -20.351704699133737 | 19.687483659728972 | -20.712766052310737 to -19.97613546559584 | 19.293196436368945 to 20.07298935692122 |
| Complete 25 | -102.24550336851168 | 103.64869310459343 | -102.83855255034324 to -101.61804407093004 | 103.03052685718114 to 104.25760110980606 |
| Added trio | -8.942830875860796 | 10.98367400628958 | -9.249667362126964 to -8.609495707685117 | 10.693093718275902 to 11.2983743555643 |
| Other 16 added | -72.95096779351717 | 72.9775354385749 | -73.02489093866879 to -72.86188564939367 | 72.90394127740011 to 73.0631727239414 |

Units are ng/L, surface-water minus groundwater systems. Let Q be the sum of quantified reports and U=Q plus native MRLs for unreported analytes. Full targeted-sum group contrast bounds are `Q_SW - U_GW` and `U_SW - Q_GW`; unreported measurements range over [0, native MRL), so the upper concentration limit is approached rather than attained. The broad interval between identification endpoints is not a sampling confidence interval. Endpoint confidence intervals use 1,999 estimable paired PWS draws of 2,000 attempted draws, retaining common standardization weights; the separate endpoint intervals are pointwise 95% percentiles.

Support is the 20,162-event same-record standardized cohort, with 7,911 PWSs. Code: `src/analyze_four_extensions_environment_v1_30.py` lines 38–49 construct Q/U and line 87–91 calculate endpoint quantiles; `src/draw_four_extensions_v1_30.py` lines 30–46 exports the endpoint plotting file; `src/reorganize_result_figures_v1_39.py` lines 261–273 preserves broad identification lines with open endpoint markers and narrow sampling intervals. Retitle b “Identification bounds”; preserve this distinction in caption and Table S13.

## Editorial decisions

1. Preserve all verified error bars; none requires deletion or numerical reconstruction from images.
2. Add count-prior group-rate intervals to Table S30 and remove the outdated missing-interval statement.
3. Identify the query-wise empirical reference once in Text S17 and concisely in Figures 6–7 captions; do not relabel it as coherent.
4. Record Figure 5 absolute-score intervals and Figure S6 endpoint intervals in formal figure-data files, with source path, cohort, weights and resampling fields.
5. All conclusions remain conditional on frozen fitted predictions and exploratory after model development. No new analysis is needed to resolve these advice items.
