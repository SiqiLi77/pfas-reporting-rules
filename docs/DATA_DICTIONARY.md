# Reading the computational data

## Identifiers and outcomes

`location_id` identifies the entry-point location used by the particular cohort constructor. `pws_id`/`PWSID` identifies a public water system; `state`/`State` is the UCMR jurisdiction or Washington label. Strings must be read as strings to preserve identifiers. Method 533 same-event tables additionally retain date, sample, facility, and sample-point identifiers.

The fixed six-target bit order is:

| Bit | Target |
|---:|---|
| 0 | PFBS |
| 1 | PFHpA |
| 2 | PFHxS |
| 3 | PFNA |
| 4 | PFOA |
| 5 | PFOS |

`observed_state_code = sum(y[j] * 2**j)` ranges from 0 to 63. Its number of set bits is the report count. Exact patterns require all six states; a designated co-reporting pair permits other targets to be reported. These are different endpoints.

`old_visible_code` records the six measurements retained at older cutoffs. Those retained positives must remain positive in valid reconstructed distributions. A withheld low-threshold outcome is an evaluation label, not an allowed predictor.

## Allowed and withheld information

In the strict six-target cohort:

- `old_visible__*` and `old_logratio__*` encode target information visible at older numeric cutoffs.
- `old_logmrl__*` and `native_logmrl__*` identify reporting limits.
- `region`, `pws_size`, `water_type`, and month sine/cosine encode available site/time metadata.
- `y_native_detect__*`, `y_hidden_by_old_mrl__*`, and `y_native_logratio__*` are outcomes. Their coexistence in an analysis file does not authorize their use as predictors.
- The final 67-dimensional industrial-background block is zeroed. It supplies no active industrial-site information.

Auxiliary files add information from analytes outside the six targets. The principal three-analyte set is PFHxA, PFPeA, and PFBA, selected using training data. Auxiliary report count, individual flags, and quantified measurements represent different input conditions. Auxiliary count is not the target report count.

## Arrays

Fixed-joint `payload/fold_N.npz` files contain distributions with shape `(locations × 16 queries, 64)` under `factorized`, `conditional_shared_shock`, and `conditional_gaussian_factor2`. Metadata gives row order, held-out fold, threshold coordinate `alpha`, and outcome code.

The 16 interior coordinates are 0.05, 0.10, 0.15, 0.20, 0.30, 0.35, 0.40, 0.45, 0.55, 0.60, 0.65, 0.70, 0.80, 0.85, 0.90, and 0.95. The native endpoint `alpha=0` is stored separately in `native_probabilities_fold_N.npz`. Native results must not be substituted for 16-query averages.

Auxiliary `foldN/predictions.npz` contains an `index` into its cohort and keys of the form `input_condition__model`. `baseline` means no auxiliary measurements; `selected3` means the three selected measurements. `independent`, `linear`, and `nonlinear` correspond to the logistic, linear, and nonlinear families used in the paper.

Probability arrays can be read with `numpy.load(..., allow_pickle=False)`. Released checkpoints are selected auxiliary PyTorch checkpoints; load trusted checkpoints with the installed PyTorch version and use its restricted `weights_only` loading where compatible.

## Scores and units

Log scores are natural-log losses in **nats**. Pattern loss equals report-count loss plus conditional-identity loss. Counts of zero or six uniquely determine the six-target identity and contribute zero conditional-identity loss. Regularization is applied to the same 64-state distribution before its count aggregation, with epsilon `1e-12` in the principal decomposition.

Rates in computational CSVs are generally fractions unless a `unit` field or column name says percent. Group contrasts are `SW minus GW`. Bias is **predicted/retained rate minus measured native-limit rate**. A loss reduction is reference loss minus candidate loss: positive favors the candidate. Percentage points are not relative percent changes.

Supply-system source category describes the systems supplying the sampled drinking water, not direct samples of untreated groundwater or surface water. Jurisdiction/PWS-equal, PWS-equal, event-equal, and pooled-location standardization are distinct estimands. Consult the figure provenance and the Text S1 cohort matrix exported in `results/tables/supporting/` before comparing numbers.
