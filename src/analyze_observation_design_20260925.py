#!/usr/bin/env python3
"""Fixed 2x2x2 observation comparison, paired coverage and chemical recurrence."""
from __future__ import annotations
from itertools import combinations, permutations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from prepare_observation_design_20260925 import (
    ROOT, OUT, COHORT, A25, SIX, MRL, OLD, AI, sha, dump, verify_lock,
)

REPS = 2000
SEED = 202609251
PAIRS = list(combinations(range(25), 2))
PI = np.array([p[0] for p in PAIRS])
PJ = np.array([p[1] for p in PAIRS])
PAIR_NAMES = [A25[i] + ' + ' + A25[j] for i, j in PAIRS]
SIXMASK = np.array([a in SIX for a in A25])
STRATEGIES = [(t, b, r) for t in range(2) for b in range(2) for r in range(2)]
SID = {s: ''.join(map(str, s)) for s in STRATEGIES}


def visible(c, threshold_native, expanded, native_multiplier=1.):
    cutoff = np.array([MRL[a] * native_multiplier if threshold_native or a not in OLD else OLD[a] for a in A25])
    mask = np.isfinite(c) & (c >= cutoff[None, :] - 1e-12)
    if not expanded:
        mask[:, ~SIXMASK] = False
    return mask


def pair_flags(a):
    return a[:, PI] & a[:, PJ]


def strategy_pairs(c1, c2, s, native_multiplier=1.):
    t, b, r = s
    p = pair_flags(visible(c1, t, b, native_multiplier))
    if r:
        p |= pair_flags(visible(c2, t, b, native_multiplier))
    return p


def fixtures():
    c1 = np.full((2, 25), np.nan)
    c2 = np.full((2, 25), np.nan)
    c1[0, AI['PFOS']] = .1
    c2[0, AI['PFOA']] = .1
    assert not strategy_pairs(c1, c2, (1, 1, 1))[0].any()
    c1[1, AI['PFOS']] = OLD['PFOS']
    c1[1, AI['PFOA']] = OLD['PFOA']
    assert strategy_pairs(c1, c2, (0, 0, 0))[1].sum() == 1
    # An extra-analyte report never creates a historical-six report.
    c1[0, AI['PFPeA']] = MRL['PFPeA']
    assert not strategy_pairs(c1, c2, (1, 0, 0))[0].any()
    assert strategy_pairs(c1, c2, (1, 1, 0))[0].sum() == 1


def verify_sets(c1, c2, arrays):
    """No vector pair code reused: enumerate sample-level sets for every location."""
    totals = {SID[s]: [0, 0, 0.] for s in STRATEGIES}
    all_flags = [np.isfinite(c1), np.isfinite(c2)]
    for i in range(len(c1)):
        reference = set()
        for tt in range(2):
            atoms = {j for j in range(25) if all_flags[tt][i, j]}
            reference.update(combinations(sorted(atoms), 2))
        for s in STRATEGIES:
            t, b, r = s
            retained = set()
            for c in ([c1, c2] if r else [c1]):
                atoms = []
                for j, a in enumerate(A25):
                    if not b and a not in SIX:
                        continue
                    cutoff = MRL[a] if t or a not in OLD else OLD[a]
                    if np.isfinite(c[i, j]) and c[i, j] >= cutoff - 1e-12:
                        atoms.append(j)
                retained.update(combinations(atoms, 2))
            z = arrays[SID[s]]
            assert int(z[i].sum()) == len(retained)
            assert {PAIRS[j] for j in np.flatnonzero(z[i])} == retained
            assert retained <= reference
            totals[SID[s]][0] += bool(retained)
            totals[SID[s]][1] += len(retained)
            totals[SID[s]][2] += len(retained) / len(reference) if reference else 0.
    return totals


def bootstrap(d, f):
    keys = list(d.groupby(['state', 'pws_id'], sort=True).indices)
    groups = d.groupby(['state', 'pws_id'], sort=True).indices
    sums = np.stack([f[groups[k]].sum(axis=0) for k in keys])
    counts = np.array([len(groups[k]) for k in keys])
    means = sums / counts[:, None]
    states = sorted(d.state.unique())
    si = {s: np.array([i for i, k in enumerate(keys) if k[0] == s]) for s in states}
    point = {'location_equal': f.mean(axis=0), 'pws_equal': means.mean(axis=0),
             'jurisdiction_pws_equal': np.stack([means[ix].mean(axis=0) for ix in si.values()]).mean(axis=0)}
    draws = {w: np.zeros((REPS, f.shape[1])) for w in point}
    draw_n = np.zeros(REPS)
    rng = np.random.default_rng(SEED)
    for ix in si.values():
        for start in range(0, REPS, 100):
            end = min(start+100, REPS)
            mult = rng.multinomial(len(ix), np.full(len(ix), 1/len(ix)), size=end-start)
            draws['location_equal'][start:end] += mult @ sums[ix]
            draw_n[start:end] += mult @ counts[ix]
            draws['pws_equal'][start:end] += mult @ means[ix] / len(keys)
            draws['jurisdiction_pws_equal'][start:end] += mult @ means[ix] / len(ix) / len(states)
    draws['location_equal'] /= draw_n[:, None]
    return point, draws


def divide(x, n, d):
    return np.divide(x[..., n], x[..., d], out=np.full(x[..., n].shape, np.nan), where=x[..., d] > 0)


def analyze():
    verify_lock()
    fixtures()
    manifest = json.loads((COHORT / 'MANIFEST.json').read_text())
    assert sha(COHORT / 'paired_panels.csv.gz') == manifest['paired_panels.csv.gz']
    d = pd.read_csv(COHORT / 'paired_panels.csv.gz', dtype={'location_id': str, 'pws_id': str, 'state': str})
    c1 = d[[f'first_conc_ugL__{a}' for a in A25]].to_numpy(float)
    c2 = d[[f'next_conc_ugL__{a}' for a in A25]].to_numpy(float)
    for t, c in [('first', c1), ('next', c2)]:
        assert np.array_equal(np.isfinite(c), d[[f'{t}_report__{a}' for a in A25]].to_numpy(bool))
    arrays = {SID[s]: strategy_pairs(c1, c2, s) for s in STRATEGIES}
    reference = arrays['111']
    ref_count = reference.sum(axis=1)
    den_pos = ref_count > 0
    assert np.array_equal(den_pos, d.first_k25.ge(2) | d.next_k25.ge(2))
    for s in STRATEGIES:
        assert not (arrays[SID[s]] & ~reference).any()
        for factor in range(3):
            if s[factor] == 0:
                up = list(s); up[factor] = 1
                assert not (arrays[SID[s]] & ~arrays[SID[tuple(up)]]).any()
    columns = ['reference_positive', 'reference_pairs']
    features = [den_pos.astype(float), ref_count.astype(float)]
    metrics = {}
    for s in STRATEGIES:
        name = SID[s]
        n = arrays[name].sum(axis=1)
        for metric, f, denominator in [
            ('binary_coverage', n > 0, 'reference_positive'),
            ('location_equal_pair_coverage', np.divide(n, ref_count, out=np.zeros(len(d)), where=ref_count > 0), 'reference_positive'),
            ('pooled_pair_coverage', n, 'reference_pairs')]:
            col = name+'__'+metric
            columns.append(col); features.append(np.asarray(f, float))
            metrics[(name, metric)] = (len(columns)-1, columns.index(denominator))
    p1 = pair_flags(np.isfinite(c1)); p2 = pair_flags(np.isfinite(c2))
    pair_rows = []
    pair_groups = []
    for j, (a, b) in enumerate(PAIRS):
        category = 'shared_shared' if A25[a] in SIX and A25[b] in SIX else ('additional_additional' if A25[a] not in SIX and A25[b] not in SIX else 'shared_additional')
        pair_groups.append(category)
        first = int(p1[:, j].sum()); second = int(p2[:, j].sum())
        both = int((p1[:, j] & p2[:, j]).sum())
        pair_rows.append({'pair': PAIR_NAMES[j], 'analyte_a': A25[a], 'analyte_b': A25[b],
                          'coverage_group': category, 'first': first, 'next': second, 'both': both,
                          'first_only': first-both, 'next_only': second-both,
                          'either': first+second-both, 'repeat_report_fraction': both/first if first else np.nan,
                          'denominator_locations': len(d), 'support_at_least_50': first >= 50})
    pair_table = pd.DataFrame(pair_rows).sort_values(['first', 'pair'], ascending=[False, True])
    top = pair_table.head(12).pair.tolist()
    pair_specs = {}
    for name in top:
        j = PAIR_NAMES.index(name)
        columns += [name+'__first', name+'__both']
        features += [p1[:, j].astype(float), (p1[:, j] & p2[:, j]).astype(float)]
        pair_specs[name] = (len(columns)-1, len(columns)-2)
    for group in ['shared_shared', 'shared_additional', 'additional_additional']:
        ix = np.array(pair_groups) == group
        columns += [group+'__first', group+'__both']
        features += [p1[:, ix].sum(axis=1).astype(float), (p1[:, ix] & p2[:, ix]).sum(axis=1).astype(float)]
        pair_specs[group] = (len(columns)-1, len(columns)-2)
    f = np.column_stack(features)
    print(f'Running {REPS:,} paired PWS-cluster resamples for {len(d):,} locations', flush=True)
    points, draws = bootstrap(d, f)
    rows = []
    values, boot_values = {}, {}
    for w, p in points.items():
        for (sid, metric), (num, den) in metrics.items():
            value = float(divide(p, num, den))
            bs = divide(draws[w], num, den)
            values[(w, sid, metric)] = value
            boot_values[(w, sid, metric)] = bs
            lo, hi = np.nanquantile(bs, [.025, .975])
            t, b, r = map(int, sid)
            rows.append({'strategy': sid, 'six_native_limits': t, 'expanded_panel': b,
                         'include_next_sample': r, 'weighting': w, 'metric': metric,
                         'estimate': value, 'ci95_lower': lo, 'ci95_upper': hi,
                         'locations': len(d), 'reference_positive_locations': int(den_pos.sum()),
                         'reference_location_pairs': int(ref_count.sum()),
                         'captured_positive_locations': int(arrays[sid].any(axis=1).sum()),
                         'captured_location_pairs': int(arrays[sid].sum()),
                         'finite_bootstrap_replicates': int(np.isfinite(bs).sum())})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / 'strategy_coverage.csv', index=False)
    contrast = []
    attribution = []
    factors = ['lower_shared_six_cutoffs', 'retain_25_analytes', 'include_next_sample']
    metric_names = ['binary_coverage', 'location_equal_pair_coverage', 'pooled_pair_coverage']
    for w in points:
        for metric in metric_names:
            contribution = np.zeros(3)
            boot_contribution = np.zeros((REPS, 3))
            for s in STRATEGIES:
                for j in range(3):
                    if s[j]:
                        continue
                    up = list(s); up[j] = 1
                    lowkey, highkey = (w, SID[s], metric), (w, SID[tuple(up)], metric)
                    bs = boot_values[highkey] - boot_values[lowkey]
                    lo, hi = np.nanquantile(bs, [.025, .975])
                    contrast.append({'weighting': w, 'metric': metric, 'factor': factors[j],
                                     'from_strategy': SID[s], 'to_strategy': SID[tuple(up)],
                                     'difference_pp': 100*(values[highkey]-values[lowkey]),
                                     'ci95_lower_pp': 100*lo, 'ci95_upper_pp': 100*hi})
            for order in permutations(range(3)):
                current = [0, 0, 0]
                for j in order:
                    before = SID[tuple(current)]
                    current[j] = 1
                    after = SID[tuple(current)]
                    contribution[j] += (values[(w, after, metric)]-values[(w, before, metric)])/6
                    boot_contribution[:, j] += (boot_values[(w, after, metric)]-boot_values[(w, before, metric)])/6
            assert abs(contribution.sum() - (values[(w, '111', metric)]-values[(w, '000', metric)])) < 1e-12
            assert np.allclose(boot_contribution.sum(axis=1), boot_values[(w, '111', metric)]-boot_values[(w, '000', metric)], atol=1e-12)
            for j in range(3):
                lo, hi = np.nanquantile(boot_contribution[:, j], [.025, .975])
                attribution.append({'weighting': w, 'metric': metric, 'factor': factors[j],
                                    'order_averaged_gain_pp': 100*contribution[j],
                                    'ci95_lower_pp': 100*lo, 'ci95_upper_pp': 100*hi})
    pd.DataFrame(contrast).to_csv(OUT / 'paired_single_change_contrasts.csv', index=False)
    pd.DataFrame(attribution).to_csv(OUT / 'order_averaged_attribution.csv', index=False)
    pair_intervals = []
    for name, (num, den) in pair_specs.items():
        for w, p in points.items():
            bs = divide(draws[w], num, den)
            lo, hi = np.nanquantile(bs, [.025, .975])
            pair_intervals.append({'pair_or_group': name, 'weighting': w,
                                   'repeat_report_fraction': float(divide(p, num, den)),
                                   'ci95_lower': lo, 'ci95_upper': hi,
                                   'first_location_pairs': int(f[:, den].sum()),
                                   'both_location_pairs': int(f[:, num].sum())})
    pair_table['selected_top12_by_first_support'] = pair_table.pair.isin(top)
    pair_table.to_csv(OUT / 'all_300_pair_transitions.csv', index=False)
    pd.DataFrame(pair_intervals).to_csv(OUT / 'top_pair_and_group_recurrence.csv', index=False)
    sensitivity = []
    masks = {'all': np.ones(len(d), bool), 'matched_cross_cycle': d.matched.to_numpy(bool),
             'unchanged_water_type': d.water_type.eq(d.water_type_next).to_numpy(),
             'lag_90_to_270_days': d.lag_days.between(90, 270).to_numpy()}
    masks.update({'water_type_'+str(k): d.water_type.eq(k).to_numpy() for k in sorted(d.water_type.unique())})
    for group, mask in masks.items():
        if not mask.any():
            continue
        point = f[mask].mean(axis=0)
        for (sid, metric), (num, den) in metrics.items():
            sensitivity.append({'subset': group, 'strategy': sid, 'metric': metric,
                                'locations': int(mask.sum()), 'reference_positive_locations': int(den_pos[mask].sum()),
                                'estimate': float(divide(point, num, den)),
                                'inference': 'descriptive location-equal subset estimate'})
    pd.DataFrame(sensitivity).to_csv(OUT / 'subset_coverage.csv', index=False)
    buffer1 = pair_flags(visible(c1, 1, 1, 2.)); buffer2 = pair_flags(visible(c2, 1, 1, 2.))
    buffered = []
    for j in range(300):
        n = int(buffer1[:, j].sum()); b = int((buffer1[:, j] & buffer2[:, j]).sum())
        buffered.append({'pair': PAIR_NAMES[j], 'first': n, 'next': int(buffer2[:, j].sum()),
                         'both': b, 'repeat_report_fraction': b/n if n else np.nan,
                         'cutoff': 'twice_native_limit_for_each_analyte'})
    pd.DataFrame(buffered).to_csv(OUT / 'pair_recurrence_twice_native_limits.csv', index=False)
    buffref = buffer1 | buffer2
    buffn = buffref.sum(axis=1)
    buffrows = []
    for s in STRATEGIES:
        x = strategy_pairs(c1, c2, s, 2.).sum(axis=1)
        m = buffn > 0
        buffrows.append({'strategy': SID[s], 'reference_positive_locations': int(m.sum()),
                         'binary_coverage': float(np.mean(x[m] > 0)),
                         'location_equal_pair_coverage': float(np.mean(x[m]/buffn[m])),
                         'pooled_pair_coverage': float(x.sum()/buffn.sum())})
    pd.DataFrame(buffrows).to_csv(OUT / 'coverage_twice_native_limits.csv', index=False)
    d.groupby(['first_k25', 'next_k25']).size().rename('locations').reset_index().to_csv(OUT / 'paired_count_transitions.csv', index=False)
    simple = d[['location_id', 'pws_id', 'state', 'water_type', 'water_type_next', 'lag_days', 'matched', 'first_k25', 'next_k25']].copy()
    simple['reference_pairs'] = ref_count
    for sid, ar in arrays.items():
        simple[sid+'__retained_pairs'] = ar.sum(axis=1)
    simple.to_csv(OUT / 'location_level_coverage.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    print('Independently enumerating all sample-level pair sets', flush=True)
    enumeration = verify_sets(c1, c2, arrays)
    for sid, observed in enumeration.items():
        assert observed[0] == int(arrays[sid].any(axis=1).sum())
        assert observed[1] == int(arrays[sid].sum())
        num, _ = metrics[(sid, 'location_equal_pair_coverage')]
        assert abs(observed[2] - f[:, num].sum()) < 1e-8
    # Cross-check all transition columns without paired matrix operations.
    assert pair_table['both'].sum() == int((p1 & p2).sum())
    assert pair_table.either.sum() == int(ref_count.sum())
    verify_lock()
    audit = {'status': 'PASS', 'cohort_locations': len(d), 'reference_positive_locations': int(den_pos.sum()),
             'reference_location_pairs': int(ref_count.sum()), 'all_8_strategies_nested_in_reference': True,
             'all_factor_upgrades_non_decreasing': True, 'all_locations_independently_set_replayed': len(d),
             'shapley_additivity_point_and_all_bootstraps': True, 'synthetic_no_cross_date_pairs_fixture': 'PASS',
             'bootstrap_replicates': REPS, 'bootstrap_seed': SEED, 'script_sha256': sha(Path(__file__)),
             'no_model_fitting': True, 'no_external_confirmation': True,
             'hybrid_threshold_definition': 'UCMR3 cutoffs only exist for the shared six; extra 19 retain native limits',
             'limitations': ['Retrospective thinning of already measured paired records, not a sampling-schedule trial.',
                            'Reference is two observed Method 533 panels, not complete true PFAS contamination.',
                            'No equal-cost comparison or uninterrupted persistence claim.']}
    dump(OUT / 'RESULT_AUDIT.json', audit)
    dump(OUT / 'RESULT_MANIFEST.json', {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*'))
                                       if p.is_file() and p.name != 'RESULT_MANIFEST.json'})
    print(result.loc[(result.weighting == 'location_equal') & result.metric.ne('pooled_pair_coverage'), ['strategy', 'metric', 'estimate', 'ci95_lower', 'ci95_upper']].to_string(index=False), flush=True)
    print(pd.DataFrame(pair_intervals).loc[lambda a: a.weighting.eq('location_equal')].to_string(index=False), flush=True)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    analyze()
