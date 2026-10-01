#!/usr/bin/env python3
"""Two complete same-sample panels, including all initial report-count states."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import zipfile
import numpy as np
import pandas as pd
from prepare_monitoring_followup_20260925 import (
    ARCHIVE, CACHE, REFERENCE, A25, SIX, MRL, parsedate, raw_rows, location,
    row_reasons, sha, fixture_checks,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/observation_design_20260925'
COHORT = OUT / 'cohort'
PROTOCOL = OUT / 'ANALYSIS_PROTOCOL.md'
ANALYSIS = ROOT / 'src/analyze_observation_design_20260925.py'
PREVIOUS = ROOT / 'outputs/monitoring_decision_20260925/cohort/paired_analysis.csv.gz'
OLD = dict(zip(SIX, [.09, .01, .03, .02, .02, .04]))
AI = {a: j for j, a in enumerate(A25)}
META = ('CollectionDate', 'SampleID', 'SampleEventCode', 'PWSID', 'State', 'Size', 'FacilityWaterType')


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


class Panel:
    def __init__(self):
        self.count = np.zeros(25, dtype=np.int16)
        self.conc = np.full(25, np.nan)
        self.flags = np.zeros(25, dtype=bool)
        self.meta = None
        self.errors = set()

    def add(self, r):
        j = AI[r['Contaminant']]
        self.count[j] += 1
        meta = tuple(r[k].strip() for k in META)
        if self.meta is None:
            self.meta = meta
        elif meta != self.meta:
            self.errors.add('inconsistent_date_sample_event_or_metadata')
        errors = row_reasons(r)
        self.errors.update(errors)
        if not errors and r['AnalyticalResultsSign'].strip() == '=':
            self.flags[j] = True
            self.conc[j] = float(r['AnalyticalResultValue'])

    def reasons(self):
        errors = set(self.errors)
        if not np.all(self.count == 1):
            errors.add('missing_or_duplicate_analytes')
        return sorted(errors)


def freeze():
    lock = OUT / 'DESIGN_LOCK.json'
    if lock.exists():
        raise FileExistsError('Refusing to overwrite observation-design lock')
    files = [ARCHIVE, CACHE, REFERENCE, PREVIOUS, PROTOCOL, Path(__file__), ANALYSIS]
    dump(lock, {'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'status': 'Locked before new paired-cohort outcome calculation',
                'exploratory_previously_studied_archive': True,
                'sources': {str(p.relative_to(ROOT)): sha(p) for p in files}})


def verify_lock():
    lock = json.loads((OUT / 'DESIGN_LOCK.json').read_text())
    for name, expected in lock['sources'].items():
        assert sha(ROOT / name) == expected, name


def replay(d, baseline, first_dates, invalid_dates, raw_count):
    """Independent pandas parser checks all raw pairs and earliest available dates."""
    cols = list(dict.fromkeys(['PWSID', 'FacilityID', 'SamplePointID', 'Contaminant',
                              'MethodID', 'MRL', 'Units', 'AnalyticalResultsSign',
                              'AnalyticalResultValue'] + list(META)))
    index = d.set_index('location_id')
    dates = pd.Series({k: pd.Timestamp(v) for k, v in baseline.items()})
    earliest = {}
    bad = set()
    parts = []
    count = 0
    with zipfile.ZipFile(ARCHIVE) as z, z.open('UCMR5_All.txt') as f:
        for c in pd.read_csv(f, sep='\t', encoding='latin-1', dtype=str,
                             keep_default_na=False, usecols=cols, chunksize=200000):
            count += len(c)
            c = c.loc[c.Contaminant.isin(A25) & c.SampleEventCode.isin(['SE1', 'SE2', 'SE3', 'SE4'])].copy()
            c['location_id'] = c.PWSID + '|' + c.FacilityID + '|' + c.SamplePointID
            c = c.loc[c.location_id.isin(dates.index)].copy()
            c['dt'] = pd.to_datetime(c.CollectionDate, format='%m/%d/%Y', errors='coerce')
            later_code = c.SampleEventCode.ne('SE1')
            bad.update(c.loc[later_code & c.dt.isna(), 'location_id'])
            later = c.loc[later_code & c.dt.gt(c.location_id.map(dates))]
            for k, v in later.groupby('location_id').dt.min().items():
                earliest[k] = min(v, earliest.get(k, v))
            x = c.loc[c.location_id.isin(index.index)].copy()
            x['time'] = np.where(x.SampleEventCode.eq('SE1'), 'first', 'next')
            chosen_date = np.where(x.time.eq('first'), x.location_id.map(index.date_first), x.location_id.map(index.date_next))
            x = x.loc[x.dt.eq(pd.to_datetime(chosen_date))]
            parts.append(x)
    assert count == raw_count
    assert bad == set(invalid_dates)
    assert earliest == {k: pd.Timestamp(v) for k, v in first_dates.items()}
    x = pd.concat(parts, ignore_index=True)
    assert len(x) == len(d) * 50
    assert x.MethodID.eq('EPA 533').all() and x.Units.eq('µg/L').all()
    assert x.AnalyticalResultsSign.isin(['=', '<']).all()
    assert not x.duplicated(['location_id', 'time', 'Contaminant']).any()
    assert x.groupby(['location_id', 'time']).size().eq(25).all()
    for t in ['first', 'next']:
        z = x.loc[x.time.eq(t)].copy()
        assert z.groupby('location_id').SampleID.nunique().eq(1).all()
        flags = z.assign(report=z.AnalyticalResultsSign.eq('=')).pivot(index='location_id', columns='Contaminant', values='report').loc[d.location_id, list(A25)].to_numpy(bool)
        expected = d[[f'{t}_report__{a}' for a in A25]].to_numpy(bool)
        assert np.array_equal(flags, expected)
        z['conc'] = pd.to_numeric(z.AnalyticalResultValue, errors='coerce').where(z.AnalyticalResultsSign.eq('='))
        values = z.pivot(index='location_id', columns='Contaminant', values='conc').loc[d.location_id, list(A25)].to_numpy(float)
        assert np.allclose(values, d[[f'{t}_conc_ugL__{a}' for a in A25]], atol=0, rtol=0, equal_nan=True)
        assert np.allclose(pd.to_numeric(z.MRL), z.Contaminant.map(MRL), atol=1e-12, rtol=0)
    return {'status': 'PASS', 'raw_rows': count, 'validated_paired_rows': len(x),
            'all_earliest_dates_replayed': len(earliest), 'all_concentrations_and_flags_exact': True}


def prepare():
    verify_lock()
    fixture_checks()
    COHORT.mkdir(parents=True, exist_ok=True)
    cache = pd.read_csv(CACHE, dtype=str, keep_default_na=False).set_index('location_id')
    ref = pd.read_csv(REFERENCE, dtype=str, keep_default_na=False).set_index('location_id')
    assert len(cache) == 26461 and cache.index.is_unique and ref.index.is_unique
    keys = set(cache.index)
    base = {}
    dates = {k: parsedate(r.CollectionDate) for k, r in ref.loc[ref.index.isin(keys)].iterrows()}
    follow = {}
    first_dates = {}
    invalid = Counter()
    raw_count = 0
    for r in raw_rows():
        raw_count += 1
        if r['Contaminant'] not in AI:
            continue
        k = location(r)
        if k not in keys:
            continue
        event = r['SampleEventCode'].strip()
        if event == 'SE1':
            if k not in base:
                base[k] = Panel()
            base[k].add(r)
        elif event in {'SE2', 'SE3', 'SE4'} and k in dates:
            try:
                dt = parsedate(r['CollectionDate'])
            except ValueError:
                invalid[k] += 1
                continue
            if dt <= dates[k]:
                continue
            if k not in first_dates or dt < first_dates[k]:
                first_dates[k] = dt
                follow[k] = Panel()
            if dt == first_dates[k]:
                follow[k].add(r)
        if raw_count % 1000000 == 0:
            print(f'Read {raw_count:,} archive rows', flush=True)
    availability, paired = [], []
    for k in sorted(keys):
        b = base.get(k)
        reason = b.reasons() if b else ['no_baseline_panel']
        if k not in ref.index:
            reason.append('not_in_strict25_baseline_reference')
        item = {'location_id': k, 'pws_id': cache.loc[k, 'pws_id'], 'state': cache.loc[k, 'state'],
                'water_type': cache.loc[k, 'water_type'], 'pws_size': cache.loc[k, 'pws_size'],
                'baseline_k6': int(cache.loc[k, 'n_native_detect']),
                'baseline_complete25': int(not reason)}
        if not reason:
            rr = ref.loc[k]
            assert b.meta[0] == rr.CollectionDate and b.meta[1] == rr.SampleID
            assert parsedate(b.meta[0]) == dates[k]
            assert np.array_equal(b.flags, rr[[f'report__{a}' for a in A25]].to_numpy(int).astype(bool))
            assert int(b.flags.sum()) == int(rr.k25_native)
            assert sum(bool(b.flags[AI[a]] and b.conc[AI[a]] >= OLD[a]) for a in SIX) == int(rr.k6_old)
            if invalid[k]:
                reason.append('unparseable_followup_date')
            if k not in first_dates:
                reason.append('no_strictly_later_date')
            else:
                reason.extend(follow[k].reasons())
        item['eligible'] = int(not reason)
        item['exclusion'] = ';'.join(sorted(set(reason))) or 'included'
        availability.append(item)
        if reason:
            continue
        f = follow[k]
        row = dict(item)
        row.update(date_first=dates[k].isoformat(), date_next=first_dates[k].isoformat(),
                   lag_days=(first_dates[k]-dates[k]).days, sample_id_first=b.meta[1],
                   sample_id_next=f.meta[1], event_code_next=f.meta[2], water_type_next=f.meta[6],
                   state_fold=int(cache.loc[k, 'state_fold']),
                   matched=rr['matched'].lower() == 'true')
        assert b.meta[3] == f.meta[3] == row['pws_id']
        assert b.meta[4] == f.meta[4] == row['state']
        for t, p in [('first', b), ('next', f)]:
            for a in A25:
                row[f'{t}_report__{a}'] = int(p.flags[AI[a]])
                row[f'{t}_conc_ugL__{a}'] = p.conc[AI[a]]
            row[f'{t}_k25'] = int(p.flags.sum())
            row[f'{t}_k6'] = int(sum(p.flags[AI[a]] for a in SIX))
        paired.append(row)
    d = pd.DataFrame(paired)
    av = pd.DataFrame(availability)
    assert d.location_id.is_unique and d.lag_days.gt(0).all()
    assert d.groupby('pws_id').state_fold.nunique().max() == 1
    assert d.groupby('pws_id').state.nunique().max() == 1
    # Previously inspected restricted cohort is a strict subset with identical labels.
    prior = pd.read_csv(PREVIOUS, dtype={'location_id': str, 'pws_id': str, 'state': str}).set_index('location_id')
    wanted = set(prior.index[prior.reference_se1_complete25.eq(1)])
    subset = d.loc[d.first_k6.le(1)].set_index('location_id')
    assert set(subset.index) == wanted
    prior = prior.loc[subset.index]
    assert np.array_equal(subset.next_k25, prior.future_k25)
    assert np.array_equal(subset.next_k6, prior.future_k6)
    assert np.array_equal(subset.date_next, prior.date_followup)
    for a in A25:
        assert np.array_equal(subset[f'next_report__{a}'], prior[f'future_report__{a}'])
    print(f'Prepared {len(d):,} paired locations; independently replaying archive', flush=True)
    audit = replay(d, dates, first_dates, invalid, raw_count)
    audit.update({'cohort_locations': len(d), 'pws': int(d.pws_id.nunique()), 'jurisdictions': int(d.state.nunique()),
                  'previous_followup_overlap': len(subset), 'overlap_exact': True,
                  'raw_archive_sha256': sha(ARCHIVE), 'script_sha256': sha(Path(__file__)),
                  'lag_min_median_max_days': [int(d.lag_days.min()), float(d.lag_days.median()), int(d.lag_days.max())],
                  'source_water_changes': int(d.water_type.ne(d.water_type_next).sum()),
                  'nondetect_concentrations': 'NaN; never treated as measured zero'})
    d.to_csv(COHORT / 'paired_panels.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    av.to_csv(COHORT / 'availability.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    av.groupby(['baseline_complete25', 'exclusion']).size().rename('locations').reset_index().to_csv(COHORT / 'exclusions.csv', index=False)
    av.groupby(['water_type', 'pws_size']).agg(frame_locations=('location_id', 'size'), paired_locations=('eligible', 'sum')).reset_index().to_csv(COHORT / 'availability_by_source_and_size.csv', index=False)
    dump(COHORT / 'AUDIT.json', audit)
    verify_lock()
    dump(COHORT / 'MANIFEST.json', {p.name: sha(p) for p in sorted(COHORT.iterdir()) if p.is_file() and p.name != 'MANIFEST.json'})
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['freeze', 'prepare'])
    args = parser.parse_args()
    freeze() if args.mode == 'freeze' else prepare()
