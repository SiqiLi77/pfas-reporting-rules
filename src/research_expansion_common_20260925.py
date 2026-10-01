"""Shared provenance and resampling for the locked V1.26 extension."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/research_expansion_20260925'
PAIRED = ROOT / 'outputs/observation_design_20260925/cohort/paired_panels.csv.gz'
FIRST = ROOT / 'outputs/significance_extension_20260924/panel_coverage/strict25_event_records.csv.gz'
PREDS = ROOT / 'outputs/monitoring_model_strength_20260925/blocked_predictions.csv.gz'
CALENDAR = ROOT / 'outputs/monitoring_model_strength_20260925/calendar_predictions.csv.gz'
IDENTITY = ROOT / 'outputs/observation_design_20260925/identity_probe'
REPS = 2000
SEED = 202609253


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, value):
    Path(p).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def freeze():
    lock = OUT / 'ANALYSIS_LOCK.json'
    if lock.exists():
        raise FileExistsError('Do not overwrite an analysis lock')
    inputs = [OUT/'ANALYSIS_PROTOCOL.md', PAIRED, FIRST, PREDS, CALENDAR,
              ROOT/'src/prepare_monitoring_followup_20260925.py',
              ROOT/'src/run_conditional_identity_probe_20260925.py',
              ROOT/'src/prepare_observation_design_20260925.py']
    inputs += sorted(ROOT.glob('src/*expansion*20260925.py'))
    inputs += sorted((IDENTITY/'models').glob('*shared_persistence*'))
    dump(lock, {'locked_at_utc': datetime.now(timezone.utc).isoformat(),
                'exploratory_after_v1_26': True, 'reps': REPS, 'seed': SEED,
                'sources': {str(p.relative_to(ROOT)): sha(p) for p in inputs}})


def verify():
    for p, h in json.loads((OUT/'ANALYSIS_LOCK.json').read_text())['sources'].items():
        assert sha(ROOT/p) == h, p


def read_paired():
    return pd.read_csv(PAIRED, dtype={'location_id': str, 'state': str, 'pws_id': str})


def cluster_weights(d, reps=REPS, seed=SEED):
    keys = d.state.astype(str) + '|' + d.pws_id.astype(str)
    unique, index = np.unique(keys, return_inverse=True)
    states = np.array([k.split('|')[0] for k in unique])
    strata = [np.flatnonzero(states == s) for s in np.unique(states)]
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        weight = np.zeros(len(unique))
        for ix in strata:
            weight[ix] = rng.multinomial(len(ix), np.full(len(ix), 1/len(ix)))
        yield weight[index]


def ci(values):
    values = np.asarray(values)
    valid = values[np.isfinite(values)]
    if not len(valid):
        return {'ci95_lower': None, 'ci95_upper': None, 'finite_draws': 0}
    lo, hi = np.quantile(valid, [.025, .975])
    return {'ci95_lower': float(lo), 'ci95_upper': float(hi), 'finite_draws': len(valid)}


if __name__ == '__main__':
    freeze()
