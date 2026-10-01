#!/usr/bin/env python3
"""Exact conditional Bernoulli identity probe with three fixed model specifications."""
from __future__ import annotations
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from prepare_observation_design_20260925 import ROOT, OUT as PARENT, COHORT, A25, dump, sha
from analyze_observation_design_20260925 import bootstrap, divide

OUT = PARENT / 'identity_probe'
PROTOCOL = PARENT / 'IDENTITY_PROBE_PROTOCOL.md'
DATA = COHORT / 'paired_panels.csv.gz'
MODELS = ['site_count', 'shared_persistence', 'full_identity_context']
PENALTIES = [.01, .1]


def log_partition_and_inclusions(logits, k):
    n, m = logits.shape
    maxk = int(k.max())
    prefix = np.full((n, m+1, maxk+1), -np.inf)
    suffix = np.full_like(prefix, -np.inf)
    prefix[:, 0, 0] = 0.; suffix[:, m, 0] = 0.
    for j in range(m):
        prefix[:, j+1, 0] = 0.
        upper = min(j+1, maxk)
        prefix[:, j+1, 1:upper+1] = np.logaddexp(prefix[:, j, 1:upper+1],
            logits[:, j, None] + prefix[:, j, :upper])
    for j in range(m-1, -1, -1):
        suffix[:, j, 0] = 0.
        upper = min(m-j, maxk)
        suffix[:, j, 1:upper+1] = np.logaddexp(suffix[:, j+1, 1:upper+1],
            logits[:, j, None] + suffix[:, j+1, :upper])
    rows = np.arange(n)
    z = prefix[rows, m, k]
    marg = np.zeros_like(logits)
    if maxk:
        q = np.arange(maxk)
        right = k[:, None]-1-q[None, :]
        valid = (right >= 0) & (right <= maxk)
        for j in range(m):
            terms = prefix[:, j, :maxk] + suffix[rows[:, None], j+1, np.clip(right, 0, maxk)]
            terms = np.where(valid, terms, -np.inf)
            excl = np.logaddexp.reduce(terms, axis=1)
            marg[:, j] = np.exp(logits[:, j]+excl-z)
    assert np.isfinite(z).all() and np.isfinite(marg).all()
    assert np.max(np.abs(marg.sum(axis=1)-k)) < 1e-8
    return z, marg


def lbfgs(fun, start, maximum=180):
    """Deterministic limited-memory BFGS with Armijo backtracking, memory ten."""
    x = start.copy()
    f, g = fun(x)
    history = []
    for iteration in range(maximum):
        if np.max(np.abs(g)) < 1e-6:
            return x, {'converged': True, 'iterations': iteration, 'objective': f, 'max_gradient': float(np.max(np.abs(g)))}
        q = g.copy(); alphas = []
        for s, y, rho in reversed(history):
            a = rho * np.dot(s, q); alphas.append(a); q -= a*y
        gamma = np.dot(history[-1][0], history[-1][1])/np.dot(history[-1][1], history[-1][1]) if history else 1.
        direction = gamma*q
        for (s, y, rho), a in zip(history, reversed(alphas)):
            direction += s*(a-rho*np.dot(y, direction))
        direction = -direction
        slope = np.dot(g, direction)
        if slope >= 0:
            direction = -g; slope = -np.dot(g, g); history = []
        step = 1.
        for _ in range(35):
            xn = x + step*direction
            fn, gn = fun(xn)
            if np.isfinite(fn) and fn <= f + 1e-4*step*slope:
                break
            step *= .5
        else:
            return x, {'converged': False, 'iterations': iteration, 'objective': f, 'max_gradient': float(np.max(np.abs(g))), 'failure': 'line_search'}
        s = xn-x; y = gn-g
        sy = np.dot(s, y)
        if sy > 1e-12:
            history.append((s, y, 1/sy)); history = history[-10:]
        x, f, g = xn, fn, gn
    return x, {'converged': False, 'iterations': maximum, 'objective': f, 'max_gradient': float(np.max(np.abs(g)))}


def raw_features(d, spec=None):
    k = d.first_k25.to_numpy(float)
    month = pd.to_datetime(d.date_first).dt.month.to_numpy(float)
    cols = [np.ones(len(d)), k/25, np.log1p(k), (k==0).astype(float), (k==1).astype(float), (k>=2).astype(float),
            np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12)]
    names = ['intercept', 'count_div25', 'log1p_count', 'zero', 'one', 'multiple', 'month_sin', 'month_cos']
    levels = {}
    for field in ['water_type', 'pws_size']:
        values = d[field].fillna('Unknown').astype(str)
        lev = sorted(values.unique()) if spec is None else spec['levels'][field]
        levels[field] = lev
        for value in lev:
            cols.append(values.eq(value).to_numpy(float)); names.append(field+'='+value)
    return np.column_stack(cols), {'levels': levels, 'names': names}


def input_arrays(d, spec=None):
    x, fitted = raw_features(d, spec)
    if spec is None:
        mean, scale = x.mean(axis=0), x.std(axis=0)
        mean[0] = 0.; scale[0] = 1.; scale[scale < 1e-8] = 1.
        fitted.update(mean=mean.tolist(), scale=scale.tolist())
    else:
        fitted = spec
    x = (x-np.array(fitted['mean']))/np.array(fitted['scale'])
    previous = d[[f'first_report__{a}' for a in A25]].to_numpy(float)
    return x, previous, fitted


def unnormalized(theta, x, previous, model):
    base_size = x.shape[1]*25
    logits = x @ theta[:base_size].reshape(x.shape[1], 25)
    if model == 'shared_persistence':
        logits += theta[base_size]*previous
    elif model == 'full_identity_context':
        logits += previous @ theta[base_size:].reshape(25, 25)
    return logits


def objective(theta, x, previous, y, model, penalty):
    logits = unnormalized(theta, x, previous, model)
    k = y.sum(axis=1).astype(int)
    logz, marg = log_partition_and_inclusions(logits, k)
    error = (marg-y)/len(y)
    parts = [(x.T @ error).ravel()]
    if model == 'shared_persistence':
        parts.append(np.array([(error*previous).sum()]))
    elif model == 'full_identity_context':
        parts.append((previous.T @ error).ravel())
    loss = np.mean(logz-(y*logits).sum(axis=1)) + .5*penalty*np.dot(theta, theta)
    grad = np.concatenate(parts) + penalty*theta
    return float(loss), grad


def fit(d, model, penalty):
    x, previous, spec = input_arrays(d)
    y = d[[f'next_report__{a}' for a in A25]].to_numpy(float)
    active = (y.sum(axis=1)>0) & (y.sum(axis=1)<25)
    extra = 1 if model == 'shared_persistence' else (625 if model == 'full_identity_context' else 0)
    start = np.zeros(x.shape[1]*25 + extra)
    fun = lambda theta: objective(theta, x[active], previous[active], y[active], model, penalty)
    theta, info = lbfgs(fun, start)
    info['retried'] = False
    if not info['converged']:
        first = info
        theta, info = lbfgs(fun, theta, 400)
        info['retried'] = True; info['first_attempt'] = first
    assert info['converged'], (model, penalty, info)
    info.update(training_locations=len(d), training_informative=int(active.sum()), parameters=len(theta))
    return theta, spec, info


def evaluate(theta, spec, d, model):
    x, previous, _ = input_arrays(d, spec)
    y = d[[f'next_report__{a}' for a in A25]].to_numpy(float)
    logits = unnormalized(theta, x, previous, model)
    out = np.zeros(len(d))
    for start in range(0, len(d), 1000):
        end = min(start+1000, len(d))
        z, _ = log_partition_and_inclusions(logits[start:end], y[start:end].sum(axis=1).astype(int))
        out[start:end] = z-(y[start:end]*logits[start:end]).sum(axis=1)
    assert np.isfinite(out).all() and out.min() >= -1e-8
    return np.maximum(out, 0.), logits


def numerical_tests():
    rng = np.random.default_rng(20260925)
    for m in [3, 6]:
        for k in range(m+1):
            logits = rng.normal(0, 3, (1, m))
            z, probs = log_partition_and_inclusions(logits, np.array([k]))
            masks = []
            for comb in combinations(range(m), k):
                row = np.zeros(m); row[list(comb)] = 1; masks.append(row)
            masks = np.array(masks)
            score = masks @ logits[0]
            zz = np.logaddexp.reduce(score)
            weights = np.exp(score-zz)
            assert abs(z[0]-zz) < 1e-11
            assert np.max(np.abs(probs[0]-weights @ masks)) < 1e-11
    x = np.column_stack([np.ones(4), rng.normal(size=(4, 2))])
    previous = rng.integers(0, 2, (4, 25)).astype(float)
    y = np.zeros((4, 25)); y[0, [1, 4]] = 1; y[1, [2]] = 1; y[2, [3, 5, 9]] = 1; y[3, [0, 7]] = 1
    maxerr = 0.
    for model in MODELS:
        size = 75+(1 if model == 'shared_persistence' else (625 if model == 'full_identity_context' else 0))
        theta = rng.normal(0, .1, size)
        value, grad = objective(theta, x, previous, y, model, .01)
        for j in rng.choice(size, min(size, 35), replace=False):
            step = np.zeros(size); step[j] = 1e-5
            numeric = (objective(theta+step, x, previous, y, model, .01)[0]-objective(theta-step, x, previous, y, model, .01)[0])/2e-5
            maxerr = max(maxerr, abs(numeric-grad[j]))
    assert maxerr < 1e-7, maxerr
    diag = np.arange(1., 11.)
    optimum, info = lbfgs(lambda v: (float(.5*np.dot(diag*v, v)), diag*v), np.ones(10))
    assert info['converged'] and np.max(np.abs(optimum)) < 1e-6
    return {'exact_small_panel_enumeration': 'PASS', 'max_finite_difference_gradient_error': maxerr,
            'lbfgs_quadratic_test': 'PASS'}


def freeze():
    OUT.mkdir(exist_ok=True)
    path = OUT / 'PREFIT_LOCK.json'
    if path.exists():
        raise FileExistsError('Never overwrite prefit lock')
    tests = numerical_tests()
    sources = [DATA, PROTOCOL, Path(__file__), ROOT/'src/prepare_observation_design_20260925.py',
               ROOT/'src/analyze_observation_design_20260925.py']
    dump(path, {'locked_at_utc': datetime.now(timezone.utc).isoformat(),
                'exploratory_after_observation_results': True, 'tests': tests,
                'sources': {str(p.relative_to(ROOT)): sha(p) for p in sources},
                'optimizer': 'NumPy L-BFGS with Armijo line search; no unavailable dependencies required',
                'models': MODELS, 'penalties': PENALTIES})
    print(json.dumps(tests, indent=2), flush=True)


def verify():
    for p, h in json.loads((OUT/'PREFIT_LOCK.json').read_text())['sources'].items():
        assert sha(ROOT/p) == h, p


def run():
    verify()
    d = pd.read_csv(DATA, dtype={'location_id': str, 'pws_id': str, 'state': str, 'water_type': str, 'pws_size': str})
    assert sorted(d.state_fold.unique()) == list(range(5))
    active = d.next_k25.between(1, 24).to_numpy()
    losses = {m: np.full(len(d), np.nan) for m in MODELS}
    tuning, audits = [], []
    modeldir = OUT / 'models'; modeldir.mkdir(exist_ok=True)
    for fold in range(5):
        testmask = d.state_fold.eq(fold).to_numpy()
        valmask = d.state_fold.eq((fold+1)%5).to_numpy()
        trainmask = ~(testmask | valmask)
        train, val, test = d.loc[trainmask], d.loc[valmask], d.loc[testmask]
        assert not set(train.pws_id) & set(val.pws_id)
        assert not set(train.pws_id) & set(test.pws_id)
        assert not set(val.pws_id) & set(test.pws_id)
        assert not set(train.state) & set(test.state)
        for model in MODELS:
            start = time.monotonic()
            trials = []
            for penalty in PENALTIES:
                theta, spec, info = fit(train, model, penalty)
                ll, _ = evaluate(theta, spec, val, model)
                score = float(ll[val.next_k25.between(1, 24)].mean())
                trials.append((score, penalty))
                tuning.append({'fold': fold, 'model': model, 'penalty': penalty,
                               'validation_conditional_identity_nll': score, **info})
                print(f'Fold {fold} {model} penalty={penalty:g}: validation identity NLL {score:.6f}', flush=True)
            penalty = min(trials)[1]
            fitdata = d.loc[~testmask]
            theta, spec, info = fit(fitdata, model, penalty)
            ll, logits = evaluate(theta, spec, test, model)
            losses[model][testmask] = ll
            prefix = modeldir / f'fold{fold}_{model}'
            np.savez_compressed(str(prefix)+'.npz', theta=theta)
            dump(Path(str(prefix)+'.json'), {'model': model, 'fold': fold, 'penalty': penalty, 'feature_spec': spec, 'fit': info})
            # Exact saved-model replay, including a future-field replacement audit.
            saved = np.load(str(prefix)+'.npz')['theta']
            metadata = json.loads(Path(str(prefix)+'.json').read_text())
            replay, _ = evaluate(saved, metadata['feature_spec'], test, model)
            assert np.array_equal(replay, ll)
            replaced = test.copy()
            for col in replaced.columns:
                if col.startswith('next_') or col in ['date_next', 'lag_days']:
                    replaced[col] = 0
            rx, rp, _ = input_arrays(replaced, spec)
            assert np.array_equal(unnormalized(saved, rx, rp, model), logits)
            audits.append({'fold': fold, 'model': model, 'penalty': penalty,
                           'test_locations': len(test), 'test_informative': int(test.next_k25.between(1, 24).sum()),
                           'test_identity_nll': float(ll[test.next_k25.between(1, 24)].mean()),
                           'saved_prediction_replay': True, 'future_feature_replacement_invariance': True,
                           'seconds': time.monotonic()-start, **info})
            pd.DataFrame(tuning).to_csv(OUT/'tuning.csv', index=False)
            pd.DataFrame(audits).to_csv(OUT/'fit_and_fold_audit.csv', index=False)
            print(f'Fold {fold} {model} complete in {time.monotonic()-start:.1f}s', flush=True)
    assert all(np.isfinite(v).all() for v in losses.values())
    f = np.column_stack([active.astype(float)] + [losses[m] for m in MODELS])
    points, draws = bootstrap(d, f)
    metrics, contrasts = [], []
    for w, p in points.items():
        vals, bs = {}, {}
        for j, model in enumerate(MODELS, 1):
            vals[model] = float(divide(p, j, 0)); bs[model] = divide(draws[w], j, 0)
            lo, hi = np.nanquantile(bs[model], [.025, .975])
            metrics.append({'weighting': w, 'model': model, 'conditional_identity_nll': vals[model],
                            'ci95_lower': lo, 'ci95_upper': hi, 'locations': len(d),
                            'informative_next_reports': int(active.sum())})
        for base, candidate, role in [('site_count', 'shared_persistence', 'information'),
                                      ('shared_persistence', 'full_identity_context', 'primary_algorithm'),
                                      ('site_count', 'full_identity_context', 'combined')]:
            delta = bs[base]-bs[candidate]
            lo, hi = np.nanquantile(delta, [.025, .975])
            contrasts.append({'weighting': w, 'baseline': base, 'candidate': candidate, 'role': role,
                              'identity_nll_reduction': vals[base]-vals[candidate],
                              'ci95_lower': lo, 'ci95_upper': hi,
                              'relative_reduction': (vals[base]-vals[candidate])/vals[base]})
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'paired_contrasts.csv', index=False)
    out = d[['location_id', 'pws_id', 'state', 'state_fold', 'first_k25', 'next_k25', 'water_type', 'lag_days']].copy()
    same = np.array_equal(np.array([0]), np.array([0]))  # replaced by rowwise mask below
    same = np.all(d[[f'first_report__{a}' for a in A25]].to_numpy() == d[[f'next_report__{a}' for a in A25]].to_numpy(), axis=1)
    out['same_report_set'] = same
    for m in MODELS:
        out['identity_nll__'+m] = losses[m]
    out.to_csv(OUT/'heldout_losses.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    subgroup = []
    groups = {'initial_zero': d.first_k25.eq(0).to_numpy(), 'initial_one': d.first_k25.eq(1).to_numpy(),
              'initial_multiple': d.first_k25.ge(2).to_numpy(), 'same_report_set': same, 'changed_report_set': ~same}
    for name, mask in groups.items():
        mask &= active
        for m in MODELS:
            subgroup.append({'group': name, 'model': m, 'informative_locations': int(mask.sum()),
                             'mean_identity_nll': float(losses[m][mask].mean())})
    pd.DataFrame(subgroup).to_csv(OUT/'subgroup_scores.csv', index=False)
    verify()
    dump(OUT/'RESULT_AUDIT.json', {'status': 'PASS', 'locations': len(d), 'informative_next_reports': int(active.sum()),
                                  'candidate_fits': 30, 'refits': 15, 'all_optimizers_converged': True,
                                  'all_saved_model_replays_exact': True, 'future_feature_replacement_invariance': True,
                                  'bootstrap_replicates': 2000, 'conditional_count_not_predictor': True,
                                  'interpretation': 'Conditional identity diagnostic; no complete forecast or independent confirmation claim.'})
    dump(OUT/'RESULT_MANIFEST.json', {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'RESULT_MANIFEST.json'})
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)
    print(pd.DataFrame(contrasts).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['freeze', 'run'])
    args = parser.parse_args()
    freeze() if args.mode == 'freeze' else run()
