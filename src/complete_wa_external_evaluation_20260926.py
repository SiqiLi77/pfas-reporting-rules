"""R3 post-evaluation extension; no Washington fitting or model selection."""
from __future__ import annotations
import hashlib, json, math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from analyze_ucmr5_joint_information_decomposition import score_distribution, METRICS, NLL
from analyze_ucmr5_count_baseline_20260924 import project, interpolate, maximum_upset_gain

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/wa_external_completion_20260926_r3'
R2 = ROOT/'outputs/wa_external_scoring_20260926_r2'
HIST = ROOT/'outputs/joint_information_decomposition_20260916'
COHERENT = ROOT/'outputs/joint_information_validation_20260916/coherent_baselines'
COUNT = ROOT/'outputs/significance_extension_20260924/count_baseline'
CHEMS = ('PFBS','PFHpA','PFHxS','PFNA','PFOA','PFOS')
ALPHAS = np.array([.05,.1,.15,.2,.3,.35,.4,.45,.55,.6,.65,.7,.8,.85,.9,.95])
BITS = ((np.arange(64)[:,None] >> np.arange(6)) & 1)
K = BITS.sum(1)
STRUCT = ('conditional_shared_shock','conditional_gaussian_factor2')
CP = 'count_only_log_3anchor'
EMP = 'training_composition'
PROXY = 'current_metadata_proxy'
SCENARIOS = ['S_GW','S_SW','S_GU','S_MX','L_GW','L_SW','L_GU','L_MX']
REPS, SEED = 2000, 20260926

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def dump(p,obj): p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')

def cluster_intervals(meta, vectors, seed=SEED):
    """R2 event-equal ratio bootstrap; whole PWS and all queries stay paired."""
    keys=list(vectors)
    a=np.column_stack([vectors[k] for k in keys])
    ids,inv=np.unique(meta.pws_id.to_numpy(),return_inverse=True)
    ns=np.bincount(inv)
    sums=np.zeros((len(ids),len(keys))); np.add.at(sums,inv,a)
    weights=np.random.default_rng(seed).multinomial(len(ids),np.full(len(ids),1/len(ids)),size=REPS)
    draws=(weights@sums)/(weights@ns)[:,None]
    lo,hi=np.quantile(draws,[.025,.975],axis=0)
    return pd.DataFrame(dict(key=keys,estimate=a.mean(0),lower95=lo,upper95=hi)),draws

def get_tables():
    cp=pd.read_csv(COUNT/'training_count_priors.csv')
    cp=cp[cp.outer_fold.eq(4)]
    priors=[]
    for alpha in [0,.5,1]:
        pk=cp[cp.alpha.eq(alpha)].sort_values('count').smoothed_count_probability.to_numpy()
        assert len(pk)==7
        priors.append(np.array([pk[k]/math.comb(6,int(k)) for k in K]))
    tables={}
    e=pd.read_csv(HIST/'baseline/training_composition_tables.csv.gz')
    e=e[e.outer_fold.eq(4)]
    tables[EMP]=np.stack([e[e.alpha.eq(a)].sort_values(['old_visible_code','state_code']).raw_probability.to_numpy().reshape(64,64) for a in ALPHAS])
    c=pd.read_csv(COHERENT/'training_composition_tables.csv.gz')
    c=c[c.outer_fold.eq(4)]
    for w in ['location_equal','jurisdiction_pws_equal']:
        for lam in [.1,1,10]:
            path=[]
            for alpha in ALPHAS:
                f=c[c.alpha.eq(alpha)&c.training_weighting.eq(w)].sort_values(['old_visible_code','state_code'])
                counts=f.weighted_training_count.to_numpy().reshape(64,64)
                prior=f.prior_probability.to_numpy().reshape(64,64)
                path.append((counts+lam*prior)/(counts.sum(1)[:,None]+lam))
            tables[f'coherent_composition_{w}_lambda{lam:g}']=np.array(path)
    for q in tables.values():
        np.testing.assert_allclose(q.sum(2),1,atol=3e-14,rtol=0)
        assert q.min()>=0
    return priors,tables

def replay(priors,tables,meta):
    audit={}
    cohort=pd.read_csv(ROOT/'work/ucmr5_final_20260828_r1/ucmr5_artificial_censor_strict.csv.gz',dtype={'pws_id':str})
    train=cohort[~cohort.state_fold.isin([4,1])]
    assert len(train)==15876 and set(train.state_fold)=={0,2,3}
    assert not set(train.pws_id)&set(meta.pws_id)
    assert not train.state.eq('WA').any()
    audit.update(training_locations=len(train),training_folds=[0,2,3],validation_fold=1,outer_fold=4,WA_training_system_overlap=0)
    # Recount only the UCMR training anchors to audit archived prior provenance.
    from analyze_ucmr5_count_baseline_20260924 import fit_prior
    audit['count_prior_training_replay_error']=float(max(np.abs(fit_prior(train,a)[0]-p).max() for a,p in zip([0,.5,1],priors)))
    assert audit['count_prior_training_replay_error']<1e-14
    old=pd.read_csv(HIST/'payload/metadata_fold_4.csv.gz')
    z=np.load(COUNT/'count_probabilities_fold_4.npz')
    errors=[]
    for a in ALPHAS:
        ix=np.flatnonzero(old.alpha.eq(a))
        # All distinct exact boundary masks, plus evenly spaced original rows.
        ix=np.unique(np.r_[ix[np.linspace(0,len(ix)-1,25,dtype=int)],old.index[old.alpha.eq(a)].to_numpy()[np.unique(old.loc[ix,'old_visible_code'],return_index=True)[1]]])
        q,_=project(z['frozen_margins'][ix],interpolate(a,priors))
        errors.append(float(np.abs(q-z['raw_probability'][ix]).max()))
    audit['count_probability_replay_error']=max(errors)
    assert max(errors)<1e-10
    em=pd.read_csv(HIST/'baseline/metadata.csv.gz')
    ep=np.load(HIST/'baseline/probabilities.npz')['raw_probability']
    ix=np.flatnonzero(em.outer_fold.eq(4)); e4=em.iloc[ix]
    aq=np.searchsorted(ALPHAS,e4.alpha.to_numpy())
    audit['empirical_probability_replay_error']=float(np.abs(tables[EMP][aq,e4.old_visible_code.to_numpy(int)]-ep[ix]).max())
    assert audit['empirical_probability_replay_error']<1e-12
    fields=['outer_fold','alpha','old_visible_code','observed_state_code']+[f'{m}__{s}' for m in tables if m!=EMP for s in NLL]
    cm=pd.read_csv(COHERENT/'record_scores.csv.gz',usecols=fields)
    cm=cm[cm.outer_fold.eq(4)]
    aq=np.searchsorted(ALPHAS,cm.alpha.to_numpy())
    errors=[]
    for m,table in tables.items():
        if m==EMP: continue
        q=table[aq,cm.old_visible_code.to_numpy(int)]
        loss,_,_=score_distribution(q,cm.observed_state_code.to_numpy(int))
        errors += [float(np.abs(loss[s]-cm[f'{m}__{s}']).max()) for s in NLL]
    audit['coherent_score_replay_error']=max(errors)
    assert max(errors)<1e-8
    audit['status']='PASS'
    dump(OUT/'TRAINING_AND_REPLAY_AUDIT.json',audit)
    return audit

def path_audit(prob,model,scenario):
    paths=prob.transpose(1,0,2)
    unique,weights=np.unique(paths.reshape(len(paths),-1),axis=0,return_counts=True)
    violations=0;maxgain=0.;example=None
    for path,w in zip(unique.reshape(-1,len(ALPHAS),64),weights):
        for j in range(len(ALPHAS)-1):
            gain,upset=maximum_upset_gain(path[j+1]-path[j])
            maxgain=max(maxgain,gain)
            if gain>1e-8:
                violations+=int(w)
                if example is None:example=dict(alpha_low=float(ALPHAS[j]),alpha_high=float(ALPHAS[j+1]),state_codes=upset,probability_increase=gain)
    return dict(model=model,scenario=scenario,unique_paths=len(unique),violating_event_transitions=violations,
                max_upper_set_increase=maxgain,example=example)

def score_tensor(prob,codes):
    loss,_,_=score_distribution(prob.reshape(-1,64),codes.ravel())
    return np.stack([loss[s].reshape(codes.shape) for s in METRICS],axis=2)

def main():
    OUT.mkdir(exist_ok=True)
    (OUT/'results').mkdir(exist_ok=True)
    sources=[Path(__file__),OUT/'PROTOCOL.md',ROOT/'src/analyze_ucmr5_joint_information_decomposition.py',ROOT/'src/analyze_ucmr5_count_baseline_20260924.py',
        COUNT/'training_count_priors.csv',HIST/'baseline/training_composition_tables.csv.gz',COHERENT/'training_composition_tables.csv.gz',
        R2/'inputs/permitted_inputs.csv',R2/'inputs/evaluation_metadata.csv',ROOT/'outputs/wa_external_validation_20260926/verified_event_labels.csv',
        ROOT/'work/ucmr5_final_20260828_r1/ucmr5_artificial_censor_strict.csv.gz']
    sources += sorted((R2/'results').glob('probabilities_*.npz'))
    lock=dict(locked_at_utc=datetime.now(timezone.utc).isoformat(),post_R2_extension=True,WA_fitting=False,
              sources={str(p.relative_to(ROOT)):sha(p) for p in sources})
    lockfile=OUT/'INPUT_LOCK.json'
    if lockfile.exists():
        previous=json.loads(lockfile.read_text())['sources']
        changed={k:dict(old=previous.get(k),new=v) for k,v in lock['sources'].items() if previous.get(k)!=v}
        if changed:
            amendment=json.loads((OUT/'NUMERICAL_AUDIT_CORRECTION.json').read_text())
            assert changed==amendment['changed_sources'],'Unrecorded change to locked code/data'
    else:dump(lockfile,lock)
    meta=pd.read_csv(R2/'inputs/evaluation_metadata.csv',dtype={'pws_id':str})
    inputs=pd.read_csv(R2/'inputs/permitted_inputs.csv')
    raw=pd.read_csv(ROOT/'outputs/wa_external_validation_20260926/verified_event_labels.csv',dtype={'lab':str})
    assert list(raw.event_id)==list(meta.location_id)==list(inputs.location_id)
    meta['lab']=raw.lab.to_numpy();meta['year']=pd.to_datetime(meta.date).dt.year.astype(str)
    assert not any(c.startswith('y_') or 'quantity' in c for c in inputs)
    masks=inputs[['old_visible__'+c for c in CHEMS]].to_numpy(int)@(2**np.arange(6))
    priors,tables=get_tables()
    print('Archived UCMR replay',flush=True);replay(priors,tables,meta)
    # Baseline predictions are built before reading Washington target concentrations.
    lookup={m:t[:,masks,:] for m,t in tables.items()}
    np.savez_compressed(OUT/'results/transported_composition_probabilities.npz',**lookup)
    predlock={m:hashlib.sha256(q.tobytes()).hexdigest() for m,q in lookup.items()}
    value=raw[['quantity_ngL__'+c for c in CHEMS]].to_numpy(float)
    exact=raw[['range__'+c for c in CHEMS]].eq('EQ').to_numpy()
    lower=np.array([3,3,3,4,4,4.]);upper=np.array([90,10,30,20,20,40.])
    assert np.all(exact|(value<=lower))
    labels=np.stack([exact&(value>=lower*(upper/lower)**a-1e-10) for a in ALPHAS])
    codes=labels@(2**np.arange(6))
    metrics=[];threshold=[];paths=[];projection=[];scores={};audit=[];proxy_scores=None
    for scenario in [*SCENARIOS,PROXY]:
        print('Scoring',scenario,flush=True)
        idx=np.flatnonzero(meta.proxy_eligible.eq(1)) if scenario==PROXY else np.arange(len(meta))
        sm=meta.iloc[idx].reset_index(drop=True);y=codes[:,idx]
        z=np.load(R2/'results'/f'probabilities_{scenario}.npz')
        qs=[]
        for j,a in enumerate(ALPHAS):
            q,pa=project(z['margins'][j+1],interpolate(a,priors));qs.append(q)
            projection.append(dict(scenario=scenario,alpha=a,**pa))
        cq=np.array(qs)
        np.savez_compressed(OUT/'results'/f'count_probabilities_{scenario}.npz',raw_probability=cq)
        probs={'factorized':z['factorized'][1:-1],**{m:z[m][1:-1] for m in STRUCT},CP:cq,
               **{m:q[:,idx] for m,q in lookup.items()}}
        scenario_scores={}
        for m,q in probs.items():
            assert q.shape==(*y.shape,64)
            assert q.min()>=0 and np.max(np.abs(q.sum(2)-1))<1e-10
            compatible=(np.arange(64)[None,:]&masks[idx,None])==masks[idx,None]
            # The original marginal interface clips retained positives to 1-1e-8.
            # Reusing it does not imply strict zero incompatible-state mass.
            incompatible_mass=(q*(~compatible)[None,:,:]).sum(2)
            if m in tables:
                assert incompatible_mass.max()<1e-12
            else:
                union_bound=((1-z['margins'][1:-1])*BITS[masks[idx]][None,:,:]).sum(2)
                assert np.max(incompatible_mass-union_bound)<1e-8
            loss=score_tensor(q,y);means=loss.mean(0);scenario_scores[m]=means
            marginerr=float(np.max(np.abs(q@BITS-z['margins'][1:-1])))
            if m in ['factorized',*STRUCT,CP]:assert marginerr<1e-8
            audit.append(dict(scenario=scenario,model=m,normalization_error=float(np.abs(q.sum(2)-1).max()),max_incompatible_mass=float(incompatible_mass.max()),
                              marginal_difference=marginerr,chain_error=float(np.abs(loss[:,:,0]-loss[:,:,1]-loss[:,:,2]).max())))
            for weighting in ['event_equal','system_equal','earliest_per_system']:
                if weighting=='event_equal':v=means.mean(0)
                elif weighting=='system_equal':v=pd.DataFrame(means).groupby(sm.pws_id.to_numpy()).mean().mean().to_numpy()
                else:v=means[sm.first_per_system.eq(1)].mean(0)
                metrics += [dict(scenario=scenario,model=m,weighting=weighting,metric=s,loss=float(v[k]),events=len(sm),systems=sm.pws_id.nunique()) for k,s in enumerate(METRICS)]
            threshold += [dict(scenario=scenario,model=m,alpha=a,metric=s,loss=float(loss[j,:,k].mean())) for j,a in enumerate(ALPHAS) for k,s in enumerate(METRICS)]
            if m==CP or (scenario==PROXY and m in tables):paths.append(path_audit(q,m,scenario))
            if scenario==PROXY:
                if proxy_scores is None:
                    proxy_scores=pd.concat([sm.assign(alpha=a,observed_state_code=y[j],observed_count=K[y[j]],old_visible_code=masks[idx]) for j,a in enumerate(ALPHAS)],ignore_index=True)
                for k,s in enumerate(METRICS):proxy_scores[f'{m}__{s}']=loss[:,:,k].ravel()
        scores[scenario]=scenario_scores
        z.close()
    pd.DataFrame(metrics).to_csv(OUT/'results/model_metrics.csv',index=False)
    pd.DataFrame(threshold).to_csv(OUT/'results/threshold_metrics.csv',index=False)
    pd.DataFrame(projection).to_csv(OUT/'results/projection_audit.csv',index=False)
    pd.DataFrame(audit).to_csv(OUT/'results/probability_audit.csv',index=False)
    dump(OUT/'results/BOOLEAN_ORDER_AUDIT.json',paths)
    proxy_scores.to_csv(OUT/'results/proxy_record_scores.csv.gz',index=False)
    sm=meta[meta.proxy_eligible.eq(1)].reset_index(drop=True);ps=scores[PROXY]
    primary_baselines=[CP,EMP]+[m for m in tables if m.endswith('_lambda1')]
    pairs=[('factorized',m) for m in [*STRUCT,CP]]+[(r,m) for r in [CP,*tables] for m in STRUCT]
    vectors={};contrast=[]
    for r,m in pairs:
        for k,s in enumerate(METRICS):
            d=ps[r][:,k]-ps[m][:,k]
            contrast.append(dict(reference=r,candidate=m,metric=s,estimate=float(d.mean())))
            if r=='factorized' or r in primary_baselines:vectors[f'{r}|{m}|{s}']=d
    ci,draws=cluster_intervals(sm,vectors)
    ci[['reference','candidate','metric']]=ci.key.str.split('|',expand=True)
    ci.to_csv(OUT/'results/primary_paired_intervals.csv',index=False)
    pd.DataFrame(contrast).to_csv(OUT/'results/all_baseline_contrasts.csv',index=False)
    np.savez_compressed(OUT/'results/paired_bootstrap_draws.npz',draws=draws,keys=np.array(list(vectors)))
    envelopes=[]
    for r,m in [('factorized',x) for x in STRUCT]+[(CP,x) for x in STRUCT]:
        delta=np.array([scores[sc][r]-scores[sc][m] for sc in SCENARIOS])
        for k,s in enumerate(METRICS):
            envelopes.append(dict(reference=r,candidate=m,metric=s,uniform_min=delta[:,:,k].mean(1).min(),uniform_max=delta[:,:,k].mean(1).max(),
                                  eventwise_category_lower=delta[:,:,k].min(0).mean(),eventwise_category_upper=delta[:,:,k].max(0).mean()))
    pd.DataFrame(envelopes).to_csv(OUT/'results/category_envelope.csv',index=False)
    contributions=[]
    for m in STRUCT:
        for k in range(7):
            keep=proxy_scores.observed_count.eq(k)
            for s in NLL:
                d=(proxy_scores[f'factorized__{s}']-proxy_scores[f'{m}__{s}'])[keep]
                contributions.append(dict(model=m,observed_count=k,metric=s,query_records=int(keep.sum()),
                    conditional_mean=float(d.mean()) if len(d) else None,contribution=float(d.sum()/len(proxy_scores))))
    pd.DataFrame(contributions).to_csv(OUT/'results/count_stratum_contributions.csv',index=False)
    print('County/laboratory/year/panel influence checks',flush=True)
    robustness={k:v for k,v in vectors.items() if k.split('|')[0] in ['factorized',CP] and k.split('|')[1] in STRUCT and k.split('|')[2] in NLL}
    group_rows=[];leave_rows=[]
    for field in ['county','lab','year','panels']:
        for name,ix in sm.groupby(field,dropna=False,sort=True).indices.items():
            ix=np.array(ix);sub=sm.iloc[ix];other=np.ones(len(sm),bool);other[ix]=False
            intervals=None
            if sub.pws_id.nunique()>=20:intervals=cluster_intervals(sub,{k:v[ix] for k,v in robustness.items()})[0].set_index('key')
            for key,v in robustness.items():
                r,m,s=key.split('|')
                group_rows.append(dict(grouping=field,group=str(name),events=len(ix),systems=sub.pws_id.nunique(),native_multi_events=int(sub.low_count.ge(2).sum()),
                    reference=r,candidate=m,metric=s,estimate=float(v[ix].mean()),
                    lower95=float(intervals.loc[key,'lower95']) if intervals is not None else None,
                    upper95=float(intervals.loc[key,'upper95']) if intervals is not None else None,
                    contribution=float(v[ix].sum()/len(sm))))
                leave_rows.append(dict(grouping=field,excluded_group=str(name),excluded_events=len(ix),remaining_events=int(other.sum()),
                    reference=r,candidate=m,metric=s,estimate=float(v[other].mean()) if other.any() else None))
    pd.DataFrame(group_rows).to_csv(OUT/'results/subgroup_intervals.csv',index=False)
    pd.DataFrame(leave_rows).to_csv(OUT/'results/leave_one_group_out.csv',index=False)
    influence=[]
    for pws,ix in sm.groupby('pws_id').indices.items():
        for key,v in robustness.items():
            r,m,s=key.split('|');contribution=float(v[ix].sum()/len(sm))
            influence.append(dict(pws_id=pws,events=len(ix),reference=r,candidate=m,metric=s,contribution=contribution,
                                  leave_one_system_out=float((v.sum()-v[ix].sum())/(len(sm)-len(ix)))))
    pd.DataFrame(influence).to_csv(OUT/'results/system_influence.csv',index=False)
    assert predlock=={m:hashlib.sha256(q.tobytes()).hexdigest() for m,q in lookup.items()}
    dump(OUT/'RUN_AUDIT.json',dict(status='PASS',protocol_sha256=sha(OUT/'PROTOCOL.md'),script_sha256=sha(Path(__file__)),
         Washington_training=False,baseline_training_reused=True,events=len(meta),proxy_events=len(sm),proxy_systems=sm.pws_id.nunique(),
         queries=ALPHAS.tolist(),models=list(ps),subgroup_intervals_unadjusted=True,bootstrap_replicates=REPS,
         baseline_predictions_unchanged_after_outcome_access=True,created_at_utc=datetime.now(timezone.utc).isoformat()))
    print('R3 completed',flush=True)

if __name__=='__main__':main()
