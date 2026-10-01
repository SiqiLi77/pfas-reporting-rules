"""Paired out-of-fold scores, cluster intervals, and machine-readable plot data."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from run_auxiliary_information_20260926 import SIX,BITS,COUNTS,scores,dump,sha

METRICS=['pattern_nll','count_nll','conditional_identity_nll','count_ranked_probability_score','binary_brier','marginal_brier']
REPS=5000

def cluster_bootstrap(d,values):
    groups=d.groupby(['state','pws_id'],sort=True).indices
    keys=list(groups);ns=np.array([len(groups[k]) for k in keys],float)
    sums=np.stack([values[groups[k]].sum(0) for k in keys]);means=sums/ns[:,None]
    states=sorted(d.state.unique());si={s:np.array([j for j,k in enumerate(keys) if k[0]==s]) for s in states}
    state_means=np.stack([means[ix].mean(0) for ix in si.values()])
    point={'event_equal':values.mean(0),'jurisdiction_pws_equal':state_means.mean(0)}
    boot={w:np.zeros((REPS,values.shape[1])) for w in point};den=np.zeros(REPS)
    rng=np.random.default_rng(2026092619)
    for ix in si.values():
        for st in range(0,REPS,100):
            en=min(st+100,REPS);m=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=en-st)
            boot['event_equal'][st:en]+=m@sums[ix];den[st:en]+=m@ns[ix]
            boot['jurisdiction_pws_equal'][st:en]+=m@means[ix]/len(ix)/len(states)
    boot['event_equal']/=den[:,None]
    juris=np.random.default_rng(2026092620).multinomial(len(states),np.full(len(states),1/len(states)),size=REPS)
    boot['jurisdiction_resampling']=juris@state_means/len(states)
    point['jurisdiction_resampling']=point['jurisdiction_pws_equal']
    return point,boot

def run(dest):
    result=dest/'results';result.mkdir(exist_ok=True)
    d=pd.read_csv(dest/'cohort.csv.gz',dtype={'location_id':str,'pws_id':str,'state':str,'region':str})
    y=d[['y_native_detect__'+a for a in SIX]].to_numpy(int);old=d[['old_visible__'+a for a in SIX]].to_numpy(int)
    state=y@2**np.arange(6);k=y.sum(1);informative=(k>0)&(k<6)
    losses={};foldrows=[];audit=[];ranking=[];settings=[]
    coverage=np.zeros(len(d),int)
    for f in range(5):
        base=dest/f'fold{f}';done=json.loads((base/'DONE.json').read_text());assert done['models']==45
        z=np.load(base/'predictions.npz');idx=z['index'];coverage[idx]+=1
        assert np.array_equal(idx,np.flatnonzero(d.state_fold.eq(f)))
        rank=pd.read_csv(base/'training_auxiliary_ranking.csv');rank['fold']=f;ranking.append(rank)
        sett=json.loads((base/'SETTINGS.json').read_text())
        for label,a in sett.items():settings.append({'fold':f,'setting':label,'budget':len(a),'auxiliaries':' + '.join(a)})
        for name in z.files:
            if name=='index':continue
            q=z[name];assert q.shape==(len(idx),64)
            assert np.isfinite(q).all() and q.min()>=0
            assert np.max(abs(q.sum(1)-1))<2e-6
            marg=q@BITS;assert np.max(abs(marg[old[idx]==1]-1),initial=0)<2e-6
            val=scores(q,y[idx]);assert np.max(abs(val[:,0]-val[:,1]-val[:,2]))<1e-10
            if name not in losses:losses[name]=np.full((len(d),6),np.nan)
            losses[name][idx]=val
            for j,m in enumerate(METRICS):foldrows.append({'fold':f,'name':name,'metric':m,'estimate':float(val[:,j].mean()),'events':len(idx)})
            audit.append({'fold':f,'name':name,'normalization_error':float(np.max(abs(q.sum(1)-1))),'chain_rule_error':float(np.max(abs(val[:,0]-val[:,1]-val[:,2])))})
    assert np.all(coverage==1) and len(losses)==45 and all(np.isfinite(v).all() for v in losses.values())
    # Random policies are averaged as scores across three prechosen measurement sets,
    # not averaged probabilities (which would incorrectly create a larger ensemble).
    for model in ['independent','linear','nonlinear']:
        for budget in [1,2,3]:losses[f'random_mean{budget}__{model}']=np.mean([losses[f'random{r}_{budget}__{model}'] for r in range(3)],axis=0)
    names=sorted(losses)
    columns=[('informative','denominator')]+[(n,m) for n in names for m in METRICS]
    v=np.column_stack([informative.astype(float)]+[losses[n][:,j] for n in names for j in range(6)])
    point,boot=cluster_bootstrap(d,v)
    lookup={nm:i for i,nm in enumerate(columns)}
    metrics=[];contrasts=[]
    for weight,p in point.items():
        if weight=='jurisdiction_resampling':continue
        for n in names:
            for m in METRICS:
                j=lookup[n,m];lo,hi=np.quantile(boot[weight][:,j],[.025,.975])
                metrics.append({'weighting':weight,'name':n,'metric':m,'estimate':p[j],'ci95_lower':lo,'ci95_upper':hi,'events':len(d)})
            j=lookup[n,'conditional_identity_nll'];s=boot[weight][:,j]/boot[weight][:,0];lo,hi=np.quantile(s,[.025,.975])
            metrics.append({'weighting':weight,'name':n,'metric':'identity_nll_given_informative_count','estimate':p[j]/p[0],'ci95_lower':lo,'ci95_upper':hi,'events':int(informative.sum())})
    pairs=[]
    for model in ['independent','linear','nonlinear']:
        for setting in ['selected1','selected2','selected3','all19','random_mean1','random_mean2','random_mean3','shuffled3']:
            pairs.append((f'baseline__{model}',f'{setting}__{model}','additional_information'))
        pairs.append((f'shuffled3__{model}',f'selected3__{model}','same_event_matching'))
        for b in [1,2,3]:pairs.append((f'random_mean{b}__{model}',f'selected{b}__{model}','selection_rule'))
    for setting in ['baseline','selected1','selected2','selected3','all19']:
        pairs.extend([(f'{setting}__linear',f'{setting}__nonlinear','nonlinearity'),
                      (f'{setting}__independent',f'{setting}__linear','joint_model')])
    for weight,p in point.items():
        for baseline,candidate,role in pairs:
            if weight=='jurisdiction_resampling' and not (role=='nonlinearity' or ('baseline__' in baseline and 'selected3__' in candidate)):continue
            for m in METRICS+['identity_nll_given_informative_count']:
                mm='conditional_identity_nll' if m=='identity_nll_given_informative_count' else m
                i,j=lookup[baseline,mm],lookup[candidate,mm];base=p[i];delta=p[i]-p[j];draw=boot[weight][:,i]-boot[weight][:,j]
                if m=='identity_nll_given_informative_count':base/=p[0];delta/=p[0];draw/=boot[weight][:,0]
                lo,hi=np.quantile(draw,[.025,.975])
                contrasts.append({'weighting':weight,'role':role,'baseline':baseline,'candidate':candidate,'metric':m,'baseline_score':base,
                                  'reduction':delta,'relative_reduction':delta/base if base>0 else None,'ci95_lower':lo,'ci95_upper':hi})
    subgroup=[]
    for group,mask in [('all_old_masked',old.sum(1)==0)]+[(f'count{k0}',k==k0) for k0 in range(7)]+[(f'water_{t}',d.water_type.eq(t).to_numpy()) for t in sorted(d.water_type.unique())]:
        for n in names:
            for j,m in enumerate(METRICS):subgroup.append({'group':group,'name':n,'metric':m,'events':int(mask.sum()),'pws':d.loc[mask,'pws_id'].nunique(),'estimate':float(losses[n][mask,j].mean())})
    md=pd.DataFrame(metrics);cd=pd.DataFrame(contrasts)
    md.to_csv(result/'metrics.csv',index=False);cd.to_csv(result/'paired_contrasts.csv',index=False)
    pd.DataFrame(foldrows).to_csv(result/'fold_scores.csv',index=False)
    pd.DataFrame(subgroup).to_csv(result/'subgroup_scores.csv',index=False)
    pd.concat(ranking).to_csv(result/'selection_rankings.csv',index=False);pd.DataFrame(settings).to_csv(result/'selected_sets.csv',index=False)
    held=d[['location_id','pws_id','state','state_fold','water_type']].copy();held['target_count']=k;held['old_count']=old.sum(1)
    scoreframe=pd.DataFrame({n+'__'+m:a[:,j] for n,a in losses.items() for j,m in enumerate(METRICS)})
    held=pd.concat([held,scoreframe],axis=1)
    held.to_csv(result/'heldout_scores.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    plot=md.loc[md.name.str.match(r'(baseline|selected[123]|all19)__')].copy()
    plot['budget']=plot.name.str.split('__').str[0].map({'baseline':0,'selected1':1,'selected2':2,'selected3':3,'all19':19})
    plot['model']=plot.name.str.split('__').str[1]
    plot.to_csv(result/'origin_information_curve.csv',index=False)
    dump(result/'AUDIT.json',{'status':'PASS','events':len(d),'pws':d.pws_id.nunique(),'jurisdictions':d.state.nunique(),
      'informative_count_events':int(informative.sum()),'scheduled_fitted_models':225,'saved_predictions_checked':len(audit),
      'all_rows_evaluated_once':True,'primary_bootstrap_replicates':REPS,'random_scores_not_probability_ensemble':True,
      'max_chain_rule_error':max(a['chain_rule_error'] for a in audit),'per_model_checks':audit})
    summary=cd.loc[(cd.weighting=='event_equal') & cd.metric.isin(['pattern_nll','identity_nll_given_informative_count']) & ((cd.baseline.str.startswith('baseline') & cd.candidate.str.startswith('selected3'))|((cd.role=='nonlinearity') & cd.baseline.str.startswith('selected3')))]
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();run(a.directory)
