"""Post-result robustness: stronger simple count prior, ridge range, calendar split."""
from pathlib import Path
import argparse
import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from research_expansion_common_20260925 import ROOT,OUT,IDENTITY,REPS,verify,dump,sha,read_paired,cluster_weights,ci
import run_expansion_forecast_20260925 as primary
import run_conditional_identity_probe_20260925 as identity

DEST=OUT/'forecast_robustness'


def freeze():
    path=OUT/'ROBUSTNESS_LOCK.json'
    if path.exists():raise FileExistsError(path)
    paths=[OUT/'ROBUSTNESS_PROTOCOL.md',Path(__file__),ROOT/'src/run_expansion_robust_environment_20260925.py',
           OUT/'forecast/tuning.csv',OUT/'forecast/heldout_scores.csv.gz']
    paths+=sorted((OUT/'forecast/models').glob('*'))
    dump(path,{'locked_at_utc':datetime.now(timezone.utc).isoformat(),'first_expansion_results_known':True,
               'sources':{str(p.relative_to(ROOT)):sha(p) for p in paths}})


def check():
    verify()
    for p,h in json.loads((OUT/'ROBUSTNESS_LOCK.json').read_text())['sources'].items():assert sha(ROOT/p)==h,p


def keys(d):
    return [f'{min(int(k),6)}|{w}|{s}' for k,w,s in zip(d.first_k25,d.water_type,d.pws_size)]


def empirical(train,test,strength):
    y=train.next_k25.to_numpy(int)
    global_counts=np.bincount(y,minlength=26)+.5
    prior=global_counts/global_counts.sum()
    cells={}
    for key,value in zip(keys(train),y):
        if key not in cells:cells[key]=np.zeros(26)
        cells[key][value]+=1
    probs={k:(v+strength*prior)/(v.sum()+strength) for k,v in cells.items()}
    out=np.stack([probs.get(k,prior) for k in keys(test)])
    assert out.min()>0 and np.max(abs(out.sum(axis=1)-1))<1e-12
    return out,{'strength':strength,'global':prior.tolist(),'cells':{k:v.tolist() for k,v in probs.items()}}


def countloss(prob,d):
    return float(-np.log(prob[np.arange(len(d)),d.next_k25.to_numpy(int)]).mean())


def saved(kind,fold,test):
    p=OUT/'forecast/models'/f'fold{fold}_{kind}'
    meta=json.loads(Path(str(p)+'.json').read_text());theta=np.load(str(p)+'.npz')['theta']
    return primary.logits_of(theta,meta['feature_spec'],test,kind),theta,meta['feature_spec'],meta['fit']


def save_fit(name,theta,spec,info,kind):
    prefix=DEST/'models'/name
    np.savez_compressed(str(prefix)+'.npz',theta=theta)
    dump(Path(str(prefix)+'.json'),{'kind':kind,'feature_spec':spec,'fit':info})


def fit_calendar_component(train,val,fitdata,test,kind):
    trials=[]
    vy=val[[f'next_report__{a}' for a in primary.A25]].to_numpy(float)
    for penalty in [.00001,.0001,.001,.01]:
        theta,spec,info=primary.fit(train,kind,penalty)
        logit=primary.logits_of(theta,spec,val,kind)
        if kind=='count':score=countloss(primary.softmax(logit),val)
        else:score=float((np.logaddexp(0,logit)-vy*logit).sum(axis=1).mean())
        trials.append({'evaluation':'calendar','kind':kind,'penalty':penalty,'validation_nll':score,**info})
    selected=min(trials,key=lambda x:(x['validation_nll'],-x['penalty']))['penalty']
    theta,spec,info=primary.fit(fitdata,kind,selected)
    save_fit('calendar_'+kind,theta,spec,{**info,'penalty':selected},kind)
    return primary.logits_of(theta,spec,test,kind),trials,info


def run():
    check();DEST.mkdir(exist_ok=True);(DEST/'models').mkdir(exist_ok=True)
    d=read_paired();prior_tune=pd.read_csv(OUT/'forecast/tuning.csv');tunes=[];fits=[];records=[]
    for fold in range(5):
        t=d.state_fold.eq(fold);v=d.state_fold.eq((fold+1)%5)
        train=d.loc[~(t|v)];val=d.loc[v];test=d.loc[t];fitdata=d.loc[~t]
        results={}
        for kind in ['independent_history','count']:
            theta,spec,info=primary.fit(train,kind,.00001)
            logit=primary.logits_of(theta,spec,val,kind)
            yy=val[[f'next_report__{a}' for a in primary.A25]].to_numpy(float)
            score=countloss(primary.softmax(logit),val) if kind=='count' else float((np.logaddexp(0,logit)-yy*logit).sum(axis=1).mean())
            old=prior_tune.loc[prior_tune.fold.eq(fold)&prior_tune.kind.eq(kind)]
            old_score=float(old.validation_nll.min())
            tunes.append({'evaluation':f'fold{fold}','kind':kind,'penalty':.00001,'validation_nll':score,**info})
            if score<old_score:
                theta,spec,info=primary.fit(fitdata,kind,.00001)
                result=primary.logits_of(theta,spec,test,kind);penalty=.00001
            else:
                result,theta,spec,info=saved(kind,fold,test);penalty=float(old.loc[old.validation_nll.idxmin(),'penalty'])
            altered=test.copy()
            for col in altered:
                if col.startswith('next_') or col in ['date_next','lag_days']:altered[col]=98765
            assert np.array_equal(result,primary.logits_of(theta,spec,altered,kind))
            save_fit(f'fold{fold}_{kind}',theta,spec,{**info,'penalty':penalty},kind)
            results[kind]=result
            fits.append({'evaluation':f'fold{fold}','kind':kind,'selected_penalty':penalty,**info})
        empirical_trials=[]
        for strength in [10,50]:
            prob,_=empirical(train,val,strength)
            score=countloss(prob,val)
            empirical_trials.append((score,strength))
            tunes.append({'evaluation':f'fold{fold}','kind':'empirical_count','penalty':strength,'validation_nll':score})
        strength=min(empirical_trials)[1]
        emp,meta=empirical(fitdata,test,strength);dump(DEST/'models'/f'fold{fold}_empirical_count.json',meta)
        p=IDENTITY/'models'/f'fold{fold}_shared_persistence'
        theta=np.load(str(p)+'.npz')['theta'];meta=json.loads(Path(str(p)+'.json').read_text())
        x,prev,_=identity.input_arrays(test,meta['feature_spec'])
        logits=identity.unnormalized(theta,x,prev,'shared_persistence')
        schemes={'independent_history_lower_ridge':(results['independent_history'],primary.count_distribution(primary.sigmoid(results['independent_history'])),'independent'),
                 'softmax_identity_lower_ridge':(logits,primary.softmax(results['count']),'structured'),
                 'empirical_count_identity':(logits,emp,'structured')}
        for model,(ll,pp,kind) in schemes.items():
            ss,new,binary=primary.scores(test,ll,pp,kind)
            rows=test[['location_id','pws_id','state','state_fold','first_k25','next_k25']].copy()
            rows['evaluation']='blocked';rows['model']=model
            for metric,values in ss.items():rows[metric]=values
            rows['p_new_analyte']=new;rows['p_multi']=binary
            records.append(rows)
        print('Robustness fold',fold,'complete',flush=True)
        pd.DataFrame(tunes).to_csv(DEST/'tuning.csv',index=False)
    # Time-separated data must not reuse the full-archive identity model.
    test=d.loc[pd.to_datetime(d.date_first).ge('2025-01-01')].copy()
    before=pd.to_datetime(d.date_next).le('2024-12-31')&~d.pws_id.isin(test.pws_id)
    fitdata=d.loc[before].copy();val=fitdata.loc[fitdata.state_fold.eq(0)];train=fitdata.loc[fitdata.state_fold.ne(0)]
    assert len(train)>100 and len(val)>100 and len(test)>100
    assert not set(fitdata.pws_id)&set(test.pws_id)
    assert pd.to_datetime(fitdata.date_next).max()<pd.to_datetime(test.date_first).min()
    ilogit,itune,iinfo=fit_calendar_component(train,val,fitdata,test,'independent_history')
    clogit,ctune,cinfo=fit_calendar_component(train,val,fitdata,test,'count');tunes+=itune+ctune
    itrials=[]
    for penalty in [.01,.1]:
        theta,spec,info=identity.fit(train,'shared_persistence',penalty)
        ll,_=identity.evaluate(theta,spec,val,'shared_persistence')
        score=float(ll[val.next_k25.between(1,24)].mean());itrials.append((score,penalty))
        tunes.append({'evaluation':'calendar','kind':'conditional_identity','penalty':penalty,'validation_nll':score,**info})
    ipenalty=min(itrials)[1]
    theta,spec,iinfo=identity.fit(fitdata,'shared_persistence',ipenalty)
    x,prev,_=identity.input_arrays(test,spec)
    identity_logits=identity.unnormalized(theta,x,prev,'shared_persistence')
    save_fit('calendar_conditional_identity',theta,spec,{**iinfo,'penalty':ipenalty},'shared_persistence')
    etrials=[]
    for strength in [10,50]:
        prob,_=empirical(train,val,strength);score=countloss(prob,val);etrials.append((score,strength))
        tunes.append({'evaluation':'calendar','kind':'empirical_count','penalty':strength,'validation_nll':score})
    emp,meta=empirical(fitdata,test,min(etrials)[1]);dump(DEST/'models'/'calendar_empirical_count.json',meta)
    schemes={'independent_history_lower_ridge':(ilogit,primary.count_distribution(primary.sigmoid(ilogit)),'independent'),
             'softmax_identity_lower_ridge':(identity_logits,primary.softmax(clogit),'structured'),
             'empirical_count_identity':(identity_logits,emp,'structured')}
    for model,(ll,pp,kind) in schemes.items():
        ss,new,binary=primary.scores(test,ll,pp,kind)
        rows=test[['location_id','pws_id','state','state_fold','first_k25','next_k25']].copy()
        rows['evaluation']='calendar';rows['model']=model
        for metric,values in ss.items():rows[metric]=values
        rows['p_new_analyte']=new;rows['p_multi']=binary
        records.append(rows)
    allrows=pd.concat(records,ignore_index=True)
    allrows.to_csv(DEST/'heldout_scores.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(tunes).to_csv(DEST/'tuning.csv',index=False);pd.DataFrame(fits).to_csv(DEST/'fits.csv',index=False)
    fields=['pattern_nll','count_nll','conditional_identity_nll','count_brier','multi_brier','new_analyte_brier']
    summaries=[];contrasts=[]
    for evaluation in ['blocked','calendar']:
        zz=allrows.loc[allrows.evaluation.eq(evaluation)]
        names=sorted(zz.model.unique());base=zz.loc[zz.model.eq(names[0])].set_index('location_id')
        data=base.reset_index()[['location_id','pws_id','state']]
        tensor=np.stack([zz.loc[zz.model.eq(m)].set_index('location_id').loc[base.index,fields].to_numpy() for m in names],axis=1)
        point=tensor.mean(axis=0);draws=np.zeros((REPS,len(names),len(fields)))
        for r,w in enumerate(cluster_weights(data)):
            draws[r]=np.einsum('n,nmf->mf',w,tensor)/w.sum()
        for j,m in enumerate(names):
            for k,f in enumerate(fields):summaries.append({'evaluation':evaluation,'model':m,'metric':f,'locations':len(data),'estimate':point[j,k],**ci(draws[:,j,k])})
        for ref,cand in [('independent_history_lower_ridge','softmax_identity_lower_ridge'),
                         ('empirical_count_identity','softmax_identity_lower_ridge')]:
            a=names.index(ref);b=names.index(cand)
            for k,f in enumerate(fields):contrasts.append({'evaluation':evaluation,'reference':ref,'candidate':cand,'metric':f,
                                                          'loss_reduction':point[a,k]-point[b,k],**ci(draws[:,a,k]-draws[:,b,k])})
    pd.DataFrame(summaries).to_csv(DEST/'metrics.csv',index=False);pd.DataFrame(contrasts).to_csv(DEST/'paired_contrasts.csv',index=False)
    dump(DEST/'AUDIT.json',{'status':'PASS','calendar_training':len(train),'calendar_validation':len(val),'calendar_refit':len(fitdata),
                           'calendar_test':len(test),'calendar_PWS_overlap':0,'calendar_latest_training_outcome':fitdata.date_next.max(),
                           'calendar_earliest_test_baseline':test.date_first.min(),'identity_refitted_using_earlier_data_only':True,
                           'all_model_selection_used_validation_only':True,'primary_outputs_unchanged':True,
                           'post_result_sensitivity_not_confirmation':True})
    check();print(pd.DataFrame(contrasts).query("metric=='pattern_nll'").to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','run']);args=p.parse_args()
    freeze() if args.mode=='freeze' else run()
