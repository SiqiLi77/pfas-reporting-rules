"""Full next-sample distribution: predict count, then conditional identities."""
import json
from itertools import product
from pathlib import Path
import time
import numpy as np
import pandas as pd
from research_expansion_common_20260925 import OUT, IDENTITY, REPS, verify, dump, cluster_weights, ci, read_paired
from prepare_monitoring_followup_20260925 import A25
from run_conditional_identity_probe_20260925 import input_arrays, unnormalized, lbfgs

DEST=OUT/'forecast'
PENALTIES=[.0001,.001,.01]
INDEPENDENT=['independent_site_count','independent_history']
MODELS=INDEPENDENT+['predicted_count_identity']


def softmax(z):
    shift=z.max(axis=1,keepdims=True)
    a=np.exp(z-shift)
    return a/a.sum(axis=1,keepdims=True)


def sigmoid(z):
    return np.exp(-np.logaddexp(0.,-z))


def objective(theta,x,previous,y,kind,penalty):
    n=len(y);p=x.shape[1]
    if kind=='count':
        k=y.sum(axis=1).astype(int)
        logits=x@theta.reshape(p,26)
        shift=logits.max(axis=1)
        normalizer=shift+np.log(np.exp(logits-shift[:,None]).sum(axis=1))
        loss=np.mean(normalizer-logits[np.arange(n),k])
        error=softmax(logits);error[np.arange(n),k]-=1
        grad=(x.T@error/n).ravel()
    else:
        logits=x@theta[:p*25].reshape(p,25)
        if kind=='independent_history':logits+=theta[-1]*previous
        loss=np.mean((np.logaddexp(0.,logits)-y*logits).sum(axis=1))
        error=(sigmoid(logits)-y)/n
        parts=[(x.T@error).ravel()]
        if kind=='independent_history':parts.append(np.array([(error*previous).sum()]))
        grad=np.concatenate(parts)
    return float(loss+.5*penalty*np.dot(theta,theta)),grad+penalty*theta


def fit(d,kind,penalty):
    x,previous,spec=input_arrays(d)
    y=d[[f'next_report__{a}' for a in A25]].to_numpy(float)
    width=26 if kind=='count' else 25
    theta=np.zeros(x.shape[1]*width+(kind=='independent_history'))
    if kind=='count':
        counts=np.bincount(y.sum(axis=1).astype(int),minlength=26)+.5
        prior=np.log(counts/counts.sum());theta[:26]=prior-prior.mean()
    else:
        p=(y.sum(axis=0)+.5)/(len(y)+1)
        theta[:25]=np.log(p/(1-p))
    fun=lambda v:objective(v,x,previous,y,kind,penalty)
    theta,info=lbfgs(fun,theta,250)
    info['retried']=False
    if not info['converged']:
        theta,info=lbfgs(fun,theta,600);info['retried']=True
    assert info['converged'],(kind,penalty,info)
    info.update(parameters=len(theta),training_locations=len(d))
    return theta,spec,info


def logits_of(theta,spec,d,kind):
    x,previous,_=input_arrays(d,spec)
    if kind=='count':return x@theta.reshape(x.shape[1],26)
    logits=x@theta[:x.shape[1]*25].reshape(x.shape[1],25)
    if kind=='independent_history':logits+=theta[-1]*previous
    return logits


def count_distribution(p):
    n,m=p.shape
    mass=np.zeros((n,m+1));mass[:,0]=1.
    for j in range(m):
        mass[:,1:j+2]=mass[:,1:j+2]*(1-p[:,j,None])+mass[:,:j+1]*p[:,j,None]
        mass[:,0]*=1-p[:,j]
    assert np.max(np.abs(mass.sum(axis=1)-1))<1e-10
    assert mass.min()>=0
    return mass


def log_elementary(logits):
    n,m=logits.shape
    z=np.full((n,m+1),-np.inf);z[:,0]=0.
    for j in range(m):
        z[:,1:j+2]=np.logaddexp(z[:,1:j+2],logits[:,j,None]+z[:,:j+1])
    return z


def novelty_probability(count,logits,previous):
    denominator=log_elementary(logits)
    numerator=log_elementary(np.where(previous>0,logits,-np.inf))
    ratio=np.exp(numerator-denominator)
    no_new=np.sum(count*ratio,axis=1)
    assert no_new.min()>=-1e-10 and no_new.max()<=1+1e-10
    return np.clip(1-no_new,0,1),denominator


def scores(d,logits,count,kind):
    y=d[[f'next_report__{a}' for a in A25]].to_numpy(float)
    previous=d[[f'first_report__{a}' for a in A25]].to_numpy(float)
    k=y.sum(axis=1).astype(int);idx=np.arange(len(d))
    countloss=-np.log(np.maximum(count[idx,k],1e-300))
    if kind=='structured':
        novelty,z=novelty_probability(count,logits,previous)
        identity=z[idx,k]-(y*logits).sum(axis=1)
        full=countloss+identity
    else:
        full=(np.logaddexp(0.,logits)-y*logits).sum(axis=1)
        identity=full-countloss
        lognone=(-np.logaddexp(0.,logits)*(1-previous)).sum(axis=1)
        novelty=-np.expm1(lognone)
    identity[np.isin(k,[0,25])]=0.
    assert identity.min()>-1e-7
    assert np.max(np.abs(full-countloss-identity))<1e-7
    onehot=np.zeros_like(count);onehot[idx,k]=1
    binary=1-count[:,:2].sum(axis=1)
    new=((y>0)&(previous==0)).any(axis=1).astype(float)
    result={'pattern_nll':full,'count_nll':countloss,'conditional_identity_nll':np.maximum(identity,0.),
            'count_brier':((count-onehot)**2).sum(axis=1),
            'count_ranked_probability_score':((np.cumsum(count,axis=1)[:,:-1]-np.cumsum(onehot,axis=1)[:,:-1])**2).sum(axis=1),
            'multi_brier':(binary-(k>=2))**2,'new_analyte_brier':(novelty-new)**2}
    return result,novelty,binary


def tests():
    rng=np.random.default_rng(202609253)
    x=np.column_stack([np.ones(4),rng.normal(size=(4,2))]);prev=rng.integers(0,2,size=(4,25)).astype(float)
    y=rng.integers(0,2,size=(4,25)).astype(float)
    maxerr=0.
    for kind in INDEPENDENT+['count']:
        size=x.shape[1]*(26 if kind=='count' else 25)+(kind=='independent_history')
        theta=rng.normal(0,.2,size)
        f,g=objective(theta,x,prev,y,kind,.001)
        for j in rng.choice(size,20,replace=False):
            e=np.zeros(size);e[j]=1e-5
            numeric=(objective(theta+e,x,prev,y,kind,.001)[0]-objective(theta-e,x,prev,y,kind,.001)[0])/2e-5
            maxerr=max(maxerr,abs(numeric-g[j]))
    assert maxerr<1e-7,maxerr
    for m in [3,6]:
        logits=rng.normal(0,3,(5,m));p=sigmoid(logits);mass=count_distribution(p)
        patterns=np.array(list(product([0,1],repeat=m)))
        k=patterns.sum(axis=1).astype(int)
        probability=np.exp(patterns@logits.T-np.logaddexp(0,logits).sum(axis=1)).T
        enum=np.stack([probability[:,k==j].sum(axis=1) for j in range(m+1)],axis=1)
        assert np.max(np.abs(enum-mass))<1e-11
        count=softmax(rng.normal(size=(5,m+1)));previous=rng.integers(0,2,size=(5,m))
        novelty,z=novelty_probability(count,logits,previous)
        for i in range(5):
            joint=count[i,k]*np.exp(patterns@logits[i]-z[i,k])
            assert abs(joint.sum()-1)<1e-11
            new=((patterns>0)&(previous[i]==0)).any(axis=1)
            assert abs(joint[new].sum()-novelty[i])<1e-11
    return {'gradient_max_absolute_error':maxerr,'small_panel_full_enumeration':True,
            'novelty_probability_enumeration':True}


def run():
    verify();DEST.mkdir(exist_ok=True);(DEST/'models').mkdir(exist_ok=True)
    checks=tests();dump(DEST/'NUMERICAL_TESTS.json',checks)
    d=read_paired();tuning=[];fits=[];losses={}; predictions=[]
    for fold in range(5):
        testmask=d.state_fold.eq(fold).to_numpy();valmask=d.state_fold.eq((fold+1)%5).to_numpy()
        train=d.loc[~(testmask|valmask)];val=d.loc[valmask];test=d.loc[testmask]
        assert not set(train.pws_id)&set(test.pws_id)
        assert not set(val.pws_id)&set(test.pws_id)
        assert not set(train.state)&set(test.state)
        fitted={}
        for kind in INDEPENDENT+['count']:
            start=time.monotonic(); trials=[]
            vy=val[[f'next_report__{a}' for a in A25]].to_numpy(float)
            for penalty in PENALTIES:
                theta,spec,info=fit(train,kind,penalty)
                logit=logits_of(theta,spec,val,kind)
                if kind=='count':
                    prob=softmax(logit);score=float(-np.log(prob[np.arange(len(val)),vy.sum(axis=1).astype(int)]).mean())
                else:score=float((np.logaddexp(0,logit)-vy*logit).sum(axis=1).mean())
                trials.append((score,penalty))
                tuning.append({'fold':fold,'kind':kind,'penalty':penalty,'validation_nll':score,**info})
                print(f'Fold {fold} {kind} penalty {penalty:g} validation {score:.6f}',flush=True)
            penalty=min(trials)[1]
            theta,spec,info=fit(d.loc[~testmask],kind,penalty)
            logits=logits_of(theta,spec,test,kind)
            prefix=DEST/'models'/f'fold{fold}_{kind}'
            np.savez_compressed(str(prefix)+'.npz',theta=theta)
            dump(Path(str(prefix)+'.json'),{'kind':kind,'penalty':penalty,'feature_spec':spec,'fit':info})
            replay=logits_of(np.load(str(prefix)+'.npz')['theta'],spec,test,kind)
            assert np.array_equal(logits,replay)
            changed=test.copy()
            for col in changed:
                if col.startswith('next_') or col in ['date_next','lag_days']:changed[col]=98765
            assert np.array_equal(logits,logits_of(theta,spec,changed,kind))
            fitted[kind]=logits
            fits.append({'fold':fold,'kind':kind,'penalty':penalty,'test_locations':len(test),
                         'seconds':time.monotonic()-start,'future_input_invariance':True,**info})
            pd.DataFrame(tuning).to_csv(DEST/'tuning.csv',index=False)
            pd.DataFrame(fits).to_csv(DEST/'fits.csv',index=False)
        # Reuse frozen identity coefficients without fitting to the new test records.
        prior=IDENTITY/'models'/f'fold{fold}_shared_persistence'
        theta=np.load(str(prior)+'.npz')['theta']
        meta=json.loads(Path(str(prior)+'.json').read_text())
        x,previous,_=input_arrays(test,meta['feature_spec'])
        identity_logits=unnormalized(theta,x,previous,'shared_persistence')
        altered=test.copy()
        for col in altered:
            if col.startswith('next_') or col in ['date_next','lag_days']:altered[col]=123456
        ax,ap,_=input_arrays(altered,meta['feature_spec'])
        assert np.array_equal(identity_logits,unnormalized(theta,ax,ap,'shared_persistence'))
        pred=test[['location_id','pws_id','state','state_fold','first_k25','next_k25']].copy()
        for model in MODELS:
            if model=='predicted_count_identity':
                logits=identity_logits;count=softmax(fitted['count']);kind='structured'
            else:
                logits=fitted[model];count=count_distribution(sigmoid(logits));kind='independent'
            assert np.max(np.abs(count.sum(axis=1)-1))<1e-10
            ss,novel,binary=scores(test,logits,count,kind)
            if model not in losses:losses[model]={k:np.full(len(d),np.nan) for k in ss}
            for field,v in ss.items():losses[model][field][testmask]=v
            pred['p_new_analyte__'+model]=novel
            pred['p_multi__'+model]=binary
            for k in range(26):pred[f'p_count{k}__'+model]=count[:,k]
        predictions.append(pred)
        print('Completed full forecast fold',fold,flush=True)
    fields=list(losses[MODELS[0]])
    tensor=np.stack([np.column_stack([losses[m][f] for f in fields]) for m in MODELS],axis=1)
    assert tensor.shape==(len(d),len(MODELS),len(fields)) and np.isfinite(tensor).all()
    point=tensor.mean(axis=0);draws=np.zeros((REPS,len(MODELS),len(fields)))
    for r,w in enumerate(cluster_weights(d)):
        draws[r]=np.einsum('n,nmf->mf',w,tensor)/w.sum()
        if (r+1)%500==0:print('Forecast score resamples',r+1,flush=True)
    rows=[];contrasts=[]
    for j,m in enumerate(MODELS):
        for k,f in enumerate(fields):
            rows.append({'model':m,'metric':f,'estimate':point[j,k],'locations':len(d),**ci(draws[:,j,k])})
    for base,candidate,role in [('independent_site_count','independent_history','history_information'),
                               ('independent_history','predicted_count_identity','primary_algorithm')]:
        for k,f in enumerate(fields):
            b=MODELS.index(base);c=MODELS.index(candidate)
            contrasts.append({'reference':base,'candidate':candidate,'role':role,'metric':f,
                              'loss_reduction':point[b,k]-point[c,k],**ci(draws[:,b,k]-draws[:,c,k])})
    out=d[['location_id','pws_id','state','state_fold','first_k25','next_k25','lag_days','water_type']].copy()
    y=d[[f'next_report__{a}' for a in A25]].to_numpy(int)
    past=d[[f'first_report__{a}' for a in A25]].to_numpy(int)
    out['new_analyte_report']=((y>0)&(past==0)).any(axis=1).astype(int)
    same=np.all(y==past,axis=1);out['same_report_set']=same.astype(int)
    for m in MODELS:
        for f in fields:out[f+'__'+m]=losses[m][f]
    out.to_csv(DEST/'heldout_scores.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.concat(predictions).sort_index().to_csv(DEST/'heldout_probabilities.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(rows).to_csv(DEST/'metrics.csv',index=False)
    pd.DataFrame(contrasts).to_csv(DEST/'paired_contrasts.csv',index=False)
    np.savez_compressed(DEST/'score_bootstrap.npz',draws=draws,models=MODELS,fields=fields)
    sub=[]
    groups={'initial_zero':d.first_k25.eq(0).to_numpy(),'initial_one':d.first_k25.eq(1).to_numpy(),
            'initial_multiple':d.first_k25.ge(2).to_numpy(),'same_set':same,'changed_set':~same,
            'informative_next_count':d.next_k25.between(1,24).to_numpy()}
    for name,mask in groups.items():
        for j,m in enumerate(MODELS):
            for k,f in enumerate(fields):
                sub.append({'group':name,'model':m,'metric':f,'locations':int(mask.sum()),'estimate':float(tensor[mask,j,k].mean())})
    for fold in range(5):
        mask=d.state_fold.eq(fold).to_numpy()
        for j,m in enumerate(MODELS):
            sub.append({'group':f'fold{fold}','model':m,'metric':'pattern_nll','locations':int(mask.sum()),'estimate':float(tensor[mask,j,0].mean())})
    pd.DataFrame(sub).to_csv(DEST/'subgroup_scores.csv',index=False)
    dump(DEST/'AUDIT.json',{'status':'PASS','locations':len(d),'new_fits':len(tuning)+len(fits),
                           'all_converged':all(a['converged'] for a in fits+tuning),
                           'future_count_not_supplied':True,'future_feature_replacement':True,
                           'saved_replay':True,'test_jurisdictions_excluded_from_fitting':True,
                           'conditional_identity_models_reused_frozen':True,'score_decomposition_exact':True,
                           'bootstrap_replicates':REPS,'numerical_tests':checks,
                           'independent_confirmation':False})
    verify();print(pd.DataFrame(contrasts).to_string(index=False),flush=True)


if __name__=='__main__':run()
