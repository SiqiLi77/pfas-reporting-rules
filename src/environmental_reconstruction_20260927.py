"""Frozen-prediction evaluation of environmental contrasts and chemical resolution."""
from pathlib import Path
from itertools import combinations
import hashlib, json
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/environmental_reconstruction_20260927'
NATIVE = ROOT / 'outputs/est_four_extensions_v1_30_20260925/selective'
AUX = ROOT / 'outputs/auxiliary_information_20260926'
WA = ROOT / 'outputs/auxiliary_followup_20260926/washington'
COHORT = ROOT / 'work/ucmr5_final_20260828_r1/ucmr5_artificial_censor_strict.csv.gz'
CHEMS = ['PFBS','PFHpA','PFHxS','PFNA','PFOA','PFOS']
BITS = ((np.arange(64)[:, None] >> np.arange(6)) & 1)
K = BITS.sum(1)
FAMILY = (BITS[:, [1,3,4]].sum(1)>0).astype(int) + 2*(BITS[:, [0,2,5]].sum(1)>0)
PAIRS = list(combinations(range(6), 2))
ENDPOINTS = ['multiple','both_families'] + [CHEMS[a]+'+'+CHEMS[b] for a,b in PAIRS]
H = np.column_stack([K>=2,FAMILY==3]+[BITS[:,a]*BITS[:,b] for a,b in PAIRS]).astype(float)
MAIN_ENDPOINTS = ['multiple','both_families','PFOA+PFOS','PFHxS+PFOS']
CONDITIONAL = ['family_given_count_nll','pair_given_count_nll__PFOA+PFOS','pair_given_count_nll__PFHxS+PFOS']
FIXED = ['factorized','conditional_shared_shock','conditional_gaussian_factor2','training_composition']
AUX_MODELS = [s+'__'+m for s in ['baseline','selected3'] for m in ['independent','linear','nonlinear']]
REPS, SEED = 5000, 2026092701

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def dump(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def freeze():
    files=[Path(__file__),ROOT/'tests/test_environmental_reconstruction_20260927.py',OUT/'PROTOCOL.md',COHORT,AUX/'cohort.csv.gz',
           WA/'metadata.csv',WA/'permitted_inputs.csv',WA/'outcomes.csv',WA/'current_metadata_proxy.npz']
    files += [NATIVE/f'native_{typ}_fold_{f}.{ext}' for f in range(1,5)
              for typ,ext in [('probabilities','npz'),('metadata','csv.gz')]]
    files += [AUX/f'fold{f}/predictions.npz' for f in range(5)]
    record={'locked_at_utc':datetime.now(timezone.utc).isoformat(),'new_fits':0,
            'post_development_exploratory':True,'bootstrap_replicates':REPS,
            'sources':{str(p.relative_to(ROOT)):sha(p) for p in files}}
    p=OUT/'INPUT_LOCK.json'
    if p.exists():
        assert record['sources']==json.loads(p.read_text())['sources'],'Frozen inputs changed'
    else:dump(p,record)

def load_data():
    types={'location_id':str,'pws_id':str,'state':str}
    original=pd.read_csv(COHORT,dtype=types)
    aux=pd.read_csv(AUX/'cohort.csv.gz',dtype=types)
    md=pd.concat([pd.read_csv(NATIVE/f'native_metadata_fold_{f}.csv.gz',dtype=types) for f in range(1,5)],ignore_index=True)
    assert not md.location_id.duplicated().any() and len(md)==21156
    ids=pd.Index(sorted(set(md.location_id)&set(aux.location_id)))
    d=aux.set_index('location_id').loc[ids].reset_index(names='location_id')
    mm=md.set_index('location_id').loc[ids].reset_index(names='location_id')
    raw=original.set_index('location_id').loc[ids].reset_index(names='location_id')
    for c in ['pws_id','state']:
        assert d[c].equals(mm[c]) and d[c].equals(raw[c]),c
    for c in ['pws_size','water_type','state_fold']:
        assert d[c].equals(raw[c]),c
    assert np.array_equal(d.state_fold,mm.outer_fold)
    y=d[['y_native_detect__'+c for c in CHEMS]].to_numpy(int)@(2**np.arange(6))
    old=d[['old_visible__'+c for c in CHEMS]].to_numpy(int)@(2**np.arange(6))
    assert np.array_equal(y,mm.observed_state_code) and np.array_equal(old,mm.old_visible_code)
    d['observed_state_code']=y; d['old_visible_code']=old
    lookup=pd.Series(np.arange(len(md)),index=md.location_id).loc[ids].to_numpy()
    models={m:np.concatenate([np.load(NATIVE/f'native_probabilities_fold_{f}.npz')[m] for f in range(1,5)])[lookup] for m in FIXED}
    aq={m:np.full((len(aux),64),np.nan) for m in AUX_MODELS};coverage=np.zeros(len(aux),int)
    for f in range(5):
        z=np.load(AUX/f'fold{f}/predictions.npz');ix=z['index'];coverage[ix]+=1
        assert np.array_equal(ix,np.flatnonzero(aux.state_fold.eq(f)))
        for m in AUX_MODELS:aq[m][ix]=z[m]
    assert np.all(coverage==1)
    ix=pd.Series(np.arange(len(aux)),index=aux.location_id).loc[ids].to_numpy()
    models.update({m:q[ix] for m,q in aq.items()})
    wm=pd.read_csv(WA/'metadata.csv',dtype=types)
    wi=pd.read_csv(WA/'permitted_inputs.csv',dtype={'location_id':str})
    wy=pd.read_csv(WA/'outcomes.csv')[CHEMS].to_numpy(int)@(2**np.arange(6))
    wz=np.load(WA/'current_metadata_proxy.npz');wx=wz['index']
    assert np.array_equal(wx,np.flatnonzero(wm.proxy_eligible.eq(1)))
    w=wm.iloc[wx].copy().reset_index(drop=True)
    assert len(w)==1819 and w.pws_id.nunique()==1456
    assert list(wi.location_id)==list(wm.location_id)
    w['observed_state_code']=wy[wx]
    w['old_visible_code']=wi.iloc[wx][['old_visible__'+c for c in CHEMS]].to_numpy(int)@(2**np.arange(6))
    w['state']='WA';w['water_type']=w.proxy_water;w['pws_size']=w.proxy_size
    wq={m:wz[m] for m in AUX_MODELS}
    audit={'national_common_events':len(d),'national_original_fixed_events':len(md),'national_auxiliary_events':len(aux),
           'national_common_pws':d.pws_id.nunique(),'national_common_jurisdictions':d.state.nunique(),
           'national_excluded_fixed_without_aux':len(md)-len(d),'native_endpoint_alpha':0,
           'label_and_metadata_alignment':True,'same_rows_all_national_comparators':True,
           'WA_events':len(w),'WA_pws':w.pws_id.nunique(),'WA_counties':w.county.nunique()}
    dump(OUT/'COHORT_AUDIT.json',audit)
    return [('national',d,models),('washington',w,wq)]

def summarize_probabilities(q,codes,old):
    assert q.shape==(len(codes),64) and np.isfinite(q).all() and q.min()>=-1e-10
    norm_error=float(np.max(abs(q.sum(1)-1)));assert norm_error<2e-6
    q=np.maximum(q,0);q=q/q.sum(1,keepdims=True)
    retained=BITS[old].astype(bool)
    assert np.max(abs((q@BITS)[retained]-1),initial=0)<2e-6
    p=q@H;y=H[codes];k=K[codes];family=FAMILY[codes];n=len(codes);row=np.arange(n)
    qs=(1-1e-12)*q+1e-12/64
    count=qs@np.eye(7)[K];fq=qs@np.eye(4)[FAMILY]
    score={'brier__'+e:(p[:,j]-y[:,j])**2 for j,e in enumerate(ENDPOINTS)}
    score.update(pattern_nll=-np.log(qs[row,codes]),count_nll=-np.log(count[row,k]),
                 identity_given_count_nll=-np.log(qs[row,codes]/count[row,k]),
                 family_nll=-np.log(fq[row,family]))
    fc=qs@np.eye(28)[4*K+FAMILY]
    score['family_given_count_nll']=-np.log(fc[row,4*k+family]/count[row,k])
    pair_cond=np.zeros((n,15))
    for kval in range(7):
        rows=np.flatnonzero(k==kval);st=K==kval
        sub=qs[np.ix_(rows,np.flatnonzero(st))]
        yes=sub@H[st,2:];no=sub@(1-H[st,2:])
        num=np.where(y[rows,2:]>0,yes,no)
        pair_cond[rows]=-np.log(np.clip(num/count[rows,kval,None],1e-300,1))
    score.update({'pair_given_count_nll__'+e:pair_cond[:,j] for j,e in enumerate(ENDPOINTS[2:])})
    assert np.max(abs(score['pattern_nll']-score['count_nll']-score['identity_given_count_nll']))<2e-12
    assert np.max(abs(pair_cond[(k==0)|(k==6)]),initial=0)<2e-12
    assert np.max(score['family_given_count_nll']-score['identity_given_count_nll'])<2e-12
    return p,score,{'normalization_before':norm_error,'normalization_after':float(np.max(abs(q.sum(1)-1))),
                    'min_probability':float(q.min()),'chain_rule_error':float(np.max(abs(score['pattern_nll']-score['count_nll']-score['identity_given_count_nll'])))}

def pws_matrix(d,v):
    keys=d.state.astype(str)+'|'+d.pws_id.astype(str)
    unique,inv=np.unique(keys,return_inverse=True)
    num=np.zeros((len(unique),v.shape[1]));np.add.at(num,inv,v)
    ns=np.bincount(inv)
    states=np.array([s.split('|')[0] for s in unique])
    return unique,inv,num/ns[:,None],states

def score_bootstrap(d,v,scheme='pws'):
    unique,inv,means,states=pws_matrix(d,v)
    rng=np.random.default_rng(SEED);point=np.zeros(v.shape[1]);draw=np.zeros((REPS,v.shape[1]))
    if scheme=='county':
        counties=d.groupby('pws_id').county.nunique();assert counties.max()==1
        county=d.groupby('pws_id').county.first()
        lab=np.array([county[k.split('|',1)[1]] for k in unique]);labs=np.unique(lab)
        counts=np.array([(lab==a).sum() for a in labs]);sums=np.stack([means[lab==a].sum(0) for a in labs])
        for start in range(0,REPS,100):
            b=min(100,REPS-start);w=rng.multinomial(len(labs),np.full(len(labs),1/len(labs)),size=b)
            draw[start:start+b]=(w@sums)/(w@counts)[:,None]
        return means.mean(0),draw
    for state in np.unique(states):
        a=means[states==state];point+=a.mean(0)/len(np.unique(states))
        for start in range(0,REPS,100):
            b=min(100,REPS-start);w=rng.multinomial(len(a),np.full(len(a),1/len(a)),size=b)
            draw[start:start+b]+=w@a/len(a)/len(np.unique(states))
    return point,draw

def interval(point,draw):
    v=np.asarray(draw);v=v[np.isfinite(v)]
    lo,hi=np.quantile(v,[.025,.975]) if len(v) else (np.nan,np.nan)
    return {'estimate':float(point),'lower95':float(lo),'upper95':float(hi),'finite_draws':len(v)}

def model_pairs(models):
    pairs=[]
    if 'factorized' in models:
        pairs += [('factorized',m) for m in ['conditional_shared_shock','conditional_gaussian_factor2','training_composition']]
    pairs += [('baseline__'+m,'selected3__'+m) for m in ['independent','linear','nonlinear']]
    return pairs

def chemical_analysis(name,d,prob,scores):
    keys=[];vec=[]
    for m in prob:
        for j,e in enumerate(ENDPOINTS):keys.append((m,'rate__'+e));vec.append(prob[m][:,j])
        for metric,v in scores[m].items():keys.append((m,metric));vec.append(v)
    for j,e in enumerate(ENDPOINTS):keys.append(('measured','rate__'+e));vec.append(H[d.observed_state_code,j])
    v=np.column_stack(vec);lookup={k:i for i,k in enumerate(keys)}
    rows=[];contrasts=[]
    for scheme in (['pws','county'] if name=='washington' else ['pws']):
        point,draw=score_bootstrap(d,v,scheme);print(name,'chemical bootstrap',scheme,flush=True)
        for j,(model,metric) in enumerate(keys):
            rows.append({'cohort':name,'resampling':scheme,'model':model,'metric':metric,**interval(point[j],draw[:,j])})
        for before,after in model_pairs(prob):
            metrics=list(scores[before]);family=[]
            for metric in metrics:
                i,j=lookup[before,metric],lookup[after,metric]
                delta=point[i]-point[j];dd=draw[:,i]-draw[:,j]
                r={'cohort':name,'resampling':scheme,'reference':before,'candidate':after,'metric':metric,
                   'reference_score':point[i],'candidate_score':point[j],**interval(delta,dd)}
                contrasts.append(r)
                if metric in CONDITIONAL:family.append((r,delta,dd))
            sd=np.array([np.std(f[2],ddof=1) for f in family]);sd=np.maximum(sd,1e-15)
            centered=np.column_stack([(f[2]-f[1])/s for f,s in zip(family,sd)])
            critical=np.quantile(np.max(abs(centered),axis=1),.95)
            for (r,delta,_),s in zip(family,sd):
                r.update(simultaneous_lower95=delta-critical*s,simultaneous_upper95=delta+critical*s,
                         simultaneous_family_size=len(family))
        np.savez_compressed(OUT/name/f'chemical_bootstrap_{scheme}.npz',point=point,draws=draw,
                            keys=np.array(['|'.join(k) for k in keys]))
    pd.DataFrame(rows).to_csv(OUT/name/'chemical_scores.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/name/'chemical_paired_contrasts.csv',index=False)
    np.savez_compressed(OUT/name/'record_summaries.npz',values=v,keys=np.array(['|'.join(k) for k in keys]))

def source_analysis(name,d,probs):
    ixmain=[ENDPOINTS.index(e) for e in MAIN_ENDPOINTS]
    allmodels=['measured','retained_old']+list(probs)
    values=np.stack([H[d.observed_state_code][:,ixmain],H[d.old_visible_code][:,ixmain]]+
                    [p[:,ixmain] for p in probs.values()],axis=1)
    cells=d.loc[d.water_type.isin(['GW','SW'])].groupby(['state','pws_size','water_type']).agg(
        locations=('location_id','size'),pws=('pws_id','nunique')).reset_index()
    allowed=[]
    for key,g in cells.groupby(['state','pws_size']):
        if set(g.water_type)=={'GW','SW'} and (g.locations>=20).all() and (g.pws>=5).all():allowed.append(key)
    d=d.copy();d['stratum']=d.state+'|'+d.pws_size
    cellnames=[s+'|'+z for s,z in sorted(allowed)]
    take=d.stratum.isin(cellnames)&d.water_type.isin(['GW','SW'])
    cells['supported']=[(s,z) in allowed for s,z in zip(cells.state,cells.pws_size)]
    cells.to_csv(OUT/name/'source_support.csv',index=False)
    if not len(cellnames):
        dump(OUT/name/'SOURCE_AUDIT.json',{'supported_strata':0,'reason':'No metadata-defined common support'});return
    ds=d.loc[take].reset_index(drop=True);x=values[take].reshape(take.sum(),-1)
    weights=ds.stratum.value_counts().reindex(cellnames).to_numpy()/len(ds)
    sw=ds.water_type.eq('SW').to_numpy(int);cell=pd.Categorical(ds.stratum,categories=cellnames).codes
    ids,inv=np.unique(ds.state+'|'+ds.pws_id,return_inverse=True)
    states=np.array([i.split('|')[0] for i in ids]);ustates=np.unique(states)
    point=np.zeros((2,x.shape[1]));draw=np.zeros((REPS,2,x.shape[1]));valid=np.ones(REPS,bool)
    cellmeans=[];rng=np.random.default_rng(SEED+11)
    for state in ustates:
        pwsix=np.flatnonzero(states==state);wm=rng.multinomial(len(pwsix),np.full(len(pwsix),1/len(pwsix)),size=REPS)
        local={p:i for i,p in enumerate(pwsix)}
        for c,cname in enumerate(cellnames):
            if cname.split('|')[0]!=state:continue
            cm=np.empty((2,x.shape[1]))
            for s in range(2):
                takec=(cell==c)&(sw==s);ri=np.flatnonzero(takec)
                count=np.zeros(len(pwsix));sums=np.zeros((len(pwsix),x.shape[1]))
                wi=np.array([local[v] for v in inv[ri]])
                np.add.at(count,wi,1);np.add.at(sums,wi,x[ri])
                cm[s]=x[ri].mean(0);point[s]+=weights[c]*cm[s]
                denom=wm@count;valid&=denom>0
                draw[:,s]+=weights[c]*(wm@sums)/np.where(denom>0,denom,np.nan)[:,None]
            cellmeans.append((c,cm))
    draw[~valid]=np.nan
    point=point.reshape(2,len(allmodels),len(MAIN_ENDPOINTS))
    draw=draw.reshape(REPS,2,len(allmodels),len(MAIN_ENDPOINTS))
    reference=point[:,0];delta=point[1]-point[0];bd=draw[:,1]-draw[:,0]
    rows=[];errors=[]
    for m,model in enumerate(allmodels):
        for j,e in enumerate(MAIN_ENDPOINTS):
            for s,g in enumerate(['GW','SW']):
                rows.append({'cohort':name,'condition':model,'endpoint':e,'quantity':g+'_rate',**interval(point[s,m,j],draw[:,s,m,j])})
                if m:
                    errors.append({'cohort':name,'condition':model,'endpoint':e,'quantity':g+'_bias',
                                   **interval(point[s,m,j]-reference[s,j],draw[:,s,m,j]-draw[:,s,0,j])})
            rows.append({'cohort':name,'condition':model,'endpoint':e,'quantity':'SW_minus_GW',**interval(delta[m,j],bd[:,m,j])})
            if m:
                err=delta[m,j]-delta[0,j];be=bd[:,m,j]-bd[:,0,j]
                errors.append({'cohort':name,'condition':model,'endpoint':e,'quantity':'contrast_bias',**interval(err,be)})
                errors.append({'cohort':name,'condition':model,'endpoint':e,'quantity':'absolute_contrast_error',**interval(abs(err),abs(be))})
                if m>1:
                    gain=abs(delta[1,j]-delta[0,j])-abs(err)
                    bg=abs(bd[:,1,j]-bd[:,0,j])-abs(be)
                    errors.append({'cohort':name,'condition':model,'endpoint':e,'quantity':'absolute_error_reduction_vs_retained',**interval(gain,bg)})
    for before,after in model_pairs(probs):
        a,b=allmodels.index(before),allmodels.index(after)
        for j,e in enumerate(MAIN_ENDPOINTS):
            gain=abs(delta[a,j]-delta[0,j])-abs(delta[b,j]-delta[0,j])
            bg=abs(bd[:,a,j]-bd[:,0,j])-abs(bd[:,b,j]-bd[:,0,j])
            errors.append({'cohort':name,'condition':after,'reference':before,'endpoint':e,'quantity':'paired_absolute_contrast_error_reduction',**interval(gain,bg)})
    pd.DataFrame(rows).to_csv(OUT/name/'source_rates_and_contrasts.csv',index=False)
    pd.DataFrame(errors).to_csv(OUT/name/'source_reconstruction_errors.csv',index=False)
    cw=pd.DataFrame({'stratum':cellnames,'weight':weights});cw.to_csv(OUT/name/'standardization_weights.csv',index=False)
    cm=np.stack([a for _,a in sorted(cellmeans)])
    cstate=np.array([s.split('|')[0] for s in cellnames]);sens=[]
    wjur=np.zeros(len(weights))
    for s in np.unique(cstate):
        use=cstate==s;wjur[use]=weights[use]/weights[use].sum()/len(np.unique(cstate))
    schemes={'primary':weights,'jurisdiction_equal':wjur}
    for s in np.unique(cstate):
        w0=weights*(cstate!=s)
        if w0.sum():schemes['without_'+s]=w0/w0.sum()
    for label,w0 in schemes.items():
        z=np.einsum('c,csj->sj',w0,cm).reshape(2,len(allmodels),len(MAIN_ENDPOINTS))
        for m,model in enumerate(allmodels):
            for j,e in enumerate(MAIN_ENDPOINTS):sens.append({'cohort':name,'weighting':label,'condition':model,'endpoint':e,'GW_rate':z[0,m,j],'SW_rate':z[1,m,j],'contrast':z[1,m,j]-z[0,m,j]})
    pd.DataFrame(sens).to_csv(OUT/name/'source_weight_sensitivities.csv',index=False)
    np.savez_compressed(OUT/name/'source_bootstrap.npz',point=point,draws=draw,models=np.array(allmodels),endpoints=np.array(MAIN_ENDPOINTS))
    # Independent dataframe group means, with explicit common weights.
    for col in range(x.shape[1]):
        f=ds[['stratum','water_type']].copy();f['v']=x[:,col]
        independent=weights@f.groupby(['stratum','water_type']).v.mean().unstack().loc[cellnames,['GW','SW']].to_numpy()
        np.testing.assert_allclose(independent,point.reshape(2,-1)[:,col],atol=1e-12,rtol=0)
    positives=[]
    for s in ['GW','SW']:
        use=ds.water_type.eq(s)
        positives.append({'source':s,'locations':int(use.sum()),'pws':int(ds.loc[use,'pws_id'].nunique()),
                          **{e:int(H[ds.loc[use,'observed_state_code'],ENDPOINTS.index(e)].sum()) for e in MAIN_ENDPOINTS}})
    dump(OUT/name/'SOURCE_AUDIT.json',{'supported_strata':len(cellnames),'supported_locations':len(ds),'supported_pws':ds.pws_id.nunique(),
         'supported_jurisdictions':ds.state.nunique(),'excluded_locations':len(d)-len(ds),'finite_draws':int(valid.sum()),
         'undefined_draws':int((~valid).sum()),'group_support':positives,'independent_group_mean_check':True,
         'metadata_are_current_proxies':name=='washington'})
    ds[['location_id','pws_id','state','pws_size','water_type','observed_state_code','old_visible_code','stratum']].to_csv(OUT/name/'source_analysis_rows.csv.gz',index=False)
    print(name,'source bootstrap done',len(ds),'locations',len(cellnames),'strata',flush=True)

def run():
    freeze();data=load_data();audits=[]
    for name,d,models in data:
        folder=OUT/name;folder.mkdir(exist_ok=True)
        keep=['location_id','pws_id','state','pws_size','water_type','observed_state_code','old_visible_code']
        if name=='washington':keep+=['county']
        d[keep].to_csv(folder/'metadata.csv.gz',index=False)
        probs={};scores={}
        for m,q in models.items():
            p,s,a=summarize_probabilities(q,d.observed_state_code.to_numpy(int),d.old_visible_code.to_numpy(int))
            probs[m]=p;scores[m]=s;audits.append({'cohort':name,'model':m,**a})
        if name=='national':
            for m in FIXED[1:3]:np.testing.assert_allclose(models[m]@BITS,models['factorized']@BITS,atol=2e-8,rtol=0)
        chemical_analysis(name,d,probs,scores)
        source_analysis(name,d,probs)
    pd.DataFrame(audits).to_csv(OUT/'probability_audit.csv',index=False)
    freeze()
    dump(OUT/'DONE.json',{'completed_at_utc':datetime.now(timezone.utc).isoformat(),'cohorts':2,'models_checked':len(audits),'new_fits':0,
                         'bootstrap_replicates':REPS,'all15pairs_reported':True,'post_development_exploratory':True})

if __name__=='__main__':run()
