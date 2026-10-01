"""Targeted no-refit diagnostic; reuses exact frozen support and bootstrap."""
from pathlib import Path
from datetime import datetime, timezone
import json, math
import numpy as np
import pandas as pd
import environmental_reconstruction_20260927 as e
from analyze_ucmr5_count_baseline_20260924 import project

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/marginal_source_diagnostic_20260927'
OLD = ROOT/'outputs/environmental_reconstruction_20260927/national'
CP = ROOT/'outputs/significance_extension_20260924/count_baseline/training_count_priors.csv'

def run():
    OUT.mkdir(exist_ok=True); (OUT/'national').mkdir(exist_ok=True)
    sources=[Path(__file__), OUT/'PROTOCOL.md', CP,
             ROOT/'src/analyze_ucmr5_count_baseline_20260924.py',
             ROOT/'src/environmental_reconstruction_20260927.py',
             OLD/'source_analysis_rows.csv.gz', OLD/'standardization_weights.csv',
             OLD/'source_bootstrap.npz']
    sources += [e.NATIVE/f'native_{t}_fold_{f}.{ext}' for f in range(1,5)
                for t,ext in [('probabilities','npz'),('metadata','csv.gz')]]
    lock={'at_utc':datetime.now(timezone.utc).isoformat(), 'new_training':0,
          'sources':{str(p.relative_to(ROOT)):e.sha(p) for p in sources}}
    lp=OUT/'INPUT_LOCK.json'
    if lp.exists(): assert json.loads(lp.read_text())['sources']==lock['sources']
    else: e.dump(lp,lock)
    d=pd.read_csv(OLD/'source_analysis_rows.csv.gz',dtype={'location_id':str,'pws_id':str,'state':str})
    md=pd.concat([pd.read_csv(e.NATIVE/f'native_metadata_fold_{f}.csv.gz',dtype={'location_id':str,'pws_id':str,'state':str}) for f in range(1,5)],ignore_index=True)
    ix=pd.Series(np.arange(len(md)),index=md.location_id).loc[d.location_id].to_numpy()
    for c in ['location_id','pws_id','state','observed_state_code','old_visible_code']:
        assert np.array_equal(md.iloc[ix][c],d[c]), c
    models={m:np.concatenate([np.load(e.NATIVE/f'native_probabilities_fold_{f}.npz')[m] for f in range(1,5)])[ix] for m in e.FIXED}
    models={m:q/q.sum(1,keepdims=True) for m,q in models.items()}
    margins=models['factorized']@e.BITS
    anchor=pd.read_csv(CP); cp=np.zeros_like(models['factorized']);pa=[]
    for fold in range(1,5):
        pk=anchor.query('outer_fold == @fold and alpha == 0').set_index('count').smoothed_count_probability.reindex(range(7)).to_numpy()
        assert np.isfinite(pk).all() and abs(pk.sum()-1)<1e-12
        prior=np.array([pk[k]/math.comb(6,int(k)) for k in e.K])
        take=md.iloc[ix].outer_fold.to_numpy()==fold
        cp[take],audit=project(margins[take],prior);pa.append({'fold':fold,**audit})
    models['count_prior']=cp
    basis=np.column_stack([e.BITS,e.K,np.eye(7)[e.K],e.H[:,[0,1,15,11]]])
    # Explicit named pair columns: never rely on incidental pair ordering.
    basis[:,-2]=e.BITS[:,4]*e.BITS[:,5]
    basis[:,-1]=e.BITS[:,2]*e.BITS[:,5]
    labels=e.CHEMS+['mean_count']+['count_'+str(k) for k in range(7)]+['multiple','both_families','PFOA+PFOS','PFHxS+PFOS']
    e.OUT=OUT;e.H=basis;e.ENDPOINTS=labels;e.MAIN_ENDPOINTS=labels
    e.model_pairs=lambda probs:[('factorized',m) for m in models if m!='factorized']
    prob={m:q@basis for m,q in models.items()}
    fixed=['factorized','conditional_shared_shock','conditional_gaussian_factor2','count_prior']
    errors={m:float(abs(models[m]@e.BITS-margins).max()) for m in fixed}
    assert max(errors.values())<2e-8
    for m,q in models.items():
        np.testing.assert_allclose(q@e.K,(q@e.BITS).sum(1),atol=2e-12,rtol=0)
    e.source_analysis('national',d,prob)
    # Reproduce earlier point estimates and all bootstrap draws, not just printed rounding.
    old=np.load(OLD/'source_bootstrap.npz');new=np.load(OUT/'national/source_bootstrap.npz')
    for m in e.FIXED:
        a=list(old['models']).index(m);b=list(new['models']).index(m)
        for lab in old['endpoints']:
            j=list(old['endpoints']).index(lab);k=labels.index(lab)
            np.testing.assert_allclose(old['point'][:,a,j],new['point'][:,b,k],atol=2e-12,rtol=0)
            np.testing.assert_allclose(old['draws'][:,:,a,j],new['draws'][:,:,b,k],atol=2e-12,rtol=0,equal_nan=True)
    rows=pd.read_csv(OUT/'national/source_rates_and_contrasts.csv')
    err=pd.read_csv(OUT/'national/source_reconstruction_errors.csv')
    rate=rows.set_index(['condition','endpoint','quantity']).estimate
    bias=err[err.quantity.isin(['GW_bias','SW_bias','contrast_bias'])].set_index(['condition','endpoint','quantity']).estimate
    for m in models:
        for lab in labels:
            np.testing.assert_allclose(bias[m,lab,'contrast_bias'],bias[m,lab,'SW_bias']-bias[m,lab,'GW_bias'],atol=1e-12,rtol=0)
    result={'completed_at_utc':datetime.now(timezone.utc).isoformat(),'events':len(d),
            'strata':d.stratum.nunique(),'jurisdictions':d.state.nunique(),
            'new_training_runs':0,'test_calibration':False,'fixed_margin_max_error':errors,
            'count_prior_inference_audit':pa,'old_bootstrap_reproduced':True,
            'mean_count_identity_verified':True,'contrast_bias_identity_verified':True,
            'protocol':'post-development diagnostic; pointwise intervals'}
    e.dump(OUT/'VERIFICATION.json',result)
    # Independent pandas reconstruction of standardization, in addition to shared code.
    weights=pd.read_csv(OLD/'standardization_weights.csv').set_index('stratum').weight
    for m,q in models.items():
        for j,chem in enumerate(e.CHEMS):
            z=d[['stratum','water_type']].copy();z['p']=q@e.BITS[:,j]
            v=weights@z.groupby(['stratum','water_type']).p.mean().unstack().loc[weights.index,['GW','SW']]
            for g in ['GW','SW']:assert abs(v[g]-rate[m,chem,g+'_rate'])<2e-12
    # Small numeric outputs for review, preserving units in the CSV.
    tab=[]
    for lab in e.CHEMS+['mean_count']:
        rec={'endpoint':lab,'unit':'reports per event' if lab=='mean_count' else 'percent'}
        scale=1 if lab=='mean_count' else 100
        for g in ['GW','SW']:
            rec[g+'_observed']=scale*rate['measured',lab,g+'_rate']
            rec[g+'_predicted']=scale*rate['factorized',lab,g+'_rate']
            for what,col in [('estimate','bias'),('lower95','bias_lower95'),('upper95','bias_upper95')]:
                r=err[(err.condition=='factorized')&(err.endpoint==lab)&(err.quantity==g+'_bias')].iloc[0]
                rec[g+'_'+col]=scale*r[what]
        tab.append(rec)
    pd.DataFrame(tab).to_csv(OUT/'marginal_diagnostic.csv',index=False)
    print(pd.DataFrame(tab).to_string(index=False),flush=True)
    print(rows.query("endpoint=='multiple'").to_string(index=False),flush=True)

if __name__=='__main__':run()
