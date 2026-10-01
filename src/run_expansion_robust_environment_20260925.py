"""Post-result source-weighting/influence checks and complete policy intervals."""
import json
import numpy as np
import pandas as pd
from research_expansion_common_20260925 import OUT,FIRST,REPS,cluster_weights,ci,dump
from prepare_monitoring_followup_20260925 import A25,SIX
from run_expansion_robust_forecast_20260925 import check

DEST=OUT/'environment_robustness'


def run():
    check();DEST.mkdir(exist_ok=True)
    d=pd.read_csv(OUT/'environment/source_analysis_records.csv.gz',dtype={'location_id':str,'pws_id':str,'state':str})
    d=d.loc[d.common_support.eq(True)].copy().reset_index(drop=True)
    weights=pd.read_csv(OUT/'environment/standardization_weights.csv')
    names=weights.cell.tolist();lookup={v:i for i,v in enumerate(names)}
    cell=np.array([lookup[s+'|'+z] for s,z in zip(d.state,d.pws_size)])
    sw=d.water_type.eq('SW').to_numpy(int);cs=2*cell+sw;nc=len(names)
    fixed=weights.standardization_weight.to_numpy(float)
    vals=(d[['k6_old','k6_native','k25_native']].to_numpy()>=2).astype(float)
    within=d.groupby(['state','pws_size','water_type','pws_id']).location_id.transform('size').to_numpy(float)
    pwsweight=1/within

    def cellmeans(w,value):
        den=np.bincount(cs,weights=w,minlength=2*nc).reshape(nc,2)
        result=np.full((nc,2,value.shape[1]),np.nan)
        if np.all(den>0):
            for j in range(value.shape[1]):
                result[:,:,j]=np.bincount(cs,weights=w*value[:,j],minlength=2*nc).reshape(nc,2)/den
        return result

    primary=cellmeans(np.ones(len(d)),vals)
    original_rates=pd.read_csv(OUT/'environment/source_rates.csv')
    ppoint=(fixed[:,None,None]*primary).sum(axis=0)
    for j,scheme in enumerate(['six_old','six_native','twentyfive_native']):
        for s,source in enumerate(['GW','SW']):
            expected=original_rates.query('estimand=="common_support_standardized" and metric=="multi_rate" and scheme==@scheme and source==@source').estimate.iloc[0]
            assert abs(expected-ppoint[s,j])<1e-12
    point=(fixed[:,None,None]*cellmeans(pwsweight,vals)).sum(axis=0)
    draws=np.zeros((REPS,2,3))
    for r,w in enumerate(cluster_weights(d)):
        draws[r]=(fixed[:,None,None]*cellmeans(w*pwsweight,vals)).sum(axis=0)
    rows=[]
    for j,scheme in enumerate(['six_old','six_native','twentyfive_native']):
        delta=draws[:,1,j]-draws[:,0,j]
        rows.append({'scheme':scheme,'metric':'SW_minus_GW_multi_rate','estimate':point[1,j]-point[0,j],**ci(delta)})
    dd=draws[:,1]-draws[:,0];delta=point[1]-point[0]
    for a,b,name in [(0,1,'lower_six_cutoffs'),(1,2,'expand_native_panel')]:
        rows.append({'scheme':name,'metric':'change_in_source_contrast','estimate':delta[b]-delta[a],**ci(dd[:,b]-dd[:,a])})
    pd.DataFrame(rows).to_csv(DEST/'pws_equal_source_contrasts.csv',index=False)
    loo=[];cellstate=np.array([k.split('|')[0] for k in names])
    for state in sorted(set(cellstate)):
        keep=cellstate!=state;ww=fixed[keep]/fixed[keep].sum()
        rates=(ww[:,None,None]*primary[keep]).sum(axis=0)
        source_delta=rates[1]-rates[0]
        loo.append({'excluded_jurisdiction':state,'six_old':source_delta[0],'six_native':source_delta[1],
                    'twentyfive_native':source_delta[2],'panel_change_in_contrast':source_delta[2]-source_delta[1]})
    pd.DataFrame(loo).to_csv(DEST/'leave_one_jurisdiction_out.csv',index=False)
    f=pd.read_csv(FIRST,dtype={'location_id':str}).set_index('location_id').loc[d.location_id]
    add=[a for a in A25 if a not in SIX]
    zero=(d.k6_native.eq(0)&d.k25_native.ge(2)).to_numpy(float)
    one=(d.k6_native.eq(1)&d.k25_native.ge(2)).to_numpy(float)
    assert np.array_equal(zero+one,vals[:,2]-vals[:,1])
    constituents=np.column_stack([zero,one]+[f['report__'+a].to_numpy(float) for a in add])
    component_names=['six_zero_to_25_multiple','six_one_to_25_multiple']+add
    means=(fixed[:,None,None]*cellmeans(np.ones(len(d)),constituents)).sum(axis=0)
    expansion=float((means[1,:2]-means[0,:2]).sum())
    assert abs(expansion-((ppoint[1,2]-ppoint[0,2])-(ppoint[1,1]-ppoint[0,1])))<1e-12
    explanation=[{'component':name,'GW_rate':means[0,j],'SW_rate':means[1,j],
                  'SW_minus_GW':means[1,j]-means[0,j],
                  'role':'exact_disjoint_expansion_component' if j<2 else 'additional_analyte_report_frequency'} for j,name in enumerate(component_names)]
    pd.DataFrame(explanation).to_csv(DEST/'panel_expansion_explanation.csv',index=False)
    # All predeclared policy-floor contrasts, using the existing paired draws.
    b=np.load(OUT/'allocation/paired_bootstrap.npz');bname=b['names'].tolist();fields=b['fields'].tolist();boot=b['draws']
    policy=pd.read_csv(OUT/'allocation/policy_results.csv').query('evaluation=="blocked" and budget==0.2').set_index('policy')
    policyrows=[]
    for fraction in [.25,.5,.75,1.]:
        for mode in ['random','regression']:
            candidate=f'floor{fraction:g}_{mode}'
            for metric in ['overall_recall','initial_zero_recall','yield']:
                c=bname.index(candidate);a=bname.index('initial_count');j=fields.index(metric)
                policyrows.append({'candidate':candidate,'reference':'initial_count','metric':metric,
                                   'difference':policy.loc[candidate,metric]-policy.loc['initial_count',metric],**ci(boot[:,c,j]-boot[:,a,j])})
    pd.DataFrame(policyrows).to_csv(DEST/'all_floor_policy_contrasts.csv',index=False)
    dump(DEST/'AUDIT.json',{'status':'PASS','original_source_estimates_replayed':True,
                           'PWS_equal_within_source_and_stratum':True,'standardization_weights_unchanged':True,
                           'panel_expansion_decomposition_exact':True,'all_additional_analytes_retained':len(add),
                           'finite_PWS_equal_bootstrap_replicates':int(np.isfinite(draws).all(axis=(1,2)).sum()),
                           'post_result_sensitivity_not_confirmation':True})
    check();print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':run()
