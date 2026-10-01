"""Exact added-panel attribution; explicitly post-result explanatory analysis."""
from itertools import combinations
from math import factorial
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
from research_expansion_common_20260925 import ROOT, OUT, FIRST, REPS, verify, dump, sha, cluster_weights, ci
from prepare_monitoring_followup_20260925 import A25, SIX

DEST=OUT/'panel_attribution'

def verify_games():
    errors=[]
    for n in range(1,7):
        for base in range(3):
            for m in range(n+1):
                z=np.arange(n)<m
                value=lambda mask:int(base+sum(z[j] for j in mask)>=2)-int(base>=2)
                exact=[]
                for j in range(n):
                    others=[k for k in range(n) if k!=j];val=0.
                    for size in range(n):
                        w=factorial(size)*factorial(n-size-1)/factorial(n)
                        for subset in combinations(others,size):
                            val+=w*(value((*subset,j))-value(subset))
                    exact.append(val)
                formula=z*value(range(n))/max(m,1)
                errors.append(np.max(np.abs(formula-exact)))
    assert max(errors)<1e-12
    return float(max(errors))

def main():
    verify();DEST.mkdir(exist_ok=True)
    lock=OUT/'ATTRIBUTION_LOCK.json'
    if lock.exists():
        for name,h in json.loads(lock.read_text())['sources'].items():assert sha(ROOT/name)==h
    else:
        files=[OUT/'ATTRIBUTION_PROTOCOL.md',Path(__file__),FIRST,
               OUT/'environment/standardization_weights.csv',OUT/'environment/common_support_cells.csv']
        dump(lock,{'locked_at_utc':datetime.now(timezone.utc).isoformat(),
                   'post_result_explanatory':True,'sources':{str(p.relative_to(ROOT)):sha(p) for p in files}})
    d=pd.read_csv(FIRST,dtype={'PWSID':str,'State':str,'location_id':str}).rename(columns={'PWSID':'pws_id','State':'state','Size':'pws_size','FacilityWaterType':'water_type'})
    d=d[d.water_type.isin(['GW','SW'])].reset_index(drop=True)
    d['cell']=d.state+'|'+d.pws_size
    swt=pd.read_csv(OUT/'environment/standardization_weights.csv')
    cells=swt.cell.tolist();lookup={v:i for i,v in enumerate(cells)}
    idx=d.cell.map(lookup).fillna(-1).to_numpy(int);active=idx>=0
    src=d.water_type.eq('SW').to_numpy(int);code=2*idx[active]+src[active]
    wcell=swt.standardization_weight.to_numpy();ncell=len(cells)
    extra=[a for a in A25 if a not in SIX]
    report=d[[f'report__{a}' for a in extra]].to_numpy(float)
    k6=d.k6_native.to_numpy();k25=d.k25_native.to_numpy()
    assert np.array_equal(report.sum(axis=1),k25-k6)
    gained=(k6<2)&(k25>=2)
    attrib=report*gained[:,None]/np.maximum(report.sum(axis=1,keepdims=True),1)
    dropout=(k25[:,None]>=2)&(k25[:,None]-report<2)
    assert np.max(abs(attrib.sum(axis=1)-gained))<1e-12
    outcomes=np.column_stack([attrib,dropout])
    def statistic(w):
        den=np.bincount(code,weights=w[active],minlength=2*ncell).reshape(ncell,2)
        val=np.full((38,2),np.nan)
        if np.all(den>0):
            for j in range(38):
                num=np.bincount(code,weights=w[active]*outcomes[active,j],minlength=2*ncell).reshape(ncell,2)
                val[j]=(wcell[:,None]*num/den).sum(axis=0)
        return val
    point=statistic(np.ones(len(d)));draws=np.empty((REPS,38,2))
    for r,w in enumerate(cluster_weights(d)):
        draws[r]=statistic(w)
        if (r+1)%500==0:print('Attribution resamples',r+1,flush=True)
    rows=[]
    for metric,offset in [('shapley',0),('leave_one_analyte_out',19)]:
        for j,a in enumerate(extra):
            v=point[offset+j];dist=draws[:,offset+j,1]-draws[:,offset+j,0]
            rows.append({'metric':metric,'analyte':a,'GW_contribution':v[0],'SW_contribution':v[1],
                         'SW_minus_GW':v[1]-v[0],**ci(dist)})
    pd.DataFrame(rows).to_csv(DEST/'analyte_contributions.csv',index=False)
    totals=[]
    for name,names in [('all_19',extra),('PFBA_PFPeA_PFHxA',['PFBA','PFPeA','PFHxA'])]:
        indices=[extra.index(a) for a in names]
        p=point[indices].sum(axis=0);b=draws[:,indices].sum(axis=1)
        totals.append({'group':name,'GW_contribution':p[0],'SW_contribution':p[1],
                       'SW_minus_GW':p[1]-p[0],**ci(b[:,1]-b[:,0])})
    pd.DataFrame(totals).to_csv(DEST/'combined_contributions.csv',index=False)
    np.savez_compressed(DEST/'bootstrap.npz',draws=draws,analytes=extra)
    expected=pd.read_csv(OUT/'environment/source_contrasts.csv').query("estimand=='common_support_standardized' and comparison=='expand_native_panel' and metric=='multi_rate'").iloc[0]
    residual=float(abs(totals[0]['SW_minus_GW']-expected.estimate));assert residual<1e-12
    old=np.load(OUT/'environment/source_bootstrap.npz')['draws'][:,1,:3]
    olddelta=(old[:,2,1]-old[:,2,0])-(old[:,1,1]-old[:,1,0])
    delta=draws[:,:19,1].sum(axis=1)-draws[:,:19,0].sum(axis=1)
    assert np.array_equal(np.isfinite(delta),np.isfinite(olddelta))
    assert np.nanmax(abs(delta-olddelta))<1e-12
    dump(DEST/'AUDIT.json',{'status':'PASS','small_game_enumeration_max_error':verify_games(),
                          'point_efficiency_error':residual,'resample_efficiency_max_error':float(np.nanmax(abs(delta-olddelta))),
                          'all_19_additional_analytes_retained':True,'causal_attribution':False,
                          'post_result_explanatory':True,'finite_draws':int(np.isfinite(delta).sum())})
    print(pd.DataFrame(rows).query("metric=='shapley'").sort_values('SW_minus_GW',ascending=False).to_string(index=False),flush=True)
    print(pd.DataFrame(totals).to_string(index=False),flush=True)
    verify()

if __name__=='__main__':
    from pathlib import Path
    main()
