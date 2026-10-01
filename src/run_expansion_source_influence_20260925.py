"""Focused weighting checks after inspecting the complete leave-jurisdiction-out table."""
from pathlib import Path
from datetime import datetime,timezone
import json
import numpy as np
import pandas as pd
from research_expansion_common_20260925 import ROOT,OUT,REPS,sha,dump,verify,cluster_weights,ci

def main():
    verify();dest=OUT/'source_influence';dest.mkdir(exist_ok=True)
    lock=OUT/'INFLUENCE_LOCK.json'
    files=[Path(__file__),OUT/'INFLUENCE_PROTOCOL.md',OUT/'environment/source_analysis_records.csv.gz',OUT/'environment/standardization_weights.csv']
    if lock.exists():
        for name,h in json.loads(lock.read_text())['sources'].items():assert sha(ROOT/name)==h
    else:dump(lock,{'locked_at_utc':datetime.now(timezone.utc).isoformat(),'post_result':True,'sources':{str(p.relative_to(ROOT)):sha(p) for p in files}})
    d=pd.read_csv(files[2],dtype={'pws_id':str,'state':str,'location_id':str})
    t=pd.read_csv(files[3]);states=t.cell.str.split('|').str[0]
    assert states.nunique()==34
    weights={'primary':t.standardization_weight.to_numpy(),
             'without_TX':np.where(states=='TX',0,t.standardization_weight),
             'jurisdiction_equal':(t.standardization_weight/t.groupby(states).standardization_weight.transform('sum')/states.nunique()).to_numpy()}
    weights={k:v/np.sum(v) for k,v in weights.items()}
    lookup={s:i for i,s in enumerate(t.cell)};c=(d.state+'|'+d.pws_size).map(lookup).fillna(-1).to_numpy(int)
    use=c>=0;sw=d.water_type.eq('SW').to_numpy(int);code=2*c[use]+sw[use];nc=len(t)
    y=d[['k6_old','k6_native','k25_native']].to_numpy()>=2
    def stat(w):
        den=np.bincount(code,weights=w[use],minlength=2*nc).reshape(nc,2)
        rates=np.zeros((3,nc,2))
        for j in range(3):
            num=np.bincount(code,weights=w[use]*y[use,j],minlength=2*nc).reshape(nc,2)
            rates[j]=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
        results=[]
        for ww in weights.values():
            active=ww>0
            dif=(rates[:,active,1]-rates[:,active,0])@ww[active]
            results.append(np.r_[dif,dif[2]-dif[1]])
        return np.array(results)
    point=stat(np.ones(len(d)));draws=np.zeros((REPS,3,4))
    for b,w in enumerate(cluster_weights(d)):
        draws[b]=stat(w)
        if (b+1)%500==0:print('Influence resamples',b+1,flush=True)
    rows=[]
    for i,mode in enumerate(weights):
        for j,metric in enumerate(['six_old','six_native','twentyfive_native','panel_change_in_contrast']):
            rows.append({'weighting':mode,'metric':metric,'estimate':point[i,j],**ci(draws[:,i,j])})
    pd.DataFrame(rows).to_csv(dest/'contrasts.csv',index=False)
    for mode,w in weights.items():t['weight__'+mode]=w
    t.to_csv(dest/'weights.csv',index=False)
    np.savez_compressed(dest/'bootstrap.npz',draws=draws)
    old=pd.read_csv(OUT/'environment/source_contrasts.csv').query("estimand=='common_support_standardized' and metric=='multi_rate' and comparison=='SW_minus_GW'")
    assert np.max(abs(old.estimate.to_numpy()-point[0,:3]))<1e-12
    dump(dest/'AUDIT.json',{'status':'PASS','same_fixed_support':True,'jurisdictions':34,
                           'TX_primary_weight':float(weights['primary'][states=='TX'].sum()),
                           'all_results_retained':True,'post_result_influence_check':True})
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    verify()

if __name__=='__main__':main()
