"""Same-record environmental contrasts and named-pair observability."""
from itertools import combinations
import numpy as np
import pandas as pd
from research_expansion_common_20260925 import OUT, FIRST, REPS, verify, dump, cluster_weights, ci, read_paired
from prepare_monitoring_followup_20260925 import A25, SIX, MRL

DEST=OUT/'environment'
SCHEMES=['six_old','six_native','twentyfive_native']


def source_comparison():
    full=pd.read_csv(FIRST,dtype={'location_id':str,'PWSID':str,'State':str})
    d=full.rename(columns={'PWSID':'pws_id','State':'state','FacilityWaterType':'water_type','Size':'pws_size'})
    d.groupby(['water_type','pws_size']).size().rename('locations').reset_index().to_csv(DEST/'source_cohort.csv',index=False)
    d=d.loc[d.water_type.isin(['GW','SW'])].copy().reset_index(drop=True)
    cells=d.groupby(['state','pws_size','water_type']).agg(locations=('location_id','size'),pws=('pws_id','nunique')).reset_index()
    support=cells.assign(ok=cells.locations.ge(20)&cells.pws.ge(5)).pivot(index=['state','pws_size'],columns='water_type',values='ok').fillna(False)
    supported=support.GW & support.SW
    allowed=set(supported.index[supported])
    d['cell']=[s+'|'+z for s,z in zip(d.state,d.pws_size)]
    d['common_support']=[(s,z) in allowed for s,z in zip(d.state,d.pws_size)]
    cells['common_support']=[(s,z) in allowed for s,z in zip(cells.state,cells.pws_size)]
    cells.to_csv(DEST/'common_support_cells.csv',index=False)
    cellnames=sorted(d.loc[d.common_support,'cell'].unique())
    assert cellnames
    cellmap={k:i for i,k in enumerate(cellnames)}
    cc=d.cell.map(cellmap).fillna(-1).to_numpy(int)
    sw=d.water_type.eq('SW').to_numpy(int)
    active=cc>=0;cellsource=2*cc[active]+sw[active]
    ncell=len(cellnames)
    n=np.bincount(cc[active],minlength=ncell)
    swt=n/n.sum()
    cols=['k6_old','k6_native','k25_native']
    k=d[cols].to_numpy(float)
    assert np.all(k[:,0]<=k[:,1]) and np.all(k[:,1]<=k[:,2])
    endpoints=np.column_stack([k>=2,k])
    labels=[f'{scheme}__{metric}' for metric in ['multi_rate','mean_count'] for scheme in SCHEMES]

    def statistic(w):
        denom=np.bincount(sw,weights=w,minlength=2)
        raw=np.stack([np.bincount(sw,weights=w*endpoints[:,j],minlength=2)/denom for j in range(6)])
        den=np.bincount(cellsource,weights=w[active],minlength=2*ncell).reshape(ncell,2)
        std=np.full((6,2),np.nan)
        if np.all(den>0):
            for j in range(6):
                numerator=np.bincount(cellsource,weights=w[active]*endpoints[active,j],minlength=2*ncell).reshape(ncell,2)
                std[j]=(swt[:,None]*numerator/den).sum(axis=0)
        return np.stack([raw,std])

    point=statistic(np.ones(len(d)))
    draws=np.zeros((REPS,2,6,2))
    for r,w in enumerate(cluster_weights(d)):
        draws[r]=statistic(w)
        if (r+1)%500==0:print('Environmental contrast resamples',r+1,flush=True)
    rows=[];contrasts=[]
    for mode,estimand in enumerate(['unstandardized','common_support_standardized']):
        for j,label in enumerate(labels):
            scheme,metric=label.split('__')
            for s,source in enumerate(['GW','SW']):
                rows.append({'estimand':estimand,'scheme':scheme,'metric':metric,'source':source,
                             'estimate':point[mode,j,s],**ci(draws[:,mode,j,s])})
            delta=draws[:,mode,j,1]-draws[:,mode,j,0]
            contrasts.append({'estimand':estimand,'comparison':'SW_minus_GW','scheme':scheme,'metric':metric,
                              'estimate':point[mode,j,1]-point[mode,j,0],**ci(delta)})
        for offset,metric in [(0,'multi_rate'),(3,'mean_count')]:
            for before,after,name in [(0,1,'lower_six_cutoffs'),(1,2,'expand_native_panel')]:
                deltas=draws[:,mode,:,1]-draws[:,mode,:,0]
                pts=point[mode,:,1]-point[mode,:,0]
                contrasts.append({'estimand':estimand,'comparison':name,'scheme':'difference_in_source_contrasts',
                                  'metric':metric,'estimate':pts[offset+after]-pts[offset+before],
                                  **ci(deltas[:,offset+after]-deltas[:,offset+before])})
    pd.DataFrame(rows).to_csv(DEST/'source_rates.csv',index=False)
    pd.DataFrame(contrasts).to_csv(DEST/'source_contrasts.csv',index=False)
    pd.DataFrame({'cell':cellnames,'locations':n,'standardization_weight':swt}).to_csv(DEST/'standardization_weights.csv',index=False)
    d[['location_id','pws_id','state','pws_size','water_type','common_support',*cols]].to_csv(DEST/'source_analysis_records.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    np.savez_compressed(DEST/'source_bootstrap.npz',draws=draws,labels=labels)
    # Independent long-form calculation of standardized point rates.
    for j in range(6):
        tmp=d.loc[active,['cell','water_type']].copy();tmp['outcome']=endpoints[active,j]
        g=tmp.groupby(['cell','water_type']).outcome.mean().unstack().loc[cellnames,['GW','SW']]
        assert np.max(np.abs(swt@g.to_numpy()-point[1,j]))<1e-12
    out={'full_first_sample_locations':len(full),'GW_SW_locations':len(d),'common_support_locations':int(active.sum()),
         'common_support_cells':ncell,'excluded_from_source_contrast_other_types':len(full)-len(d),
         'excluded_GW_SW_outside_support':int((~active).sum()),
         'finite_standardized_resamples':int(np.isfinite(draws[:,1]).all(axis=(1,2)).sum()),
         'independent_standardization_replay':True,'same_records_all_schemes':True}
    print(pd.DataFrame(contrasts).query("estimand=='common_support_standardized'").to_string(index=False),flush=True)
    return out


def named_pairs():
    d=read_paired(); pairs=list(combinations(range(25),2))
    c1=d[[f'first_conc_ugL__{a}' for a in A25]].to_numpy(float)
    c2=d[[f'next_conc_ugL__{a}' for a in A25]].to_numpy(float)
    mrl=np.array([MRL[a] for a in A25])
    reports={}
    for multiple in [1,2,5]:
        a=np.isfinite(c1)&(c1>=multiple*mrl-1e-12)
        b=np.isfinite(c2)&(c2>=multiple*mrl-1e-12)
        reports[multiple]=(a,b)
    rows=[]
    pfcAs={'PFBA','PFPeA','PFHxA','PFHpA','PFOA','PFNA','PFDA','PFUnA','PFDoA'}
    pfsAs={'PFBS','PFPeS','PFHxS','PFHpS','PFOS'}
    family=lambda a:'carboxylate' if a in pfcAs else ('sulfonate' if a in pfsAs else 'other')
    for i,j in pairs:
        first=reports[1][0][:,i]&reports[1][0][:,j]
        nxt=reports[1][1][:,i]&reports[1][1][:,j]
        n=int(first.sum());both=int((first&nxt).sum())
        states=d.loc[first,'state'].value_counts()
        rec=[]
        for state in states.index:
            mask=first&d.state.ne(state).to_numpy();den=int(mask.sum())
            if den:rec.append(float((mask&nxt).sum()/den))
        row={'pair':A25[i]+' + '+A25[j],'analyte_a':A25[i],'analyte_b':A25[j],
             'family_a':family(A25[i]),'family_b':family(A25[j]),
             'both_in_shared_six':int(A25[i] in SIX and A25[j] in SIX),
             'initial_support':n,'jurisdiction_support':len(states),'repeat_support':both,
             'repeat_fraction':both/n if n else np.nan,
             'max_jurisdiction_support_fraction':float(states.max()/n) if n else np.nan,
             'leave_one_jurisdiction_out_repeat_min':min(rec) if rec else np.nan,
             'leave_one_jurisdiction_out_repeat_max':max(rec) if rec else np.nan,
             'display_eligible':int(n>=100 and len(states)>=5),
             'median_lag_days_initial_positive':float(d.loc[first,'lag_days'].median()) if n else np.nan}
        for mult in [2,5]:
            aa,bb=reports[mult];f=aa[:,i]&aa[:,j];nn=bb[:,i]&bb[:,j]
            assert not (f&~first).any()
            row[f'initial_retention_{mult}x']=int(f.sum())/n if n else np.nan
            row[f'both_dates_retained_{mult}x']=int((f&nn).sum())/n if n else np.nan
        rows.append(row)
    res=pd.DataFrame(rows).sort_values(['initial_support','pair'],ascending=[False,True])
    res.to_csv(DEST/'all_pair_observability.csv',index=False)
    res.loc[res.display_eligible.eq(1)].to_csv(DEST/'display_pair_observability.csv',index=False)
    assert len(res)==300
    return {'all_pairs':300,'display_pairs':int(res.display_eligible.sum()),'selection_uses_only_initial_support':True,
            'retention_denominator':'first sample native-limit co-reports','no_cross_date_pseudo_pairs':True}


if __name__=='__main__':
    verify();DEST.mkdir(exist_ok=True)
    a=source_comparison();b=named_pairs()
    dump(DEST/'AUDIT.json',{'status':'PASS','source':a,'pairs':b})
    verify()
