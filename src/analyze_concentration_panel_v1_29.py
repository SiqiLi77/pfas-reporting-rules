"""Locked post-result concentration-resolution sensitivity; no model fitting."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/est_competitiveness_v1_29_20260925'
DATA = OUT / 'analysis'
from prepare_monitoring_followup_20260925 import ARCHIVE, A25, SIX, MRL
from research_expansion_common_20260925 import FIRST, cluster_weights, ci
WEIGHTS = ROOT / 'outputs/research_expansion_20260925/environment/standardization_weights.csv'
REFERENCE = ROOT / 'outputs/research_expansion_20260925/environment/source_rates.csv'
PAIRED = ROOT / 'outputs/observation_design_20260925/cohort/paired_panels.csv.gz'
FLOORS = [0, 5, 10, 20, 40]
KEY = ['PWSID', 'FacilityID', 'SamplePointID', 'CollectionDate', 'SampleID']

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(p, o):
    p.write_text(json.dumps(o, indent=2, ensure_ascii=False, allow_nan=False)+'\n')

def verify():
    for path, h in json.loads((OUT/'ANALYSIS_LOCK.json').read_text())['sources'].items():
        assert sha(ROOT/path) == h, path

def freeze():
    p=OUT/'ANALYSIS_LOCK.json'
    if p.exists(): raise FileExistsError('Preserve the existing analysis lock')
    sources=[ARCHIVE,FIRST,WEIGHTS,REFERENCE,PAIRED,Path(__file__),OUT/'CONCENTRATION_PROTOCOL.md',
             ROOT/'src/prepare_monitoring_followup_20260925.py', ROOT/'src/research_expansion_common_20260925.py']
    dump(p, {'locked_at_utc':datetime.now(timezone.utc).isoformat(),'post_result_sensitivity':True,
             'floors_ng_L':FLOORS,'bootstrap_reps':2000,'seed':202609253,
             'sources':{str(x.relative_to(ROOT)):sha(x) for x in sources}})

def prepare():
    verify(); DATA.mkdir(exist_ok=True)
    ref=pd.read_csv(FIRST,dtype=str,keep_default_na=False)
    assert len(ref)==26423 and ref.location_id.is_unique
    weights=pd.read_csv(WEIGHTS)
    d=ref.copy();d['cell']=d.State+'|'+d.Size
    d=d[d.cell.isin(weights.cell)&d.FacilityWaterType.isin(['GW','SW'])].copy().reset_index(drop=True)
    assert len(d)==20162
    lookup=pd.MultiIndex.from_frame(d[KEY]);parts=[]
    cols=KEY+['Contaminant','SampleEventCode','AnalyticalResultsSign','AnalyticalResultValue','Units','MRL','MethodID']
    with zipfile.ZipFile(ARCHIVE) as z, z.open('UCMR5_All.txt') as f:
        for ch in pd.read_csv(f,sep='\t',encoding='latin-1',dtype=str,keep_default_na=False,usecols=cols,chunksize=200000):
            ch=ch[ch.SampleEventCode.eq('SE1')&ch.Contaminant.isin(A25)]
            parts.append(ch[pd.MultiIndex.from_frame(ch[KEY]).isin(lookup)].copy())
    raw=pd.concat(parts,ignore_index=True)
    assert len(raw)==25*len(d) and not raw.duplicated(KEY+['Contaminant']).any()
    assert raw.MethodID.eq('EPA 533').all() and raw.Units.eq('µg/L').all()
    assert raw.AnalyticalResultsSign.isin(['=','<']).all()
    assert np.allclose(pd.to_numeric(raw.MRL),raw.Contaminant.map(MRL),rtol=0,atol=1e-12)
    raw['reported']=raw.AnalyticalResultsSign.eq('=').astype(int)
    raw['concentration']=pd.to_numeric(raw.AnalyticalResultValue.where(raw.reported.eq(1)),errors='coerce')
    assert raw.loc[raw.reported.eq(1),'concentration'].notna().all()
    assert (raw.loc[raw.reported.eq(1),'concentration']>=raw.loc[raw.reported.eq(1),'Contaminant'].map(MRL)-1e-12).all()
    values=raw.pivot(index=KEY,columns='Contaminant',values='concentration').reindex(lookup)[list(A25)].to_numpy(float)
    flags=raw.pivot(index=KEY,columns='Contaminant',values='reported').reindex(lookup)[list(A25)].to_numpy(int)
    assert np.array_equal(flags,d[[f'report__{a}' for a in A25]].to_numpy(int))
    assert np.array_equal(np.isfinite(values),flags.astype(bool))
    for j,a in enumerate(A25):d[f'conc_ugL__{a}']=values[:,j]
    paired=pd.read_csv(PAIRED,dtype={'location_id':str}).set_index('location_id')
    common=d.location_id.isin(paired.index)
    old=paired.loc[d.loc[common,'location_id'],[f'first_conc_ugL__{a}' for a in A25]].to_numpy(float)
    assert np.allclose(values[common],old,atol=0,rtol=0,equal_nan=True)
    d.to_csv(DATA/'same_record_concentrations.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    dump(DATA/'PREPARATION_AUDIT.json',{'status':'PASS','locations':len(d),'PWSs':d.PWSID.nunique(),
        'jurisdictions':d.State.nunique(),'strata':d.cell.nunique(),'raw_rows':len(raw),
        'paired_first_sample_concentrations_exact_locations':int(common.sum()),
        'nondetect_concentrations':'NaN with preserved reporting mask, not zero',
        'all_25_original_masks_exact':True,'full_sample_key_used':KEY})
    verify()

def analyze():
    verify()
    d=pd.read_csv(DATA/'same_record_concentrations.csv.gz',dtype={'location_id':str,'PWSID':str,'State':str,'Size':str,'cell':str})
    d=d.rename(columns={'PWSID':'pws_id','State':'state'})
    ws=pd.read_csv(WEIGHTS);cells=ws.cell.tolist();w=ws.standardization_weight.to_numpy()
    cc=d.cell.map({v:i for i,v in enumerate(cells)}).to_numpy(int)
    sw=d.FacilityWaterType.eq('SW').to_numpy(int);code=2*cc+sw
    conc=d[[f'conc_ugL__{a}' for a in A25]].to_numpy(float)
    limits=np.array([MRL[a] for a in A25]);six=[A25.index(a) for a in SIX]
    outcomes=[];counts=[]
    for floor in FLOORS:
        cutoff=np.maximum(limits,floor/1000)
        flag=np.isfinite(conc)&(conc>=cutoff-1e-12)
        k6=flag[:,six].sum(axis=1);k25=flag.sum(axis=1)
        assert np.all(k6<=k25)
        counts.append(np.stack([k6,k25],axis=1));outcomes.extend([k6>=2,k25>=2])
    counts=np.stack(counts,axis=1)
    assert np.all(np.diff(counts,axis=1)<=0)
    y=np.array(outcomes,dtype=float).T
    def statistic(mult):
        den=np.bincount(code,weights=mult,minlength=2*len(cells)).reshape(-1,2)
        if np.any(den==0):return np.full((len(FLOORS),2,2),np.nan)
        ans=[]
        for j in range(y.shape[1]):
            num=np.bincount(code,weights=mult*y[:,j],minlength=2*len(cells)).reshape(-1,2)
            ans.append((w[:,None]*num/den).sum(axis=0))
        return np.array(ans).reshape(len(FLOORS),2,2)
    point=statistic(np.ones(len(d)))
    ref=pd.read_csv(REFERENCE).query("estimand=='common_support_standardized' and metric=='multi_rate'")
    for p,panel in enumerate(['six_native','twentyfive_native']):
        for s,source in enumerate(['GW','SW']):
            expected=ref.query('scheme==@panel and source==@source').iloc[0].estimate
            assert abs(point[0,p,s]-expected)<1e-12
    # Independent point calculation by grouped dataframe means.
    maxerr=0.
    for j in range(y.shape[1]):
        z=d[['cell','FacilityWaterType']].copy();z['value']=y[:,j]
        tab=z.groupby(['cell','FacilityWaterType']).value.mean().unstack().loc[cells,['GW','SW']]
        maxerr=max(maxerr,float(np.max(np.abs(w@tab.to_numpy()-point.reshape(-1,2)[j]))))
    assert maxerr<1e-12
    draws=np.stack([statistic(a) for a in cluster_weights(d)])
    finite=np.isfinite(draws).all(axis=(1,2,3));bd=draws[finite]
    assert len(bd)>1900
    delta=point[:,:,1]-point[:,:,0];dd=bd[:,:,:,1]-bd[:,:,:,0]
    effect=delta[:,1]-delta[:,0];de=dd[:,:,1]-dd[:,:,0]
    increment=point[:,1]-point[:,0];di=bd[:,:,1]-bd[:,:,0]
    assert np.max(abs(effect-(increment[:,1]-increment[:,0])))<1e-12
    sd=de[:,1:].std(axis=0,ddof=1)
    standardized=np.divide(abs(de[:,1:]-effect[None,1:]),sd[None,:],out=np.zeros_like(de[:,1:]),where=sd[None,:]>0)
    critical=float(np.quantile(standardized.max(axis=1),.95))
    rows=[];contrasts=[];origin=[]
    for f,floor in enumerate(FLOORS):
        for p,panel in enumerate(['shared_six','complete_25']):
            for s,source in enumerate(['GW','SW']):
                rows.append({'floor_ng_L':floor,'condition':'native' if floor==0 else f'floor_{floor}',
                             'panel':panel,'source':source,'estimate':point[f,p,s],**ci(bd[:,f,p,s])})
            contrasts.append({'floor_ng_L':floor,'comparison':f'{panel}_SW_minus_GW','estimate':delta[f,p],**ci(dd[:,f,p])})
        rr={'floor_ng_L':floor,'comparison':'panel_induced_source_contrast_change','estimate':effect[f],**ci(de[:,f])}
        if f:rr.update(simultaneous_lower=float(effect[f]-critical*sd[f-1]),simultaneous_upper=float(effect[f]+critical*sd[f-1]))
        contrasts.append(rr)
        for s,source in enumerate(['GW','SW']):
            contrasts.append({'floor_ng_L':floor,'comparison':f'panel_increment_{source}','estimate':increment[f,s],**ci(di[:,f,s])})
        origin.append({'floor_ng_L':floor,'estimate_pp':effect[f]*100,
            'ci_low_pp':rr['ci95_lower']*100,'ci_high_pp':rr['ci95_upper']*100,
            'x_error_minus':(effect[f]-rr['ci95_lower'])*100,'x_error_plus':(rr['ci95_upper']-effect[f])*100,
            'simultaneous_low_pp':rr.get('simultaneous_lower',np.nan)*100,
            'simultaneous_high_pp':rr.get('simultaneous_upper',np.nan)*100})
    comparison={'comparison':'native_minus_common_20_panel_effect','estimate':float(effect[0]-effect[3]),**ci(de[:,0]-de[:,3])}
    pd.DataFrame(rows).to_csv(DATA/'source_rates_by_floor.csv',index=False)
    pd.DataFrame(contrasts).to_csv(DATA/'contrasts_by_floor.csv',index=False)
    pd.DataFrame(origin).to_csv(DATA/'origin_panel_effect_by_floor.csv',index=False)
    pd.DataFrame([comparison]).to_csv(DATA/'primary_resolution_contrast.csv',index=False)
    np.savez_compressed(DATA/'bootstrap.npz',rates=draws,floors=np.array(FLOORS),point=point)
    dump(DATA/'RESULT_AUDIT.json',{'status':'PASS','post_result_sensitivity':True,'bootstrap_reps':len(draws),
        'finite_draws':int(finite.sum()),'grouped_point_replay_max_abs_error':maxerr,
        'native_rates_match_v1_28':True,'nested_panels_and_monotone_count_path':True,
        'paired_increment_identity':True,'simultaneous_band_critical':critical,
        'equal_cutoff_conditions_ng_L':[20,40],'common_floor_not_equal_at':[5,10],
        'primary_resolution_contrast':comparison,
        'sources_after_analysis':{str(p.relative_to(ROOT)):sha(p) for p in [ARCHIVE,FIRST,WEIGHTS]}})
    verify()
    print(pd.DataFrame(contrasts).query("comparison=='panel_induced_source_contrast_change'").to_string(index=False))
    print(json.dumps(comparison,indent=2))

if __name__=='__main__':
    {'freeze':freeze,'prepare':prepare,'analyze':analyze}[sys.argv[1]]()
