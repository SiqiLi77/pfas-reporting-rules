"""Prepare an exact same-sample, six-target/19-auxiliary reconstruction cohort."""
from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/auxiliary_information_20260926'
CACHE = ROOT/'work/ucmr5_final_20260828_r1/ucmr5_artificial_censor_strict.csv.gz'
REF = ROOT/'outputs/significance_extension_20260924/panel_coverage/strict25_event_records.csv.gz'
ARCHIVE = ROOT/'data/ucmr_raw/ucmr5-occurrence-data_final_20260828.zip'
SIX = ['PFBS','PFHpA','PFHxS','PFNA','PFOA','PFOS']
MRL = {'11Cl-PF3OUdS':.005,'4:2 FTS':.003,'6:2 FTS':.005,'8:2 FTS':.005,
       '9Cl-PF3ONS':.002,'ADONA':.003,'HFPO-DA':.005,'NFDHA':.020,'PFBA':.005,
       'PFBS':.003,'PFDA':.003,'PFDoA':.003,'PFEESA':.003,'PFHpA':.003,
       'PFHpS':.003,'PFHxA':.003,'PFHxS':.003,'PFMBA':.003,'PFMPA':.004,
       'PFNA':.004,'PFOA':.004,'PFOS':.004,'PFPeA':.003,'PFPeS':.004,'PFUnA':.002}
OLD = np.array([.09,.01,.03,.02,.02,.04])

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def dump(p,x):
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def prepare():
    assert not (OUT/'INPUT_LOCK.json').exists(), 'Do not overwrite a frozen experiment'
    d=pd.read_csv(CACHE,dtype={'location_id':str,'pws_id':str,'state':str,'region':str})
    r=pd.read_csv(REF,dtype=str,keep_default_na=False).set_index('location_id')
    d=d.loc[d.location_id.isin(r.index)].sort_values('location_id').reset_index(drop=True)
    assert len(d)==26423 and d.location_id.is_unique
    assert set(d.location_id)==set(r.index)
    records=[]
    cols=['PWSID','FacilityID','SamplePointID','SampleID','CollectionDate','SampleEventCode',
          'Contaminant','MRL','Units','MethodID','AnalyticalResultsSign','AnalyticalResultValue']
    with zipfile.ZipFile(ARCHIVE) as z,z.open('UCMR5_All.txt') as f:
        for c in pd.read_csv(f,sep='\t',encoding='latin-1',dtype=str,keep_default_na=False,usecols=cols,chunksize=250000):
            c=c.loc[c.SampleEventCode.eq('SE1') & c.Contaminant.isin(MRL)].copy()
            c['location_id']=c.PWSID+'|'+c.FacilityID+'|'+c.SamplePointID
            c=c.loc[c.location_id.isin(r.index)]
            records.append(c)
    x=pd.concat(records,ignore_index=True)
    assert len(x)==len(d)*25 and not x.duplicated(['location_id','Contaminant']).any()
    assert x.MethodID.eq('EPA 533').all() and x.Units.eq('µg/L').all()
    assert x.AnalyticalResultsSign.isin(['=','<']).all()
    assert x.SampleID.eq(x.location_id.map(r.SampleID)).all()
    assert x.CollectionDate.eq(x.location_id.map(r.CollectionDate)).all()
    assert np.allclose(pd.to_numeric(x.MRL),x.Contaminant.map(MRL),atol=1e-12,rtol=0)
    x['report']=x.AnalyticalResultsSign.eq('=').astype(int)
    x['conc']=pd.to_numeric(x.AnalyticalResultValue,errors='coerce').where(x.report.eq(1))
    assert x.loc[x.report.eq(1),'conc'].notna().all()
    assert (x.loc[x.report.eq(1),'conc']>=x.loc[x.report.eq(1),'Contaminant'].map(MRL)-1e-12).all()
    rep=x.pivot(index='location_id',columns='Contaminant',values='report').loc[d.location_id]
    conc=x.pivot(index='location_id',columns='Contaminant',values='conc').loc[d.location_id]
    for a in MRL:
        assert np.array_equal(rep[a],r.loc[d.location_id,'report__'+a].astype(int))
    y=rep[SIX].to_numpy(int)
    assert np.array_equal(y,d[['y_native_detect__'+a for a in SIX]].to_numpy(int))
    visible=(conc[SIX].to_numpy(float)>=OLD).astype(int)
    assert np.array_equal(visible,d[['old_visible__'+a for a in SIX]].to_numpy(int))
    for j,a in enumerate(SIX):
        expected=np.where(visible[:,j]>0,np.log1p(conc[a].to_numpy(float)/OLD[j]),0.)
        assert np.allclose(expected,d['old_logratio__'+a],atol=1e-10,rtol=0)
    for a in sorted(set(MRL)-set(SIX)):
        d['aux_report__'+a]=rep[a].to_numpy(int)
        d['aux_logratio__'+a]=np.where(rep[a].to_numpy(bool),np.log(conc[a].to_numpy(float)/MRL[a]),0.)
    metadata=['location_id','pws_id','state','state_fold','region','pws_size','water_type','month_sin','month_cos']
    keep=metadata+['old_'+part+'__'+a for a in SIX for part in ['visible','logratio']]
    keep+=['y_native_detect__'+a for a in SIX]
    keep+=[kind+'__'+a for a in sorted(set(MRL)-set(SIX)) for kind in ['aux_report','aux_logratio']]
    d=d[keep]
    assert d.select_dtypes('number').notna().all().all()
    assert d.groupby('pws_id').state_fold.nunique().max()==1
    assert d.groupby('state').state_fold.nunique().max()==1
    target=OUT/'cohort.csv.gz'
    d.to_csv(target,index=False,compression={'method':'gzip','mtime':0})
    sources=[CACHE,REF,ARCHIVE,Path(__file__),OUT/'PROTOCOL.md',ROOT/'src/run_auxiliary_information_20260926.py',target]
    dump(OUT/'INPUT_LOCK.json',{'created_at_utc':pd.Timestamp.now(tz='UTC').isoformat(),
      'status':'Frozen before new model fitting','locations':len(d),'pws':d.pws_id.nunique(),
      'jurisdictions':d.state.nunique(),'raw_verified_rows':len(x),'targets':SIX,
      'auxiliaries':sorted(set(MRL)-set(SIX)),'native_mrl_ugL':MRL,
      'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources}})
    print(json.dumps({'locations':len(d),'pws':d.pws_id.nunique(),'folds':d.state_fold.value_counts().to_dict(),'raw_verified_rows':len(x)},indent=2))

if __name__=='__main__':prepare()
