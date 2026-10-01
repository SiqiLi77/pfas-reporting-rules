"""Descriptive score alignment on the same source rows and weights, no refits."""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/environmental_reconstruction_20260927'
for cohort in ['national','washington']:
    p=OUT/cohort
    md=pd.read_csv(p/'metadata.csv.gz',dtype={'location_id':str})
    ds=pd.read_csv(p/'source_analysis_rows.csv.gz',dtype={'location_id':str})
    weights=pd.read_csv(p/'standardization_weights.csv').set_index('stratum').weight
    z=np.load(p/'record_summaries.npz');keys=z['keys'].tolist()
    ix=pd.Series(np.arange(len(md)),index=md.location_id).loc[ds.location_id].to_numpy()
    models=list(dict.fromkeys(k.split('|')[0] for k in keys if not k.startswith('measured|')))
    rows=[]
    for model in models:
        for metric in ['pattern_nll','count_nll','brier__multiple']:
            f=ds[['stratum','water_type']].copy();f['value']=z['values'][ix,keys.index(model+'|'+metric)]
            cells=f.groupby(['stratum','water_type']).value.mean().unstack().loc[weights.index,['GW','SW']]
            for group,value in (weights@cells).items():
                rows.append({'cohort':cohort,'model':model,'metric':metric,'source':group,'estimate':value,'events':len(ds),'descriptive_post_result_check':True})
    pd.DataFrame(rows).to_csv(p/'source_aligned_scores.csv',index=False)
print('Source-population scores exported for every model and both cohorts.')
