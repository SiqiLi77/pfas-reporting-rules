#!/usr/bin/env python3
"""Recompute core point results from frozen inputs; never overwrite released data."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import analyze_ucmr5_joint_information_decomposition as joint
import environmental_reconstruction_20260927 as env

def close(a,b,label,tol=2e-12):
    error=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    if error>tol:raise AssertionError(f'{label}: numerical error {error}')
    return error

def observation():
    path=ROOT/'outputs/ucmr5_threshold_pattern_transitions_v1_5_20260913/count_transition_matrix.csv'
    transition=pd.read_csv(path)
    n=int(transition.events.sum())
    native=int(transition.loc[transition.native_count.ge(2),'events'].sum())
    common=int(transition.loc[transition.common_count.ge(2),'events'].sum())
    assert (n,native,common)==(7677,829,10)
    d=pd.read_csv(ROOT/'outputs/significance_extension_20260924/panel_coverage/strict25_event_records.csv.gz',dtype={'PWSID':str,'State':str})
    assert len(d)==26423
    original=ROOT/'outputs/research_expansion_20260925/environment'
    w=pd.read_csv(original/'standardization_weights.csv').set_index('cell').standardization_weight
    d['cell']=d.State+'|'+d.Size
    d=d.loc[d.FacilityWaterType.isin(['GW','SW']) & d.cell.isin(w.index)]
    assert len(d)==20162
    reference=pd.read_csv(original/'source_rates.csv')
    estimates={}
    for scheme,col in [('six_old','k6_old'),('six_native','k6_native'),('twentyfive_native','k25_native')]:
        data=d[['cell','FacilityWaterType']].copy();data['outcome']=d[col].ge(2).astype(float)
        cellmeans=data.groupby(['cell','FacilityWaterType']).outcome.mean().unstack().loc[w.index,['GW','SW']]
        rates=w.to_numpy()@cellmeans.to_numpy()
        expected=reference.query("estimand=='common_support_standardized' and metric=='multi_rate' and scheme==@scheme").set_index('source').loc[['GW','SW'],'estimate']
        close(rates,expected,scheme)
        estimates[scheme]={'GW':float(rates[0]),'SW':float(rates[1])}
    delta=(estimates['twentyfive_native']['SW']-estimates['twentyfive_native']['GW'])-(estimates['six_native']['SW']-estimates['six_native']['GW'])
    return {'matched_events':n,'native_multiple':native,'shared_cutoff_multiple':common,
            'panel_standardized_rates':estimates,'panel_contrast_change_percentage_points':100*delta}

def fixed_joint():
    base=ROOT/'outputs/joint_information_decomposition_20260916'
    md=pd.concat([pd.read_csv(base/'payload'/f'metadata_fold_{f}.csv.gz',dtype={'location_id':str,'pws_id':str,'state':str}) for f in range(1,5)],ignore_index=True)
    assert len(md)==338496 and md.location_id.nunique()==21156
    weights=joint.evaluation_weights(md);codes=md.observed_state_code.to_numpy(int)
    reference=pd.read_csv(base/'analysis/model_metrics.csv').set_index(['model','weighting','metric']).loss
    results={};errors=[]
    for model in (*joint.MODELS,joint.BASELINE):
        if model==joint.BASELINE:q=joint.align_baseline(md,base/'baseline')
        else:q=np.concatenate([np.load(base/'payload'/f'fold_{f}.npz',allow_pickle=False)[model] for f in range(1,5)])
        losses,_,_=joint.score_distribution(q,codes)
        errors.append(close(losses['pattern_nll'],losses['count_nll']+losses['identity_given_count_nll'],model))
        results[model]={}
        for weighting,w in weights.items():
            vals={metric:float(w@v) for metric,v in losses.items()}
            for metric,value in vals.items():errors.append(close(value,reference.loc[model,weighting,metric],model+' '+metric))
            results[model][weighting]=vals
        del q,losses
        print('Recomputed joint scores:',model,flush=True)
    before=results['factorized']['jurisdiction_pws_equal'];after=results['conditional_shared_shock']['jurisdiction_pws_equal']
    gain=before['pattern_nll']-after['pattern_nll']
    return {'locations':21156,'threshold_queries':16,'maximum_score_error':max(errors),
            'shared_shock_relative_pattern_reduction_percent':100*gain/before['pattern_nll'],
            'count_share_of_pattern_gain_percent':100*(before['count_nll']-after['count_nll'])/gain,'scores':results}

def environment(out):
    env.OUT=out/'aligned_reconstruction';env.OUT.mkdir(parents=True,exist_ok=True)
    datasets=env.load_data();report={};errors=[]
    for name,d,models in datasets:
        base=ROOT/'outputs/environmental_reconstruction_20260927'/name
        reference=pd.read_csv(base/'chemical_scores.csv')
        expected=reference.loc[reference.resampling.eq('pws')].set_index(['model','metric']).estimate
        metadata=pd.read_csv(base/'metadata.csv.gz',dtype={'location_id':str,'pws_id':str,'state':str})
        assert list(d.location_id)==list(metadata.location_id)
        saved=np.load(base/'record_summaries.npz',allow_pickle=False);lookup={k:j for j,k in enumerate(saved['keys'])}
        source=pd.read_csv(base/'source_analysis_rows.csv.gz',dtype={'location_id':str,'pws_id':str,'state':str})
        index=pd.Series(np.arange(len(d)),index=d.location_id).loc[source.location_id].to_numpy()
        sw=pd.read_csv(base/'standardization_weights.csv').set_index('stratum').weight
        rate_reference=pd.read_csv(base/'source_rates_and_contrasts.csv').set_index(['condition','endpoint','quantity']).estimate
        for model,q in models.items():
            p,s,a=env.summarize_probabilities(q,d.observed_state_code.to_numpy(int),d.old_visible_code.to_numpy(int))
            for metric,values in s.items():
                errors.append(close(values,saved['values'][:,lookup[model+'|'+metric]],name+' '+model+' '+metric))
            _,_,means,states=env.pws_matrix(d,np.column_stack(list(s.values())))
            point=np.mean([means[states==st].mean(0) for st in np.unique(states)],axis=0)
            for metric,val in zip(s,point):errors.append(close(val,expected.loc[model,metric],name+' chemical score'))
            for endpoint in env.MAIN_ENDPOINTS:
                j=env.ENDPOINTS.index(endpoint);frame=source[['stratum','water_type']].copy();frame['p']=p[index,j]
                means=frame.groupby(['stratum','water_type']).p.mean().unstack().loc[sw.index,['GW','SW']]
                rates=sw.to_numpy()@means.to_numpy()
                for group,val in zip(['GW','SW'],rates):errors.append(close(val,rate_reference.loc[model,endpoint,group+'_rate'],name+' source rate'))
                errors.append(close(rates[1]-rates[0],rate_reference.loc[model,endpoint,'SW_minus_GW'],name+' group contrast'))
            print('Recomputed aligned endpoints:',name,model,flush=True)
        report[name]={'events':len(d),'systems':int(d.pws_id.nunique()),'source_supported_events':len(source),'models':len(models)}
    report['maximum_numeric_error']=max(errors)
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--section',choices=['all','observational','joint','environmental'],default='all')
    parser.add_argument('--output',type=Path,default=ROOT/'replay')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    result={'scope':'Frozen point-result replay, not retraining, calibration, or new confirmatory evaluation'}
    if args.section in ['all','observational']:result['observational']=observation()
    if args.section in ['all','joint']:result['joint']=fixed_joint()
    if args.section in ['all','environmental']:result['environmental']=environment(args.output)
    result['status']='PASS'
    target=args.output/'REPLAY_REPORT.json';target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('PASS; report:',target,flush=True)

if __name__=='__main__':main()
