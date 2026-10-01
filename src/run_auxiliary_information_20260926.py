"""Frozen five-fold auxiliary-information experiment. Torch required for fitting."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import time
import numpy as np
import pandas as pd

SIX=['PFBS','PFHpA','PFHxS','PFNA','PFOA','PFOS']
BITS=((np.arange(64)[:,None]>>np.arange(6))&1).astype(int)
COUNTS=BITS.sum(1)
PENALTIES=[.0001,.01]
CLASSES=['independent','linear','nonlinear']

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def dump(p,x):
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def cmi(a,y,k):
    total=len(y); ans=0.
    for level in np.unique(k):
        ix=k==level
        tab=np.zeros((3,64));np.add.at(tab,(a[ix],y[ix]),1.)
        tab=tab[np.any(tab>0,axis=1)][:,np.any(tab>0,axis=0)]
        p=tab/tab.sum();ind=p.sum(1)[:,None]*p.sum(0)[None,:]
        good=p>0
        ans+=ix.sum()/total*np.sum(p[good]*np.log(p[good]/ind[good]))
    return float(ans)

def select_aux(train,aux):
    y=train[['y_native_detect__'+a for a in SIX]].to_numpy(int)@2**np.arange(6)
    rows=[]
    for a in aux:
        report=train['aux_report__'+a].to_numpy(int)
        b=report+(train['aux_logratio__'+a].to_numpy(float)>=np.log(2)).astype(int)
        rows.append({'analyte':a,'reports':int(report.sum()),'eligible':bool(report.sum()>=20),
                     'conditional_mutual_information':cmi(b,y,COUNTS[y])})
    ranking=pd.DataFrame(rows).sort_values(['eligible','conditional_mutual_information','analyte'],ascending=[False,False,True])
    assert ranking.eligible.sum()>=3
    return ranking.loc[ranking.eligible,'analyte'].tolist(),ranking

def feature_matrix(d,aux,spec=None):
    cols=[d[c].to_numpy(float) for c in ['month_sin','month_cos']]
    names=['month_sin','month_cos'];levels={}
    for c in ['region','pws_size','water_type']:
        v=d[c].fillna('Unknown').astype(str)
        lev=sorted(set(v)|{'Unknown'}) if spec is None else spec['levels'][c]
        v=v.where(v.isin(lev),'Unknown');levels[c]=lev
        for value in lev:cols.append(v.eq(value).to_numpy(float));names.append(c+'='+value)
    for a in SIX:
        for kind in ['old_visible','old_logratio']:
            key=kind+'__'+a;cols.append(d[key].to_numpy(float));names.append(key)
    for a in aux:
        for kind in ['aux_report','aux_logratio']:
            key=kind+'__'+a;cols.append(d[key].to_numpy(float));names.append(key)
    x=np.column_stack(cols)
    if spec is None:
        mean=x.mean(0);scale=np.maximum(x.std(0),.1)
        spec={'levels':levels,'names':names,'mean':mean.tolist(),'scale':scale.tolist(),'auxiliaries':aux}
    assert names==spec['names']
    return ((x-np.array(spec['mean']))/np.array(spec['scale'])).astype('float32'),spec

def shuffle_aux(d,aux,seed):
    z=d.copy();rng=np.random.default_rng(seed)
    old=d[['old_visible__'+a for a in SIX]].to_numpy(int)@2**np.arange(6)
    strata=d[['region','water_type','pws_size']].astype(str).agg('|'.join,axis=1)+'|'+pd.Series(old,index=d.index).astype(str)
    col=[f'{kind}__{a}' for a in aux for kind in ['aux_report','aux_logratio']]
    origin=np.arange(len(d));mapping=origin.copy()
    for idx in strata.groupby(strata).indices.values():mapping[idx]=rng.permutation(idx)
    z.loc[:,col]=d.iloc[mapping][col].to_numpy()
    changed=np.any(z[col].to_numpy()!=d[col].to_numpy(),axis=1)
    return z,{'mapping_moved_fraction':float(np.mean(mapping!=origin)),'values_changed_fraction':float(changed.mean())}

def label_arrays(d):
    y=d[['y_native_detect__'+a for a in SIX]].to_numpy(int)
    old=d[['old_visible__'+a for a in SIX]].to_numpy(int)
    assert np.all(y>=old)
    return y,old

def scores(prob,y):
    q=(1-1e-12)*prob/prob.sum(1,keepdims=True)+1e-12/64
    state=y@2**np.arange(6);k=y.sum(1);rows=np.arange(len(y))
    mass=np.column_stack([q[:,COUNTS==j].sum(1) for j in range(7)])
    full=-np.log(q[rows,state]);count=-np.log(mass[rows,k]);identity=full-count
    assert identity.min()>-1e-10
    p=q@BITS
    truth=np.eye(7)[k]
    return np.column_stack([full,count,np.maximum(identity,0),
       ((np.cumsum(mass,1)[:,:6]-np.cumsum(truth,1)[:,:6])**2).sum(1),
       (mass[:,2:].sum(1)-(k>=2))**2,((p-y)**2).mean(1)])

def run_fold(dest,fold,device):
    import torch
    from torch import nn
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    bits=torch.tensor(BITS,dtype=torch.float32,device=device)
    counts=torch.tensor(COUNTS,device=device)
    d=pd.read_csv(dest/'cohort.csv.gz',dtype={'location_id':str,'pws_id':str,'state':str,'region':str})
    lock=json.loads((dest/'INPUT_LOCK.json').read_text())
    assert sha(dest/'cohort.csv.gz')==lock['source_sha256']['outputs/auxiliary_information_20260926/cohort.csv.gz']
    assert sha(Path(__file__))==lock['source_sha256']['src/run_auxiliary_information_20260926.py']
    aux=lock['auxiliaries'];testidx=np.flatnonzero(d.state_fold.eq(fold));validx=np.flatnonzero(d.state_fold.eq((fold+1)%5))
    trainidx=np.flatnonzero(~d.state_fold.isin([fold,(fold+1)%5]))
    parts=[d.iloc[i].reset_index(drop=True) for i in [trainidx,validx,testidx]]
    for key in ['state','pws_id']:
        for i,j in [(0,1),(0,2),(1,2)]:assert not set(parts[i][key])&set(parts[j][key])
    ranking,ranktable=select_aux(parts[0],aux)
    out=dest/f'fold{fold}';out.mkdir(exist_ok=True)
    assert not (out/'DONE.json').exists(),'Never overwrite a completed fold'
    ranktable.to_csv(out/'training_auxiliary_ranking.csv',index=False)
    settings={'baseline':[]}
    settings.update({f'selected{b}':ranking[:b] for b in [1,2,3]})
    settings['all19']=aux
    for seed in range(3):
        order=np.random.default_rng(2026092600+seed).permutation(aux).tolist()
        for b in [1,2,3]:settings[f'random{seed}_{b}']=order[:b]
    settings['shuffled3']=ranking[:3]
    dump(out/'SETTINGS.json',settings)
    allpred={};tuning=[];audits=[]

    def tensors(frame,x,head):
        y,old=label_arrays(frame)
        state=y@2**np.arange(6);k=y.sum(1)
        compatible=np.all(BITS[None,:,:]>=old[:,None,:],axis=2)
        if head=='identity':
            allowed=compatible & (COUNTS[None,:]==k[:,None])
            active=allowed.sum(1)>1
        elif head=='count':
            allowed=np.arange(7)[None,:]>=old.sum(1)[:,None];active=allowed.sum(1)>1
        else:allowed=old==0;active=allowed.any(1)
        label=y if head=='independent' else (state if head=='identity' else k)
        return (torch.tensor(x[active],device=device),torch.tensor(label[active],device=device),
                torch.tensor(allowed[active],device=device),active)

    def lossfn(logits,label,allowed,head):
        if head=='independent':
            return (torch.nn.functional.binary_cross_entropy_with_logits(logits,label.float(),reduction='none')*allowed).sum(1).mean()
        return torch.nn.functional.cross_entropy(logits.masked_fill(~allowed,-1e9),label)

    def make(width,head,model,seed,bias):
        torch.manual_seed(seed)
        outputs={'independent':6,'count':7,'identity':64}[head]
        net=nn.Linear(width,outputs) if model!='nonlinear' else nn.Sequential(nn.Linear(width,64),nn.Tanh(),nn.Linear(64,outputs))
        last=net if model!='nonlinear' else net[-1]
        nn.init.zeros_(last.weight)
        with torch.no_grad():last.bias.copy_(torch.tensor(bias,dtype=torch.float32))
        return net.to(device)

    def prior(frame,head):
        y,_=label_arrays(frame)
        if head=='independent':
            p=(y.sum(0)+.5)/(len(y)+1);return np.log(p/(1-p))
        if head=='count':return np.log(np.bincount(y.sum(1),minlength=7)+.5)
        return np.log(np.bincount(y@2**np.arange(6),minlength=64)+.5)

    def fit_head(frames,xs,head,model,setting):
        tr=tensors(frames[0],xs[0],head);va=tensors(frames[1],xs[1],head)
        assert len(tr[0])>0 and len(va[0])>0
        seed=20260926+100*fold+{'independent':0,'count':1,'identity':2}[head]
        best=None
        for penalty in PENALTIES:
            net=make(xs[0].shape[1],head,model,seed,prior(frames[0],head))
            def penalty_term():return .5*penalty*sum((p*p).sum() for n,p in net.named_parameters() if n.endswith('weight'))
            begun=time.monotonic()
            epochs=0;valbest=math.inf;beststate=None;lastbetter=0
            if model!='nonlinear':
                opt=torch.optim.LBFGS(net.parameters(),lr=1,max_iter=500,tolerance_grad=1e-6,tolerance_change=1e-9,line_search_fn='strong_wolfe')
                def closure():
                    opt.zero_grad();loss=lossfn(net(tr[0]),tr[1],tr[2],head)+penalty_term();loss.backward();return loss
                opt.step(closure)
                with torch.no_grad():valbest=float(lossfn(net(va[0]),va[1],va[2],head))
                beststate={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
                epochs=int(opt.state[next(iter(net.parameters()))]['n_iter'])
            else:
                opt=torch.optim.Adam(net.parameters(),lr=.01)
                for epoch in range(1,501):
                    opt.zero_grad();loss=lossfn(net(tr[0]),tr[1],tr[2],head)+penalty_term();loss.backward();opt.step()
                    if epoch%5==0:
                        with torch.no_grad():value=float(lossfn(net(va[0]),va[1],va[2],head))
                        if value<valbest-1e-7:
                            valbest=value;lastbetter=epoch;beststate={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
                        if epoch-lastbetter>=60:break
                epochs=lastbetter
            assert math.isfinite(valbest)
            record={'fold':fold,'setting':setting,'model':model,'head':head,'penalty':penalty,
                    'validation_loss':valbest,'iterations_or_best_epoch':epochs,'train_informative':len(tr[0]),
                    'validation_informative':len(va[0]),'seconds':time.monotonic()-begun}
            tuning.append(record)
            if best is None or valbest<best[0]:best=(valbest,penalty,beststate,record)
        _,penalty,state,record=best
        net=make(xs[0].shape[1],head,model,seed,prior(frames[0],head));net.load_state_dict(state);net.eval()
        torch.save({'state_dict':state,'record':record},out/f'{setting}_{model}_{head}.pt')
        return net

    def predict(heads,x,frame,model):
        # No native target, true count, auxiliary label, or future value is read here.
        old=torch.tensor(frame[['old_visible__'+a for a in SIX]].to_numpy(int),device=device)
        xt=torch.tensor(x,device=device)
        with torch.no_grad():
            if model=='independent':
                logits=heads['independent'](xt)
                logp=torch.nn.functional.logsigmoid(logits);logn=torch.nn.functional.logsigmoid(-logits)
                v=old.bool();logp=logp.masked_fill(v,0.);logn=logn.masked_fill(v,-1e9)
                q=torch.exp(logp@bits.T+logn@(1-bits).T)
            else:
                countlog=heads['count'](xt)
                allowedk=torch.arange(7,device=device)[None,:]>=old.sum(1)[:,None]
                mass=torch.softmax(countlog.masked_fill(~allowedk,-1e9),1)
                logits=heads['identity'](xt)
                compatible=(bits[None,:,:]>=old[:,None,:]).all(2)
                q=torch.zeros_like(logits)
                for k in range(7):
                    ix=COUNTS==k;allowed=compatible[:,ix]
                    cond=torch.softmax(logits[:,ix].masked_fill(~allowed,-1e9),1)
                    q[:,ix]=cond*mass[:,k,None]
            return q.double().cpu().numpy()

    for setting,chosen in settings.items():
        frames=parts;shuffle_audit=None
        if setting=='shuffled3':
            sh=[shuffle_aux(f,chosen,2026092610+100*fold+i) for i,f in enumerate(parts)]
            frames=[a[0] for a in sh];shuffle_audit=[a[1] for a in sh]
        xtr,spec=feature_matrix(frames[0],chosen)
        xs=[xtr]+[feature_matrix(f,chosen,spec)[0] for f in frames[1:]]
        dump(out/f'{setting}_feature_spec.json',spec)
        replaced=frames[2].copy()
        for c in replaced:
            if c.startswith('y_'):replaced[c]=1-replaced[c]
        assert np.array_equal(feature_matrix(replaced,chosen,spec)[0],xs[2])
        for model in CLASSES:
            start=time.monotonic()
            headnames=['independent'] if model=='independent' else ['count','identity']
            heads={h:fit_head(frames,xs,h,model,setting) for h in headnames}
            q=predict(heads,xs[2],frames[2],model)
            assert np.isfinite(q).all() and q.min()>=0 and np.max(abs(q.sum(1)-1))<2e-6
            y,old=label_arrays(frames[2]);marg=q@BITS
            assert np.max(abs(marg[old==1]-1),initial=0)<2e-6
            replacedq=predict(heads,xs[2],replaced,model)
            assert np.array_equal(q,replacedq)
            # Independently reload each checkpoint before checking prediction replay.
            for h,net in heads.items():net.load_state_dict(torch.load(out/f'{setting}_{model}_{h}.pt',map_location=device,weights_only=False)['state_dict'])
            assert np.array_equal(q,predict(heads,xs[2],frames[2],model))
            allpred[f'{setting}__{model}']=q.astype('float64')
            value=scores(q,y)
            audits.append({'setting':setting,'model':model,'fold':fold,'test_events':len(y),
                           'mean_pattern_nll':float(value[:,0].mean()),'mean_identity_nll':float(value[:,2].mean()),
                           'normalization_max_error':float(np.max(abs(q.sum(1)-1))),
                           'hidden_target_invariance':True,'saved_checkpoint_replay':True,
                           'shuffle':shuffle_audit,'seconds':time.monotonic()-start})
            print(f'fold={fold} {setting} {model} NLL={value[:,0].mean():.6f} identity={value[:,2].mean():.6f} sec={time.monotonic()-start:.1f}',flush=True)
            pd.DataFrame(tuning).to_csv(out/'tuning.csv',index=False)
            dump(out/'fit_audit.json',audits)
        np.savez_compressed(out/'predictions.npz',index=testidx,**allpred)
    assert len(allpred)==45
    dump(out/'DONE.json',{'fold':fold,'models':45,'candidate_head_fits':len(tuning),
                         'test_events':len(testidx),'torch':torch.__version__,
                         'device':str(device),'state_disjoint':True,'pws_disjoint':True,
                         'code_sha256':sha(Path(__file__))})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--fold',type=int,required=True);parser.add_argument('--device',default='cuda:0')
    a=parser.parse_args();run_fold(a.directory,a.fold,a.device)
