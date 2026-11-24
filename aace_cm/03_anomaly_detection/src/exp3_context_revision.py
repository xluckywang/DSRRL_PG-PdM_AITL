"""Condition alarm calibration on older measured context; keep inputs and score fixed."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import evaluate_exp3_revision_alarm as replay
from evaluate_exp3_revision_causal_fill import dense_input
import exp3_fusion_revision as fusion


CONTEXTS = ((0,), (1,), (2,), (0,2), (1,2))


def context(ts, values, mean, scale):
    idx, dense, filled = dense_input(ts, values)
    output = np.full((len(dense),3),np.nan,dtype=np.float32)
    scored = np.zeros(len(dense),dtype=bool)
    for l,r in replay.spans(dense,np.isfinite(filled).all(axis=1)):
        a=(filled[l:r].astype(np.float64)-mean)/scale
        t=np.arange(len(a))
        left,right=np.maximum(0,t-330),np.maximum(0,t-65)
        n=np.maximum(1,right-left)[:,None]
        cum=np.vstack([np.zeros((1,4)),a.cumsum(axis=0)])
        sq=np.vstack([np.zeros((1,4)),(a**2).cumsum(axis=0)])
        avg=(cum[right]-cum[left])/n
        var=np.maximum(0,(sq[right]-sq[left])/n-avg**2)
        out=np.column_stack([np.sqrt((avg[:,:2]**2+var[:,:2]).mean(1)),
                             np.sqrt(var[:,:2].mean(1)),avg[:,3]])
        out[(right-left)<60]=np.nan
        output[l+replay.DELAY:r-replay.DELAY]=out[replay.DELAY:-replay.DELAY]
        scored[l+replay.DELAY:r-replay.DELAY]=True
    return output[idx],scored[idx]


def extract(args):
    plan=json.loads((args.plan/"plan.json").read_text(encoding="utf-8"))
    ph=replay.sha(args.plan/"plan.json")
    assert args.receipt.read_text(encoding="utf-8")==ph
    if args.stage=="test":
        freeze=json.loads(args.selection.read_text(encoding="utf-8"))
        assert freeze["plan_sha"]==ph and freeze["driver_sha"]==replay.sha(Path(__file__))
    fusion.configure(args.root,args.root)
    assert replay.sha(args.reference)==plan["fits"]["42"]["reference"]
    with np.load(args.reference) as a:
        mean,scale=a["mean"],a["scale"]
    args.out.mkdir(parents=True,exist_ok=False)
    days={}
    keys=("calibration","tune","check") if args.stage=="dev" else ("test",)
    for role in keys:
        for day in plan[role]:
            ts,raw=fusion.load_background(replay.DAILY/f"{day}_preprocessed.csv")
            mask=args.plan/f"{day}.npz"
            assert replay.sha(mask)==plan["masks"][day]
            with np.load(mask) as m:
                np.testing.assert_array_equal(ts,m["timestamp"])
                seen,state=m["observed"],m["state"]
            days[day]=ts,seen,state
            if role=="calibration":
                z,valid=context(ts,np.where(seen,raw,np.nan),mean,scale)
                np.savez_compressed(args.out/f"cal_{day}.npz",z=z[valid&(state==2)])
    for i,(role,day,fault,level,pct,name) in enumerate(fusion.case_plan(plan,args.stage),1):
        ts,seen,state=days[day]
        values,y=replay.read_scenario(name,ts,state)
        z,valid=context(ts,np.where(seen,values,np.nan),mean,scale)
        keep=valid&(y | (state==2))
        np.savez_compressed(args.out/f"{name}.npz",z=z[keep],y=y[keep])
        if i%120==0:
            print(f"Context extraction: {i} {args.stage} cases",flush=True)
    (args.out/"verified.json").write_text(json.dumps(dict(plan_sha=ph,stage=args.stage,count=i,
        driver_sha=replay.sha(Path(__file__))),indent=2),encoding="utf-8")


def bins(z,dims,edges):
    indices=np.zeros(len(z),dtype=np.int16)
    invalid=np.zeros(len(z),dtype=bool)
    for j,e in zip(dims,edges):
        invalid|=~np.isfinite(z[:,j])
        indices=indices*(len(e)+1)+np.searchsorted(e,z[:,j],side="right")
    indices[invalid]=-1
    return indices


def survival(ordered,x):
    return (1+len(ordered)-np.searchsorted(ordered,x,side="left"))/(len(ordered)+1)


def read(file):
    with np.load(file) as a:
        return tuple(a[k] for k in a.files)


def fit_reference(z,s,dims):
    edges=[np.unique(np.quantile(z[np.isfinite(z[:,j]),j],[.25,.50,.75])) for j in dims]
    cells=bins(z,dims,edges)
    references={i:np.sort(s[cells==i],axis=0) for i in np.unique(cells) if i>=0 and (cells==i).sum()>=800}
    return edges,references


def scored(x,z,normal,inc,models,raw_reference,weights,options):
    incumbent=fusion.score_bank(x,normal,inc)[:,0]
    merged=np.column_stack([x[:,[0,1,2,5]],incumbent])
    p0=np.column_stack([survival(raw_reference[:,j],merged[:,j]) for j in range(5)])
    lookup={}
    for dims,(edges,refs) in models.items():
        cells=bins(z,dims,edges)
        p=p0.copy()
        for key,ref in refs.items():
            mask=cells==key
            if mask.any():
                for j in range(5):
                    p[mask,j]=survival(ref[:,j],merged[mask,j])
        lookup[dims]=p
    answer=np.empty((len(x),len(options)),dtype=np.float32)
    for i,c in enumerate(options.values()):
        if c is None:
            answer[:,i]=incumbent
        else:
            dims,mode,blend=c
            p=(1-blend)*p0+blend*lookup[tuple(dims)]
            score=-np.log(p)
            answer[:,i]=score[:,-1] if mode=="final" else score[:,:4]@weights
    return answer


def assess(args,phase):
    plan_hash=replay.sha(args.plan/"plan.json")
    prior=json.loads(args.incumbent.read_text(encoding="utf-8"))
    assert prior["plan_sha"]==plan_hash
    chosen=prior["selected"]
    inc={chosen:prior["candidates"][chosen]}
    weights=np.array(prior["candidates"][chosen]["weights"])[[0,1,2,5]]
    dev=phase=="dev"
    options={"incumbent":None}
    if dev:
        for dims in CONTEXTS:
            for mode in ("final","components"):
                for blend in (.5,1.):
                    options[f"{mode}_{''.join(map(str,dims))}_{blend}"]=[dims,mode,blend]
    else:
        frozen=json.loads(args.selection.read_text(encoding="utf-8"))
        assert frozen["plan_sha"]==plan_hash and frozen["incumbent_sha"]==replay.sha(args.incumbent)
        options=frozen["options"]
    needed={tuple(v[0]) for v in options.values() if v is not None}
    rows,cutoffs,calibration_fpr=[],{},{}
    for s in replay.SEEDS:
        store=args.runs/f"{phase}_{s}"
        cc=args.context/f"context_{phase}"
        calctx=args.context/"context_dev"
        for p in (store,cc,calctx):
            assert json.loads((p/"verified.json").read_text(encoding="utf-8"))["plan_sha"]==plan_hash
        xs=[]; zs=[]
        for file in sorted((args.runs/f"dev_{s}").glob("cal_*.npz")):
            x,=read(file)
            z,=read(calctx/file.name)
            assert len(x)==len(z)
            xs.append(x)
            zs.append(z)
        assert len(xs)==8
        x,z=np.concatenate(xs),np.concatenate(zs)
        normal=np.sort(x[:,:6],axis=0)
        inc_scores=fusion.score_bank(x,normal,inc)[:,0]
        combined=np.column_stack([x[:,[0,1,2,5]],inc_scores])
        raw_reference=np.sort(combined,axis=0)
        models={dims:fit_reference(z,combined,dims) for dims in needed}
        if dev:
            a=scored(x,z,normal,inc,models,raw_reference,weights,options)
            limits=np.quantile(a,.99,axis=0,method="higher")
            np.testing.assert_allclose(limits[0],prior["cutoffs"][str(s)][chosen],rtol=0,atol=1e-6)
            fpr=(a>limits).mean(0)
            assert (fpr<=.01).all()
            calibration_fpr[str(s)]=dict(zip(options,map(float,fpr)))
        else:
            limits=np.array([frozen["cutoffs"][str(s)][name] for name in options],dtype=np.float32)
        cutoffs[str(s)]=dict(zip(options,map(float,limits)))
        tab=pd.read_csv(store/"cases.csv")
        for (role,fault,level),g in tab.groupby(["role","mechanism","level"]):
            parts=[]
            for case in g.scenario:
                x,y=read(store/f"{case}.npz")
                z,y2=read(cc/f"{case}.npz")
                np.testing.assert_array_equal(y,y2)
                parts.append((x,z,y))
            x,z,y=[np.concatenate(a) for a in zip(*parts)]
            a=scored(x,z,normal,inc,models,raw_reference,weights,options)
            hit=a>limits
            tp,fp=hit[y].sum(0),hit[~y].sum(0)
            pos,neg=int(y.sum()),int((~y).sum())
            for j,name in enumerate(options):
                rows.append(dict(seed=s,role=role,mechanism=fault,level=level,candidate=name,
                    tp=int(tp[j]),fp=int(fp[j]),fn=int(pos-tp[j]),tn=int(neg-fp[j]),
                    precision=tp[j]/(tp[j]+fp[j]) if tp[j]+fp[j] else np.nan,recall=tp[j]/pos,
                    fpr=fp[j]/neg,auc_roc=np.nan if dev else roc_auc_score(y,a[:,j]),
                    coverage=pos/g.fault_total.sum()))
        print(f"Conditional {phase} seed {s} completed",flush=True)
    df=pd.DataFrame(rows)
    args.out.mkdir(parents=True,exist_ok=False)
    df.to_csv(args.out/"by_seed.csv",index=False)
    if dev:
        t=df.loc[df.role.eq("tune")]
        rank=t.groupby(["candidate","mechanism"]).recall.mean().unstack()
        rank["fpr"]=t.groupby("candidate").fpr.mean()
        base=rank.loc["incumbent"]
        allowed=rank.fpr.le(max(.015,base.fpr+.001))
        for fault in replay.FAULTS:
            if fault!="imbalance":
                allowed&=rank[fault].ge(base[fault]-.03)
        name=rank.loc[allowed].sort_values(["imbalance","fpr"],ascending=[False,True]).index[0]
        keep=list(dict.fromkeys(["incumbent",name]))
        rank.assign(qualifies=allowed).to_csv(args.out/"ranking.csv")
        (args.out/"selection.json").write_text(json.dumps(dict(plan_sha=plan_hash,
            incumbent_sha=replay.sha(args.incumbent),driver_sha=replay.sha(Path(__file__)),selected=name,
            options={n:options[n] for n in keep},
            cutoffs={s:{n:v[n] for n in keep} for s,v in cutoffs.items()},
            calibration_fpr={s:{n:v[n] for n in keep} for s,v in calibration_fpr.items()}),
            indent=2),encoding="utf-8")
        print(f"Selected on development: {name}",flush=True)
        chosen=df.loc[df.role.eq("check")&df.candidate.isin(keep)]
    else:
        assert len(df)==200*len(options)
        chosen=df.loc[df.level.between(4,8)]
    middle=chosen.groupby(["candidate","mechanism"])[list(fusion.METRICS)].mean()
    middle.to_csv(args.out/"comparison.csv")
    print(middle.to_string(),flush=True)


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(required=True)
    p=sub.add_parser("extract")
    for f in ("plan","root","receipt","reference","out"):
        p.add_argument(f"--{f}",type=Path,required=True)
    p.add_argument("--stage",choices=("dev","test"),required=True)
    p.add_argument("--selection",type=Path)
    p.set_defaults(action=extract)
    for mode,stage in (("select","dev"),("evaluate","test")):
        p=sub.add_parser(mode)
        for f in ("plan","runs","context","incumbent","out"):
            p.add_argument(f"--{f}",type=Path,required=True)
        if stage=="test":
            p.add_argument("--selection",type=Path,required=True)
        p.set_defaults(action=lambda a,phase=stage:assess(a,phase))
    args=parser.parse_args()
    args.action(args)


if __name__=="__main__":
    main()
