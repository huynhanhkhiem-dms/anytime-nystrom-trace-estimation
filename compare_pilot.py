"""Exploratory fair-query comparison, not an optimized runtime benchmark."""
import json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from core import trajectory,TraceEvidence
from baselines import gaussian_hutchinson,hutchpp,nystrompp,xnystrace

ROOT=Path(__file__).resolve().parent

def ablation(mv,n,budget,rng,mode):
    k=budget//2 if mode=='fixed_split' else 0
    L=np.zeros((n,budget));rank=0;t=0.;e=TraceEvidence(.05);rows=[]
    for step in range(1,budget+1):
        g=rng.standard_normal(n);r=mv(g)-L[:,:rank]@(L[:,:rank].T@g)
        q=float(g@r);v=float(r@r)
        if step<=k:
            L[:,rank]=r/np.sqrt(q);rank+=1;t+=v/q
        else:
            e.append(t,q,v)
            if step==budget:
                lo,hi=e.interval();rows.append({'step':step,'lower':lo,'upper':hi})
    return rows

def main():
    begin=time.time();raw=np.loadtxt(ROOT/'spambase.data',delimiter=',')[:,:-1]
    X=(raw-raw.mean(0))/raw.std(0);C=X.T@X/len(X);d,U=eigh(C);vals=1/(d+.1)
    truth=float(vals.sum());n=len(vals)
    def mv(g):return U@(vals*(U.T@g))
    allrows=[];alpha=.05
    for seed in range(100):
        rows,e,L=trajectory(mv,n,55,np.random.default_rng(seed),alpha=alpha,check_every=5)
        for row in rows:
            if row['step'] in [20,40,55]:
                m=row['step'];obs=e.records[:m];est=float(np.mean([r.t+r.q for r in obs]))
                allrows.append({'method':'Prequential Nyström','budget':m,'seed':seed,'estimate':est,
                                'lower':row['lower'],'upper':row['upper'],'interval_kind':'proposed anytime theorem; exact arithmetic',
                                'matvecs':m})
        for mode in ['no_sketch','fixed_split']:
            for budget in [20,40,55]:
                for row in ablation(mv,n,budget,np.random.default_rng(seed),mode):
                    allrows.append({'method':mode,'budget':row['step'],'seed':seed,**row,
                                    'interval_kind':'same evidence, no changing sketch / independent split ablation',
                                    'matvecs':row['step']})
        for m in [20,40,55]:
            for j,method in enumerate([gaussian_hutchinson,hutchpp,nystrompp,xnystrace]):
                ans=method(mv,n,m,np.random.default_rng(100000+seed*29+j*3000+m))
                allrows.append({'budget':m,'seed':seed,**ans})
    summary=[]
    methods=sorted(set(x['method'] for x in allrows))
    for method in methods:
        for m in [20,40,55]:
            a=[x for x in allrows if x['method']==method and x['budget']==m]
            if not a:continue
            row={'method':method,'budget':m,'trials':len(a)}
            if 'estimate' in a[0]:
                errs=np.array([x['estimate']/truth-1 for x in a])
                row.update(relative_rmse=float(np.sqrt(np.mean(errs**2))),
                           median_absolute_relative_error=float(np.median(abs(errs))))
            if 'lower' in a[0]:
                row.update(median_relative_certified_radius=float(np.median([(x['upper']-x['lower'])/(x['upper']+x['lower']) for x in a])),
                           failures_at_time=sum(not x['lower']<=truth<=x['upper'] for x in a),
                           five_pct_certificates=sum((x['upper']-x['lower'])/(x['upper']+x['lower'])<=.05 for x in a))
            summary.append(row)
    out={'status':'DEVELOPMENT_PILOT','truth':truth,'summary':summary,'raw':allrows,
         'seconds':time.time()-begin,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'core.py',ROOT/'baselines.py',Path(__file__)]},
         'limitations':['Not confirmatory seeds','No optimized block-oracle timing comparison',
                        'Fixed split allocates floor(each evaluated budget/2) to sketch; wider split tuning remains outstanding',
                        'Baseline point error and proposed certificate radius are different quantities and cannot be called equal-confidence speedups']}
    def encode(value):
        if isinstance(value,np.generic):return value.item()
        raise TypeError(type(value).__name__)
    (ROOT/'comparison_results.json').write_text(json.dumps(out,indent=2,default=encode),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='raw'},indent=2,default=encode))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
