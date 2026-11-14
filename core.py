"""Research prototype: exact-arithmetic theorem, floating-point implementation.

No claim of certified floating-point matrix arithmetic is made by this prototype.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import brentq

POS = np.array([1/32, 1/16, 1/8, 1/4, 1/2], dtype=float)
NEG = np.array([1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8], dtype=float)

def kappa(c):
    return (np.expm1(2*c)-2*c)/(2*c*c)

@dataclass
class Record:
    t: float
    q: float
    v: float

class TraceEvidence:
    def __init__(self, alpha=.05):
        if not 0 < alpha < 1: raise ValueError('alpha must lie in (0,1)')
        self.alpha=alpha
        self.records=[]
        self.hard_lower=0.
        self.running_lower=0.
        self.running_upper=np.inf
        self.empty=False
        self.threshold=np.log(2/alpha)

    def append(self, t, q, v):
        if not (np.isfinite([t,q,v]).all() and t>=0 and q>0 and v>0):
            raise ArithmeticError('Invalid positive residual observation; no silent clipping')
        self.records.append(Record(t,q,v))
        self.hard_lower=max(self.hard_lower,t+v/q)

    def log_e(self, theta, sign):
        if theta < self.hard_lower: return np.inf
        a=np.array([(r.t,r.q,r.v) for r in self.records])
        if not len(a):return 0.
        mu=theta-a[:,0]
        if np.any(mu<=0): return np.nan
        cs=POS if sign=='positive' else NEG
        u=a[:,1]/mu
        w=a[:,2]/mu**2
        if sign=='positive':
            logs=np.sum(cs[:,None]*(u-1)-cs[:,None]**2*w,axis=1)
        elif sign=='negative':
            logs=np.sum(cs[:,None]*(1-u)-kappa(cs)[:,None]*cs[:,None]**2*w,axis=1)
        else:raise ValueError(sign)
        return float(logsumexp(logs)-np.log(len(cs)))

    def interval(self):
        if self.empty:return (np.inf,-np.inf)
        d=self.hard_lower
        if not self.records:return (0.,np.inf)
        threshold=self.threshold
        # On theta >= hard_lower, each positive branch is decreasing because
        # c<=1/2 and theta-t_i>=v_i/q_i. Negative branches are increasing.
        def f(theta,sign):return self.log_e(theta,sign)-threshold
        scale=max(d,np.mean([r.t+r.q for r in self.records]),np.finfo(float).tiny)
        lo=d
        if f(d,'positive')>0:
            hi=2*scale
            while f(hi,'positive')>0:hi*=2
            lo=brentq(lambda z:f(z,'positive'),d,hi,xtol=1e-13*scale)
            lo=np.nextafter(lo,-np.inf)
        hi=np.inf
        if f(d,'negative')>0:
            self.empty=True
            return (np.inf,-np.inf)  # Empty confidence set; a recorded failure.
        # Finite upper endpoint exists only if the strategy mixture's limit
        # as theta -> infinity exceeds the testing threshold.
        limit=logsumexp(len(self.records)*NEG)-np.log(len(NEG))
        if limit>threshold:
            right=2*scale
            while f(right,'negative')<=0 and right<1e250:right*=2
            if right<1e250:
                hi=brentq(lambda z:f(z,'negative'),d,right,xtol=1e-13*scale)
                hi=np.nextafter(hi,np.inf)
        self.running_lower=max(self.running_lower,lo)
        self.running_upper=min(self.running_upper,hi)
        if self.running_lower>self.running_upper:
            self.empty=True
            return (np.inf,-np.inf)
        return (self.running_lower,self.running_upper)

def trajectory(matvec,n,budget,rng,alpha=.05,check_every=1):
    L=np.zeros((n,budget))
    rank=0;t=0.;e=TraceEvidence(alpha)
    records=[]
    for step in range(1,budget+1):
        g=rng.standard_normal(n)
        ag=matvec(g)
        r=ag-L[:,:rank]@(L[:,:rank].T@g)
        q=float(g@r);v=float(r@r)
        if q<=0:
            raise ArithmeticError(f'Nonpositive numerical residual at step {step}: {q}')
        e.append(t,q,v)
        L[:,rank]=r/np.sqrt(q)
        rank+=1
        t+=v/q
        if step%check_every==0 or step==budget:
            lower,upper=e.interval()
            records.append({'step':step,'t_before':e.records[-1].t,'q':q,'v':v,
                            'trace_sketch':t,'lower':lower,'upper':upper})
    return records,e,L[:,:rank]
