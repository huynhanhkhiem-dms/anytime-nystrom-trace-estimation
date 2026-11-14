"""Real-data kill test; run before any manuscript is written."""
import json,time,hashlib,sys,argparse
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.stats import beta
from threadpoolctl import threadpool_limits
from core import trajectory, kappa

ROOT=Path(__file__).resolve().parent

def finite_json(x):
    if isinstance(x,dict):return {k:finite_json(v) for k,v in x.items()}
    if isinstance(x,list):return [finite_json(v) for v in x]
    if isinstance(x,(float,np.floating)) and not np.isfinite(x):return str(x)
    if isinstance(x,np.generic):return x.item()
    return x

def check_math():
    x=np.r_[np.linspace(-.5,0,10001),np.geomspace(1e-10,1e5,10000)]
    k=float(kappa(np.array(.5)))
    gap=np.log1p(-2*x+2*k*x*x)+2*x
    # Diagonal exact Gaussian expectation, not a Monte Carlo proof.
    rng=np.random.default_rng(117)
    worst=0.
    for j in range(200):
        eig=np.exp(rng.uniform(-10,5,37));mu=eig.sum()
        for lam in [0.01/mu,.5/mu,5/mu,-.01/mu,-.5/mu]:
            kk=1. if lam>=0 else kappa(-lam*mu)
            logex=float(np.sum(-lam*eig-.5*np.log(1-2*lam*eig+2*kk*lam**2*eig**2)))
            worst=max(worst,logex)
    if gap.min() < -1e-12 or worst>1e-12:raise AssertionError('Exponential inequality failed')
    # Invariant test against explicitly formed residual.
    z=rng.normal(size=(35,25));A=z@z.T;B=np.zeros_like(A)
    max_negative=0.;identity_error=0.
    for i in range(15):
        g=rng.normal(size=35);R=A-B;r=R@g;q=float(g@r);v=float(r@r)
        delta=np.outer(r,r)/q
        identity_error=max(identity_error,abs(np.trace(delta)-v/q))
        B+=delta
        max_negative=min(max_negative,float(np.linalg.eigvalsh(A-B).min()))
    return {'scalar_grid_min_gap':float(gap.min()),'largest_log_exact_gaussian_moment':worst,
            'residual_min_eigenvalue_numeric':max_negative,'trace_increment_identity_abs_error':identity_error,
            'note':'Numerical checks support derivation; they do not replace a proof or certify floating-point arithmetic.'}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,default=100)
    parser.add_argument('--budget',type=int,default=40)
    parser.add_argument('--name',default='pilot_results')
    args=parser.parse_args()
    start=time.time();raw=np.loadtxt(ROOT/'spambase.data',delimiter=',')
    X=raw[:,:-1];X=(X-X.mean(0))/X.std(0)
    C=X.T@X/len(X)
    eig,U=eigh(C); ridge=.1
    # A = (C + ridge I)^-1 : real-data covariance inverse, matrix-free solve
    # implemented spectrally for pilot speed; eigenvalues used for ground truth.
    vals=1/(eig+ridge);truth=float(vals.sum());n=len(vals)
    def mv(g):return U@(vals*(U.T@g))
    results=[];paths=[];seeds=args.seeds;budget=args.budget;alpha=.05;epsilon=.05
    for seed in range(seeds):
        rows,e,L=trajectory(mv,n,budget,np.random.default_rng(seed),alpha=alpha)
        covered=all(x['lower']<=truth<=x['upper'] for x in rows)
        stop=next((x['step'] for x in rows if np.isfinite(x['upper']) and
                   x['lower']>0 and (x['upper']-x['lower'])/(x['upper']+x['lower'])<=epsilon),None)
        final=rows[-1]
        results.append({'seed':seed,'all_checked_times_covered':covered,'stop_at_5pct':stop,
                        'final_lower':final['lower'],'final_upper':final['upper'],
                        'final_relative_radius':(final['upper']-final['lower'])/(final['upper']+final['lower'])})
        if seed<3:paths.append({'seed':seed,'rows':rows})
    fails=sum(not x['all_checked_times_covered'] for x in results)
    cp=(float(beta.ppf(.025,fails,seeds-fails+1)) if fails else 0.,
        float(beta.ppf(.975,fails+1,seeds-fails)) if fails<seeds else 1.)
    summary={'status':'PILOT_ONLY_NOT_SUBMISSION_READY','date':'2026-11-06','dataset':'UCI Spambase',
        'raw_shape':list(raw.shape),'nonfinite_raw':int((~np.isfinite(raw)).sum()),
        'operator':'(standardized sample covariance + 0.1 I)^(-1)',
        'operator_dimension':n,'trace_ground_truth_eigendecomposition':truth,
        'ground_truth_warning':'Double precision reference, not interval-arithmetic certified.',
        'matvec_implementation':'Spectral reference oracle; does not establish performance on expensive iterative solves.',
        'random_seeds':seeds,'max_probes_per_seed':budget,'alpha':alpha,
        'all_time_failures':fails,'binomial_95pct_interval_failure_probability':cp,
        'stopped_at_5pct_within_budget':sum(x['stop_at_5pct'] is not None for x in results),
        'median_final_relative_radius':float(np.median([x['final_relative_radius'] for x in results])),
        'math_sanity':check_math(),'seconds':time.time()-start,
        'limitations':['One real dataset/operator only','No baseline comparison yet','Not a novelty proof',
                      'No finite-precision guarantee','Pilot seeds are development data, not confirmatory evidence']}
    (ROOT/(args.name+'.json')).write_text(json.dumps(finite_json({'summary':summary,'seeds':results,'example_paths':paths}),indent=2),encoding='utf-8')
    print(json.dumps(finite_json(summary),indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
