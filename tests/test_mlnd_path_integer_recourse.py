"""Cross-design certificates against the independently built full path SOCP."""
import itertools

import numpy as np
import pytest

pytest.importorskip('gurobipy')
from mlnd_path.data import Instance
from mlnd_path.integer_recourse import FinitePathOracle, solve_integer_recourse
from mlnd_path.solver import solve_recourse


def _fixture():
    a=Instance('two_base_cut_audit',2,2,1,[3.,5.],[9.,10.],[1.,1.],[.2,.2],[.4],
               rho_max=.8,kappa=2.,fleet_cost=2.)
    for od,l,m in itertools.product(range(2),repeat=3):
        counts=[0,0,0];counts[m]=1
        a.service.append(dict(od=od,l=l,m=m,i=0,j=1,time=.2+.13*(l!=m),
                              cost=1.2+.7*(od!=l),swaps=tuple(counts)))
        counts[2]=1
        a.service.append(dict(od=od,l=l,m=m,i=1,j=0,time=.3+.11*(l!=m),
                              cost=.3+.4*(od!=m),swaps=tuple(counts)))
    for l,m in [(0,1),(1,0)]:
        counts=[0,0,0];counts[m]=1
        a.empty.append(dict(l=l,m=m,time=.2,cost=.6,swaps=tuple(counts)))
        counts[2]=1
        a.empty.append(dict(l=l,m=m,time=.25,cost=.1,swaps=tuple(counts)))
    return a


def test_small_recourse_and_every_binary_cut_match_complete_path_model():
    a=_fixture();designs=[]
    for f in itertools.product((0.,1.),repeat=5):
        pairs=[None]+[(l,m) for l,m in itertools.product(range(2),repeat=2) if f[l] and f[m]]
        for chosen in itertools.product(pairs,repeat=2):
            d=a.zero_design()
            d.update(dict(zip(['X0','X1','Y0','Y1','S2'],f)))
            for od,pair in enumerate(chosen):
                if pair is not None:d[f'B{od}:{pair[0]}:{pair[1]}']=1.
            designs.append(d)
    oracle=FinitePathOracle(a)
    allopen=oracle.evaluate(dict.fromkeys(a.design_keys(),1.))
    results=[];truth=[]
    for d in designs:
        reference=solve_recourse(a,d,time_limit=5.)
        result=solve_integer_recourse(a,d,time_limit=5.,universal_cost_lower=allopen['best_cost'],
                    universal_empty_lower=allopen['empty_cost'])
        assert reference['status']=='optimal'
        assert result['cut_valid'] and result['cg_complete'],result
        assert result['objective']==pytest.approx(reference['objective'],abs=8e-6)
        assert result['primal_residual']<=1e-6
        results.append(result);truth.append(reference['objective'])
    keys=a.design_keys()
    points=np.array([[d[k] for k in keys] for d in designs])
    slopes=np.array([[r['cut_slope'].get(k,0.) for k in keys] for r in results])
    lower=np.array([r['cut_intercept'] for r in results])[:,None]+slopes@points.T
    assert np.max(lower-np.asarray(truth)[None,:])<=2e-6
    assert np.max(np.asarray(truth)-np.diag(lower))<=1e-4
    maximum_saving=a.rho_max*float(np.sum(np.asarray(a.demand)*np.maximum(0.,-allopen['best_cost']).max(axis=(1,2))))
    for d,value in zip(designs,truth):
        for excluded in range(a.n_h):
            cover=a.c0-maximum_saving*sum(d[f'Y{h}'] for h in range(a.n_h) if h!=excluded)
            assert cover<=value+1e-6


def test_closed_origin_empty_and_missing_service_route_keep_zero_recourse_feasible():
    a=_fixture();d=a.zero_design();d.update(X0=1.,Y0=1.,S2=1.)
    d['B0:0:0']=1.
    result=solve_integer_recourse(a,d,time_limit=5.)
    assert result['unreachable_assigned_ods']==1
    assert result['rho'][0]==pytest.approx(0.,abs=1e-7)
    assert result['objective']==pytest.approx(a.c0,abs=1e-6)
    assert result['cut_valid'] and result['cg_complete']
