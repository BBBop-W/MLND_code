"""Independent finite resource-path enumeration for the Nanshan mission oracle.

No optimization solver is imported or called. The reference explicitly visits
stations and locker actions while carrying residual energy across all stages;
it does not reuse the oracle's Floyd/body factorization or old label search.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from mlnd_path import nanshan_oracle as oracle


@pytest.fixture(autouse=True)
def _without_numba_compilation(monkeypatch):
    """Tiny fixtures use Python kernels so testing creates no JIT workload."""
    for name in ('_station_shortest', '_body', '_combine', '_finish'):
        function = getattr(oracle, name)
        monkeypatch.setattr(oracle, name, getattr(function, 'py_func', function))


def _fixture(bases, hubs, remote=(), *, battery=4.3, ready=.11,
             origins=None, destinations=None, od_pairs=None):
    bases = np.asarray(bases, dtype=float).reshape(-1, 2)
    hubs = np.asarray(hubs, dtype=float).reshape(-1, 2)
    remote = np.asarray(remote, dtype=float).reshape(-1, 2)
    origins = np.asarray(origins if origins is not None else [[.3,.4],[7.4,.8]], dtype=float)
    destinations = np.asarray(destinations if destinations is not None else [[1.2,.8],[7.8,-.3]], dtype=float)
    pairs = od_pairs if od_pairs is not None else list(itertools.product(range(len(origins)),range(len(destinations))))
    distance = lambda a,b: np.sqrt(((a[:,None,:]-b[None,:,:])**2).sum(axis=2))
    oh = distance(origins,hubs)
    dh = distance(destinations,hubs)
    return dict(n_l=len(bases),n_h=len(hubs),n_k=len(remote),n_od=len(pairs),
                bases=bases,hubs=hubs,remote=remote,origins=origins,destinations=destinations,
                od_origin=np.asarray([p[0] for p in pairs]),
                od_destination=np.asarray([p[1] for p in pairs]),
                first_cost=.29+.17*oh,ground_time=.027+.074*oh,
                terminal_cost=.41+.13*dh,ground=np.asarray([4.5+.071*k for k in range(len(pairs))]),
                battery=battery,speed=10.,flight_cost=.23,swap_time=.07,fleet_cost=2.9,ready=ready)


def _enumerate_mission(data, facilities, od, l, m):
    """Enumerate simple station walks separately in each of three phases.

    A station may reappear after a locker action. Within one phase, every visit
    restores the same full battery; removing a cycle reduces both elapsed time
    and positive monetary cost. Thus no phase needs a repeated station.
    """
    L,H = data['n_l'],data['n_h']
    stations = np.vstack((data['bases'],data['remote']))
    open_s = [k for k in range(len(stations))
              if facilities[k if k<L else L+H+k-L]]
    open_h = [h for h in range(H) if facilities[L+h]]
    if l not in open_s or m not in open_s:
        return math.inf,[]
    best = math.inf
    optimum = []
    o,d = int(data['od_origin'][od]),int(data['od_destination'][od])
    B,speed,swap = data['battery'],data['speed'],data['swap_time']

    for i,j in itertools.permutations(open_h,2):
        fixed_cost = data['first_cost'][o,i]+data['terminal_cost'][d,j]-data['ground'][od]

        def visit(stage, position, residual, elapsed, flown, used, swaps, minimum_energy):
            nonlocal best,optimum
            target = data['hubs'][i if stage==0 else j] if stage<2 else stations[m]
            distance = float(np.linalg.norm(position-target))
            if distance <= residual+1e-9:
                next_energy = residual-distance
                next_time = elapsed+distance/speed
                low = min(minimum_energy,next_energy)
                if stage==0:
                    samples=data.get('ready_samples',np.asarray([data['ready']]))
                    next_time = float(np.mean(np.maximum(next_time,samples+data['ground_time'][o,i])))
                    visit(1,target,next_energy,next_time,flown+distance,frozenset(),swaps,low)
                elif stage==1:
                    visit(2,target,next_energy,next_time,flown+distance,frozenset(),swaps,low)
                else:
                    next_time += swap
                    cost = fixed_cost+data['flight_cost']*(flown+distance)+data['fleet_cost']*next_time
                    record = dict(time=next_time,i=i,j=j,swaps=swaps+((stage,m),),minimum_energy=low)
                    if cost < best-1e-10:
                        best=float(cost)
                        optimum=[record]
                    elif abs(cost-best)<=1e-10:
                        optimum.append(record)
            for k in open_s:
                if k in used or (stage==2 and k==m):
                    continue
                distance=float(np.linalg.norm(position-stations[k]))
                if distance <= residual+1e-9:
                    visit(stage,stations[k],B,elapsed+distance/speed+swap,flown+distance,
                          used|{k},swaps+((stage,k),),min(minimum_energy,residual-distance))

        visit(0,stations[l],B,0.,0.,frozenset({l}),(),B)
    return best,optimum


def _enumerate_empty(data, facilities, l, m):
    L,H = data['n_l'],data['n_h']
    stations=np.vstack((data['bases'],data['remote']))
    opened=[k for k in range(len(stations)) if facilities[k if k<L else L+H+k-L]]
    if l==m or l not in opened or m not in opened:
        return math.inf,[]
    best=math.inf
    times=[]

    def visit(u,elapsed,distance,used):
        nonlocal best,times
        leg=float(np.linalg.norm(stations[u]-stations[m]))
        if leg<=data['battery']+1e-9:
            duration=elapsed+leg/data['speed']+data['swap_time']
            cost=data['flight_cost']*(distance+leg)+data['fleet_cost']*duration
            if cost<best-1e-10:
                best=float(cost);times=[duration]
            elif abs(cost-best)<=1e-10:
                times.append(duration)
        for k in opened:
            if k in used or k==m:
                continue
            leg=float(np.linalg.norm(stations[u]-stations[k]))
            if leg<=data['battery']+1e-9:
                visit(k,elapsed+leg/data['speed']+data['swap_time'],distance+leg,used|{k})

    visit(l,0.,0.,{l})
    return best,times


def test_empirical_sync_expectation_matches_explicit_average_and_path_enumeration():
    data=_fixture([[0,0],[6,0]],[[1,0],[5,0]],[[3,0]],battery=4.3)
    data['ready_samples']=np.asarray([.01,.02,.09,.5])
    data['ready_prefix']=np.r_[0.,np.cumsum(data['ready_samples'])]
    facility=np.ones(5)
    answer=oracle.price_design(data,facility)
    for od,l,m in itertools.product(range(data['n_od']),range(2),range(2)):
        cost,records=_enumerate_mission(data,facility,od,l,m)
        assert answer['cost'][od,l,m]==pytest.approx(cost,abs=1e-9)
        assert any(abs(answer['time'][od,l,m]-r['time'])<1e-9 for r in records)


def _assert_result_matches_enumeration(data,facilities):
    answer=oracle.price_design(data,facilities)
    assert answer['complete'] is True
    for od in range(data['n_od']):
        for l in range(data['n_l']):
            for m in range(data['n_l']):
                expected,records=_enumerate_mission(data,facilities,od,l,m)
                actual=answer['cost'][od,l,m]
                if math.isinf(expected):
                    assert math.isinf(actual),(facilities,od,l,m,actual)
                else:
                    assert actual==pytest.approx(expected,abs=2e-9),(facilities,od,l,m)
                    actual_j=int(answer['j'][od,l,m])
                    valid=[r for r in records if r['j']==actual_j]
                    assert valid,(facilities,od,l,m,actual_j,records)
                    assert min(abs(answer['time'][od,l,m]-r['time']) for r in valid)<2e-9
                    assert all(r['i']!=r['j'] and r['minimum_energy']>=-1e-9 for r in valid)
    for l in range(data['n_l']):
        for m in range(data['n_l']):
            expected,times=_enumerate_empty(data,facilities,l,m)
            if math.isinf(expected):
                assert math.isinf(answer['empty_cost'][l,m]),(facilities,l,m)
            else:
                assert answer['empty_cost'][l,m]==pytest.approx(expected,abs=2e-9),(facilities,l,m)
                assert min(abs(answer['empty_time'][l,m]-t) for t in times)<2e-9
    return answer


def test_every_facility_pattern_matches_independent_resource_enumeration():
    data=_fixture([[0.,0.],[8.2,.15]],[[1.6,.2],[3.4,1.1],[6.8,.4]],[[4.1,-.2]])
    n=data['n_l']+data['n_h']+data['n_k']
    for pattern in itertools.product((0,1),repeat=n):
        _assert_result_matches_enumeration(data,np.asarray(pattern))


@pytest.mark.parametrize('open_lockers',[0,1])
def test_empty_repositioning_remains_available_without_two_open_lockers(open_lockers):
    data=_fixture([[0.,0.],[3.,0.]],[[.8,.2],[2.2,.3]],battery=4.)
    facility=np.asarray([1,1,open_lockers,0])
    answer=_assert_result_matches_enumeration(data,facility)
    assert np.isfinite(answer['empty_cost'][0,1])
    assert np.isinf(answer['cost']).all()


def test_soc_does_not_reset_at_pickup_or_delivery_locker():
    data=_fixture([[0.,0.]],[[2.5,0.],[5.,0.]],battery=6.,
                  origins=[[0.,0.]],destinations=[[5.,0.]],od_pairs=[(0,0)])
    answer=_assert_result_matches_enumeration(data,np.ones(3,dtype=int))
    # Each individual segment can fit in B=6, but no complete two-locker
    # roundtrip has enough battery without an additional station.
    assert np.isinf(answer['cost']).all()


def test_pickup_synchronization_max_and_terminal_swap_are_in_duration_and_cost():
    data=_fixture([[0.,0.]],[[2.,0.],[4.,0.]],battery=9.,ready=.6,
                  origins=[[0.,0.]],destinations=[[4.,0.]],od_pairs=[(0,0)])
    data['first_cost']=np.asarray([[0.,100.]])
    data['terminal_cost']=np.asarray([[100.,0.]])
    data['ground_time']=np.asarray([[.2,.4]])
    data['ground']=np.asarray([0.])
    answer=_assert_result_matches_enumeration(data,np.ones(3,dtype=int))
    duration=max(2/data['speed'],.6+.2)+(2+4)/data['speed']+data['swap_time']
    expected=8*data['flight_cost']+duration*data['fleet_cost']
    assert answer['cost'][0,0,0]==pytest.approx(expected)
    assert answer['time'][0,0,0]==pytest.approx(duration)
    assert answer['j'][0,0,0]==1


def test_multiple_swaps_and_station_revisits_in_different_stages():
    data=_fixture([[0.,0.],[12.,0.]],[[1.,0.],[11.,0.]],[[4.,0.],[8.,0.]],battery=5.,
                  origins=[[0.,0.]],destinations=[[11.,0.]],od_pairs=[(0,0)])
    data['first_cost']=np.asarray([[0.,100.]])
    data['terminal_cost']=np.asarray([[100.,0.]])
    facilities=np.ones(6,dtype=int)
    answer=_assert_result_matches_enumeration(data,facilities)
    expected,records=_enumerate_mission(data,facilities,0,0,0)
    assert math.isfinite(expected)
    assert min(len(r['swaps']) for r in records)>=5
    assert any(len({station for stage,station in r['swaps']})<len(r['swaps']) for r in records)
    assert math.isfinite(answer['cost'][0,0,1])
    facilities[-1]=0
    closed_answer=_assert_result_matches_enumeration(data,facilities)
    assert math.isinf(closed_answer['cost'][0,0,1])


def test_station_prefix_front_retains_time_cost_tradeoffs():
    # Two hops through (3,2) are faster but longer than three hops through
    # (1.5,0),(4.5,0). Both prefixes must survive before synchronization.
    xy=np.asarray([[0.,0.],[3.,2.],[1.5,0.],[4.5,0.],[6.,0.]])
    dist=np.sqrt(((xy[:,None,:]-xy[None,:,:])**2).sum(axis=2))
    fronts=oracle._arrival_fronts(dist,0,4.,10.,.2,.4)
    paths=[]
    for length in range(4):
        for middle in itertools.permutations((1,2,3),length):
            route=(0,)+middle+(4,)
            legs=[dist[a,b] for a,b in zip(route,route[1:])]
            if max(legs)<=4.+1e-9:
                paths.append((sum(legs)/10.+.4*len(legs),.2*sum(legs)))
    expected=[p for p in paths if not any(q[0]<=p[0]+1e-12 and q[1]<=p[1]+1e-12
                                        and (q[0]<p[0]-1e-12 or q[1]<p[1]-1e-12) for q in paths)]
    assert len(fronts[4])>=2
    for point in expected:
        assert any(np.allclose(point,other,atol=1e-10,rtol=0) for other in fronts[4])
    for point in fronts[4]:
        assert any(np.allclose(point,other,atol=1e-10,rtol=0) for other in expected)
