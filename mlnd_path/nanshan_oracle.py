"""Implicit complete-mission oracle for the historical Nanshan instances.

No OD, base pair, locker pair, or battery-swap station is screened.  A mission
has exactly one pickup and one delivery locker (different locations). Energy
is carried across both lockers; only a station visit restores the battery.
All travel costs are positive, so station cycles can be removed. Before pickup
we retain the Pareto frontier of arrival time and flight cost; after pickup
the positive fleet price makes the objective additive.
"""
from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import time
from pathlib import Path

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
MEANS = {11:15.0178,24:15.2137,43:15.3356,68:15.3205,87:15.3947,120:15.3936}


def _rows(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def load_nanshan(hubs=11, rider_speed=18.0):
    directory = next((ROOT/'data/instances/nanshan_nested_h120').glob(f'H{hubs:03d}_*'))
    sites = _rows(directory/'sites.csv')
    nodes = _rows(directory/'od_nodes.csv')
    orders = _rows(directory/'od_demand.csv')
    cfg_path = ROOT/'results/nanshan_h120_final_schema5/run_config.json'
    cfg = json.loads(cfg_path.read_text())['model']
    xy = lambda rows: np.asarray([[float(r['x_km']),float(r['y_km'])] for r in rows])
    bases=xy([r for r in sites if r['type']=='L'])
    lockers=xy([r for r in sites if r['type']=='H'])
    remote=xy([r for r in sites if r['type']=='E'])
    origins=[r for r in nodes if r['role']=='origin']
    destinations=[r for r in nodes if r['role']=='destination']
    oi={r['node_id']:i for i,r in enumerate(origins)}
    di={r['node_id']:i for i,r in enumerate(destinations)}
    dxy=lambda a,b: np.sqrt(((a[:,None,:]-b[None,:,:])**2).sum(axis=2))
    tariff=lambda distance: cfg['rider_initial_charge_cny']+cfg['rider_incremental_cost_per_km']*np.maximum(0,cfg['road_distance_factor']*distance-cfg['rider_included_distance_km'])
    dist_oh=dxy(xy(origins),lockers)
    dist_dh=dxy(xy(destinations),lockers)
    ch=cfg['choice']
    share=1/(1+np.exp(-(ch['locker_asc']+ch['delivery_time_coefficient']*(ch['home_eta_minutes']-ch['locker_eta_minutes'])+ch['price_coefficient']*ch['locker_discount_cny']-ch['walk_distance_coefficient']*dist_dh)))
    share[dist_dh>ch['max_walk_km']+1e-9]=0
    terminal=share*(cfg['locker_pickup_cost_per_order']+ch['locker_discount_cny'])+(1-share)*cfg['feeder_rider_cost_factor']*tariff(dist_dh)
    data=dict(name=f'Nanshan_H{hubs}',n_l=len(bases),n_h=len(lockers),n_k=len(remote),
              bases=bases,hubs=lockers,remote=remote,origins=xy(origins),destinations=xy(destinations),
              od_origin=np.asarray([oi[r['origin_node']] for r in orders]),
              od_destination=np.asarray([di[r['destination_node']] for r in orders]),
              demand=np.asarray([float(r['lambda_per_hour']) for r in orders]),
              ground=np.asarray([float(tariff(float(r['direct_distance_km']))) for r in orders]),
              first_cost=cfg['feeder_rider_cost_factor']*tariff(dist_oh),
              ground_time=cfg['road_distance_factor']*dist_oh/rider_speed,
              terminal_cost=terminal,pickup_share=share,ready=MEANS[hubs]/60,
              battery=cfg['battery_range_km'],speed=cfg['drone_speed_mps']*3.6,
              flight_cost=cfg['flight_cost_per_km'],swap_time=cfg['swap_service_time_hours'],
              fleet_cost=cfg['drone_cost_per_hour'],rho_max=max(cfg['availability_levels']),
              kappa=30.0,base_cost=cfg['base_cost_per_hour'],hub_cost=cfg['locker_fixed_cost_per_hour'],
              swap_cost=cfg['swap_cost_per_hour'])
    data['n_od']=len(orders)
    data['c0']=float(data['demand']@data['ground'])
    data['provenance']=dict(directory=str(directory),config=str(cfg_path),
        sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob('*.csv')},
        orders_sampled=False,demand_rescaled=False,screening_is_exact=True,
        historical_top_k_screening_used=False,deterministic_composite_pickup_minutes=MEANS[hubs],
        rider_speed_kmh_scenario=rider_speed,kappa_per_hour_scenario=30.0,
        mission_energy='10 km shared across every stage; station-only reset; same loaded/empty rate',
        fixed_corridor_cost='not charged: new model selects facilities and paths, no corridor binaries',
        locker_capacity='not present in the new delay-node path formulation',
        pickup_discount='paid on self-pickup orders, included in terminal expected cost')
    from .nanshan_timing import load_empirical_timing, timing_metadata
    data['ready_samples']=load_empirical_timing(hubs)
    data['ready_prefix']=np.r_[0.,np.cumsum(data['ready_samples'])]
    data['ready']=float(np.mean(data['ready_samples']))
    data['provenance']['pickup_timing']=timing_metadata(hubs)
    data['provenance']['pickup_expectation']='Exact expectation under the empirical marginal distribution; no optimization scenarios'
    return data


@njit(cache=True)
def _station_shortest(dist,battery,speed,flight,swap,fleet):
    n=len(dist)
    cost=np.full((n,n),np.inf); dur=np.full((n,n),np.inf)
    for i in range(n):
        cost[i,i]=0;dur[i,i]=0
        for j in range(n):
            if i!=j and dist[i,j]<=battery+1e-9:
                dur[i,j]=dist[i,j]/speed+swap
                cost[i,j]=flight*dist[i,j]+fleet*dur[i,j]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                value=cost[i,k]+cost[k,j]
                if value<cost[i,j]-1e-12:
                    cost[i,j]=value;dur[i,j]=dur[i,k]+dur[k,j]
    return cost,dur


@njit(cache=True)
def _body(sh,hh,sc,st,bpos,battery,speed,flight,swap,fleet):
    """After pickup: first station a determines residual battery at locker i."""
    S,H=sh.shape;L=len(bpos)
    # Return from j after arriving from last loaded station b.
    ret=np.full((S,H,L),np.inf); rett=np.full((S,H,L),np.inf)
    for b in range(S):
        for j in range(H):
            energy=battery-sh[b,j]
            if energy < -1e-9: continue
            for m in range(L):
                if bpos[m]<0:continue
                for k in range(S):
                    if sh[k,j]<=energy+1e-9:
                        t=sh[k,j]/speed+swap+st[k,bpos[m]]
                        c=flight*sh[k,j]+fleet*(sh[k,j]/speed+swap)+sc[k,bpos[m]]
                        if c<ret[b,j,m]:ret[b,j,m]=c;rett[b,j,m]=t
    tail=np.full((S,H,L),np.inf); tailt=np.full((S,H,L),np.inf)
    for k in range(S):
        for j in range(H):
            for m in range(L):
                for b in range(S):
                    if sh[b,j]<=battery+1e-9:
                        c=sc[k,b]+(flight+fleet/speed)*sh[b,j]+ret[b,j,m]
                        if c<tail[k,j,m]:
                            tail[k,j,m]=c;tailt[k,j,m]=st[k,b]+sh[b,j]/speed+rett[b,j,m]
    cost=np.full((S,H,H,L),np.inf);dur=np.full((S,H,H,L),np.inf)
    for a in range(S):
        for i in range(H):
            energy=battery-sh[a,i]
            if energy < -1e-9: continue
            for j in range(H):
                if i==j:continue
                for m in range(L):
                    if bpos[m]<0:continue
                    if hh[i,j]<=energy+1e-9:
                        after=energy-hh[i,j]
                        for k in range(S):
                            if sh[k,j]<=after+1e-9:
                                t=(hh[i,j]+sh[k,j])/speed+swap+st[k,bpos[m]]
                                c=flight*(hh[i,j]+sh[k,j])+fleet*((hh[i,j]+sh[k,j])/speed+swap)+sc[k,bpos[m]]
                                if c<cost[a,i,j,m]:cost[a,i,j,m]=c;dur[a,i,j,m]=t
                    for k in range(S):
                        if sh[k,i]<=energy+1e-9:
                            c=(flight+fleet/speed)*sh[k,i]+fleet*swap+tail[k,j,m]
                            if c<cost[a,i,j,m]:
                                cost[a,i,j,m]=c;dur[a,i,j,m]=sh[k,i]/speed+swap+tailt[k,j,m]
    return cost,dur


def _arrival_fronts(dist,l,battery,speed,flight,swap):
    """All nondominated (elapsed time, monetary flight cost) station prefixes."""
    fronts=[[] for _ in range(len(dist))]
    fronts[l]=[(0.,0.)];heap=[(0.,0.,l)]
    while heap:
        elapsed,cost,u=heapq.heappop(heap)
        if (elapsed,cost) not in fronts[u]:continue
        for v in range(len(dist)):
            if u==v or dist[u,v]>battery+1e-9:continue
            nt=elapsed+dist[u,v]/speed+swap;nc=cost+flight*dist[u,v]
            if any(t<=nt+1e-12 and c<=nc+1e-12 for t,c in fronts[v]):continue
            fronts[v]=[(t,c) for t,c in fronts[v] if not(nt<=t+1e-12 and nc<=c+1e-12)]
            fronts[v].append((nt,nc));heapq.heappush(heap,(nt,nc,v))
    return fronts


@njit(cache=True)
def _combine(body,bodyt,sh,prefix_a,prefix_t,prefix_c,ground,first,samples,sample_sum,speed,flight,fleet):
    O=ground.shape[0];S,H,_,L=body.shape
    cost=np.full((O,H,L),np.inf);dur=np.full((O,H,L),np.inf)
    for o in range(O):
        for z in range(len(prefix_a)):
            a=prefix_a[z]
            for i in range(H):
                arrive=prefix_t[z]+sh[a,i]/speed
                n=len(samples)
                count=np.searchsorted(samples,arrive-ground[o,i],side='right')
                sync=(count*arrive+(n-count)*ground[o,i]+sample_sum[n]-sample_sum[count])/n
                before=prefix_c[z]+flight*sh[a,i]+fleet*sync+first[o,i]
                for j in range(H):
                    for m in range(L):
                        c=before+body[a,i,j,m]
                        if c<cost[o,j,m]:cost[o,j,m]=c;dur[o,j,m]=sync+bodyt[a,i,j,m]
    return cost,dur


@njit(cache=True)
def _finish(cost,dur,terminal,oi,di,ground,hub_ids):
    D=len(oi);H=cost.shape[1];L=cost.shape[2]
    out=np.full((D,L),np.inf);times=np.zeros((D,L));js=np.full((D,L),-1,np.int64)
    for od in range(D):
        for j in range(H):
            for m in range(L):
                c=cost[oi[od],j,m]+terminal[di[od],j]-ground[od]
                if c<out[od,m]:out[od,m]=c;times[od,m]=dur[oi[od],j,m];js[od,m]=hub_ids[j]
    return out,times,js


def price_design(data, facility=None):
    """Exact cheapest open mission for every OD and L-L; no path enumeration."""
    start=time.perf_counter();L,H,K=data['n_l'],data['n_h'],data['n_k']
    if facility is None:facility=np.ones(L+H+K,dtype=int)
    facility=np.asarray(facility)
    hs=np.flatnonzero(facility[L:L+H]>.5)
    stations=np.vstack((data['bases'],data['remote']))
    ss=np.r_[np.flatnonzero(facility[:L]>.5),L+np.flatnonzero(facility[L+H:]>.5)]
    bpos=np.asarray([int(np.where(ss==l)[0][0]) if l in ss else -1 for l in range(L)])
    best=np.full((data['n_od'],L,L),np.inf);times=np.zeros_like(best);js=np.full(best.shape,-1,dtype=np.int64)
    ec=np.full((L,L),np.inf);et=np.zeros((L,L))
    if len(ss)==0:
        return dict(cost=best,time=times,j=js,empty_cost=ec,empty_time=et,wall=time.perf_counter()-start,facility=facility.tolist(),complete=True)
    xy=stations[ss];hxy=data['hubs'][hs]
    distances=lambda a,b:np.sqrt(((a[:,None,:]-b[None,:,:])**2).sum(axis=2))
    sd=distances(xy,xy);sh=distances(xy,hxy);hh=distances(hxy,hxy)
    args=[data[k] for k in ['battery','speed','flight_cost','swap_time','fleet_cost']]
    sc,st=_station_shortest(sd,*args)
    for l in range(L):
        for m in range(L):
            if l!=m and bpos[l]>=0 and bpos[m]>=0:
                ec[l,m]=sc[bpos[l],bpos[m]];et[l,m]=st[bpos[l],bpos[m]]
    if len(hs)<2:
        return dict(cost=best,time=times,j=js,empty_cost=ec,empty_time=et,wall=time.perf_counter()-start,facility=facility.tolist(),complete=True)
    body,bodyt=_body(sh,hh,sc,st,bpos,*args)
    prefixes=0
    for l in range(L):
        if bpos[l]<0:continue
        fronts=_arrival_fronts(sd,bpos[l],*args[:-1])
        records=[(a,t,c) for a,f in enumerate(fronts) for t,c in f]
        prefixes+=len(records)
        pa=np.asarray([v[0] for v in records],dtype=np.int64)
        pt=np.asarray([v[1] for v in records]);pc=np.asarray([v[2] for v in records])
        samples=data.get('ready_samples',np.asarray([data['ready']]))
        prefix=data.get('ready_prefix',np.r_[0.,np.cumsum(samples)])
        cost,dur=_combine(body,bodyt,sh,pa,pt,pc,data['ground_time'][:,hs],data['first_cost'][:,hs],samples,prefix,data['speed'],data['flight_cost'],data['fleet_cost'])
        best[:,l,:],times[:,l,:],js[:,l,:]=_finish(cost,dur,data['terminal_cost'][:,hs],data['od_origin'],data['od_destination'],data['ground'],hs)
    return dict(cost=best,time=times,j=js,empty_cost=ec,empty_time=et,wall=time.perf_counter()-start,
                facility=facility.tolist(),complete=True,prefix_labels=prefixes,
                implicit_basic_paths=int(data['n_od']*L*L*H*(H-1)),enumerated_complete_paths=0)
