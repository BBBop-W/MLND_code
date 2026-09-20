"""Production data/config contracts, calibration semantics and sensitivity."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data.instances import load_instance, configure
from src.data.choice import pickup_probability
from src.data.calibration import estimate_dispatch_delay, postfetch_records, deadline_table
from src.data.instance_io import ROOT, sha256


@pytest.mark.parametrize('hubs,bases,swaps,ods,demand', [
    (11,2,2,2723,2114.7142857573), (24,2,3,6445,4906.857142958),
    (43,3,5,9162,6828.5714287189), (68,5,7,10571,7726.2857144554),
    (87,6,9,11293,8153.0000001821), (120,8,13,11427,8196.7142858995)])
def test_canonical_instance_schema_and_operating_units(hubs,bases,swaps,ods,demand):
    d=load_instance(hubs, {'data': {'demand_hubs': None}})
    assert (d['n_l'],d['n_k'],d['n_od'])==(bases,swaps,ods)
    assert d['demand'].sum()==pytest.approx(demand)
    assert d['model_version']=='path_cqn_v1'
    assert all(key not in d for key in ('rho_max','kappa','ready','ready_samples'))
    assert d['flight_times'].shape==(bases+hubs+swaps,)*2
    assert np.allclose(d['level_capacity'],np.broadcast_to(30.4*np.arange(1,6),(hubs,5)))
    assert np.allclose(d['level_cost'],np.broadcast_to(12.6*np.arange(1,6),(hubs,5)))
    assert d['airline_hh_cost_per_km_hour']==pytest.approx(.5175)
    assert d['base_site_cost']==pytest.approx(20000/21900)
    assert np.allclose(d['terminal_time'],d['last_ground_time']+4.519596062942166/60)
    assert np.allclose(d['terminal_cost'],(1-d['pickup_share'])*d['last_rider_cost'])
    assert np.array_equal(d['direct_rider_hours'],d['delivery_deadline'])
    assert np.all(d['delivery_deadline']>0)
    assert d['c0']==pytest.approx(d['demand']@d['ground'])
    assert 'mlnd_path/' not in json.dumps(d['provenance'])
    assert '/results/' not in json.dumps(d['provenance'])


def test_input_assets_match_integrity_manifest():
    root=ROOT/'data/model_inputs'
    manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        assert sha256(root/name)==digest


def test_config_overrides_recalculate_cost_demand_and_time_components():
    base=load_instance(11)
    cfg=load_config(overrides=dict(data=dict(demand_scale=2),depreciation=dict(years=10),
        operations=dict(rider_minutes_per_km=6.,terminal_dispatch_minutes=3.),
        costs=dict(rider_initial_charge=8.,rider_additional_per_km=2.)))
    new=load_instance(11,cfg)
    assert np.array_equal(new['demand'],2*base['demand'])
    for key in ('base_cost','hub_cost','swap_cost','fleet_cost','base_site_cost','hub_site_cost',
                'swap_site_cost','airline_hh_cost_per_km_hour'):
        assert new[key]==pytest.approx(base[key]/2)
    assert np.allclose(new['ground_time'],base['ground_time']*6/5.733438163127652)
    assert new['terminal_dispatch_time']==pytest.approx(3/60)
    assert np.all(new['ground']>base['ground'])
    assert np.array_equal(new['flight_times'],base['flight_times'])
    assert np.array_equal(new['pickup_share'],base['pickup_share'])
    assert load_config()['depreciation']['years']==5


def test_choice_uses_kilometers_and_hard_cutoff():
    d=np.array([0.,.1,.799999,.8,1.])
    result=pickup_probability(d)
    assert result[1]==pytest.approx(1/(1+np.exp(.29)))
    assert result[2]>0
    assert np.array_equal(result[3:],[0,0])
    all_pickup=configure(11,settings=dict(choice=dict(locker_asc=700.,max_walk_km=100.)))
    assert np.count_nonzero(all_pickup['terminal_cost'])==0


@pytest.mark.parametrize('changes', [dict(unknown=1),dict(operations=dict(max_cabinets=1.5)),
    dict(operations=dict(failure_probability=1)),dict(costs=dict(cabinet_investment=-1)),
    dict(data=dict(demand_scale=0)),dict(choice=dict(max_walk_km=-1)),
    dict(solver=dict(threads=0)),dict(operations=dict(rider_minutes_per_km=0))])
def test_invalid_config_is_rejected(changes):
    with pytest.raises(ValueError):load_config(overrides=changes)


def test_dispatch_filter_excludes_unaccepted_prebook_and_bad_clock_records():
    raw=pd.DataFrame(dict(is_courier_grabbed=[1,1,0,1,1],is_prebook=[0,0,0,1,0],
        platform_order_time=[100,100,0,100,100],grab_time=[160,220,0,220,99],
        fetch_time=[200,230,0,230,200],arrive_time=[300,300,0,300,300]))
    result=estimate_dispatch_delay(raw)
    assert result['valid_rows']==2
    assert result['mean_minutes']==1.5
    assert 'acceptance' in result['observed_event']


def test_observed_deadline_excludes_preparation_and_shrinks_small_samples():
    raw=pd.DataFrame(dict(dt_seq_id=[1]*5,pickup_longitude=[0.]*5,pickup_latitude=[0.]*5,
        dropoff_longitude=[0.]*5,dropoff_latitude=[0.]*5,customer_pay_time=['12:00']*5,
        fetch_time=['12:10','12:00','12:20','12:10','invalid'],
        arrived_time=['12:25','12:25','12:10','12:10','12:25']))
    metadata=dict(selected_days=['1'],projection_center_lat=0.,projection_center_lon=0.,
        data_config=dict(demand_hour_start=12,demand_hour_end=13,map_half_size_km=3.))
    records,audit=postfetch_records(raw,metadata)
    assert records.minutes.tolist()==[15.]
    assert audit['rejected_chronology_or_nonfinite']==4
    d=dict(n_od=2,origins=np.array([[.1,.1]]),destinations=np.array([[.1,.1],[1.1,.1]]),
        od_origin=np.array([0,0]),od_destination=np.array([0,1]))
    records=pd.DataFrame(dict(ox=[.1,.1],oy=[.1,.1],dx=[.1,.1],dy=[.1,.1],minutes=[10.,20.]))
    table=deadline_table(d,records,dict(data_config=dict(od_grid_km=.5)),slope=5.,intercept=8.,prior_weight=5.)
    assert table.observed_count.tolist()==[2,0]
    assert table.deadline_minutes.iloc[0]==pytest.approx((30+5*8)/7)
    assert table.deadline_minutes.iloc[1]==13.


def test_raw_orders_generate_audited_nested_family(tmp_path):
    import csv
    from src.data.scales import build, NanshanScaleLevel
    rows=[]
    for k in range(36):
        rows.append(dict(order_seq_id=k,dt_seq_id='1',customer_pay_time='12:30',
            pickup_longitude=113.94+(k%6)*.005,pickup_latitude=22.55+(k//6)*.005,
            dropoff_longitude=113.94+((k+1)%6)*.005,dropoff_latitude=22.55+((k+2)//6)*.005))
    raw=tmp_path/'orders.csv'
    with raw.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    manifest_path=build(raw,tmp_path/'family',
        levels=(NanshanScaleLevel(2.,2,1,1),NanshanScaleLevel(5.,3,1,1)),dates=('1',))
    manifest=json.loads(manifest_path.read_text())
    assert manifest['raw_demand_order_multisets_nested']
    assert manifest['orders_sampled'] is False
    assert len(manifest['levels'])==2
    for case in manifest['levels']:
        assert case['demand_order_count']==36
        assert case['total_demand_per_hour']==36
        for file,digest in case['canonical_file_sha256'].items():
            assert sha256(Path(case['instance_dir'])/file)==digest
    with pytest.raises(FileExistsError):
        build(raw,tmp_path/'family',levels=(NanshanScaleLevel(2.,2,1,1),),dates=('1',))
