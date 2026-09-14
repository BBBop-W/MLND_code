"""Empirical payment-to-pickup times matched to canonical Nanshan crops.

The raw file is read only. Every cached observation uses the canonical seven
days, 12:00--13:00 payment window, both-endpoint crop, and all business lines.
Clock-invalid records are excluded from timing estimation, not from OD demand.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INSTANCE_ROOT = ROOT / 'data/instances/nanshan_nested_h120'
CACHE_DIR = ROOT / 'mlnd_path/nanshan_integer_20260914/timing'
CACHE_SCHEMA = 1


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _clock_minutes(value):
    try:
        hour, minute = map(int, value.split(':'))
    except (ValueError, AttributeError):
        return None
    return 60*hour+minute if 0<=hour<24 and 0<=minute<60 else None


def _paths(hubs):
    return CACHE_DIR / f'H{int(hubs):03d}.npz', CACHE_DIR / 'manifest.json'


def _atomic_json(path, value):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tmp', dir=path.parent,
                                     encoding='utf-8', delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    temporary.replace(path)


def build_timing_cache(force=False):
    """Build all six small compressed arrays; return their source manifest."""
    family_path = INSTANCE_ROOT / 'scale_manifest.json'
    family = json.loads(family_path.read_text())
    full = Path(family['full_parent_instance'])
    metadata = json.loads((full / 'metadata.json').read_text())
    raw_path = Path(metadata['raw_path'])
    raw_stat = raw_path.stat()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = CACHE_DIR / 'manifest.json'
    family_hash = _sha256(family_path)
    if not force and manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if (old.get('schema_version')==CACHE_SCHEMA
                and old.get('raw_file_size_bytes')==raw_stat.st_size
                and old.get('raw_file_mtime_ns')==raw_stat.st_mtime_ns
                and old.get('family_manifest_sha256')==family_hash
                and all((CACHE_DIR/r['file']).exists() for r in old.get('levels',{}).values())
                and len(old.get('levels',{}))==len(family['levels'])):
            return old
    contract=family['family_contract']
    selected_days=set(map(str,contract['selected_days']))
    start_hour,end_hour=contract['demand_hours']
    center_lon=contract['projection_center_lon']
    center_lat=contract['projection_center_lat']
    km_per_lon=111.32*math.cos(math.radians(center_lat))
    levels={int(item['candidate_counts']['H']):item for item in family['levels']}
    observations={h:[] for h in levels}
    counts={h:Counter() for h in levels}
    businesses={h:Counter() for h in levels}
    days={h:Counter() for h in levels}
    total_raw_rows=0
    with raw_path.open(newline='',encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            total_raw_rows+=1
            if row['dt_seq_id'] not in selected_days:
                continue
            pay=_clock_minutes(row['customer_pay_time'])
            if pay is None or not start_hour*60<=pay<end_hour*60:
                continue
            ox=(float(row['pickup_longitude'])-center_lon)*km_per_lon
            oy=(float(row['pickup_latitude'])-center_lat)*111.32
            dx=(float(row['dropoff_longitude'])-center_lon)*km_per_lon
            dy=(float(row['dropoff_latitude'])-center_lat)*111.32
            extent=max(abs(ox),abs(oy),abs(dx),abs(dy))
            fetch=_clock_minutes(row['fetch_time'])
            arrival=_clock_minutes(row['arrived_time'])
            for h,level in levels.items():
                if extent>float(level['crop_half_size_km']):
                    continue
                counts[h]['demand_window_rows']+=1
                businesses[h][row['biz_line']]+=1
                if fetch is None or arrival is None:
                    counts[h]['invalid_clock_format']+=1
                elif not pay<fetch<=arrival:
                    counts[h]['invalid_event_order']+=1
                else:
                    observations[h].append((fetch-pay)/60.)
                    days[h][row['dt_seq_id']]+=1
    raw_hash=_sha256(raw_path)
    if raw_hash!=contract['raw_file_sha256']:
        raise ValueError('The raw data hash differs from the canonical Nanshan family')
    output={}
    for h,level in levels.items():
        expected_count=int(level['demand_order_count'])
        if counts[h]['demand_window_rows']!=expected_count:
            raise ValueError(f'H{h}: raw crop/window count differs from canonical demand')
        samples=np.sort(np.asarray(observations[h],dtype=np.float64))
        if len(samples)==0 or not np.all(np.isfinite(samples)) or np.any(samples<=0):
            raise ValueError(f'H{h}: expected positive finite timing observations')
        prefix=np.r_[0.,np.cumsum(samples)]
        array_path,_=_paths(h)
        with tempfile.NamedTemporaryFile(mode='wb',suffix='.npz.tmp',dir=CACHE_DIR,delete=False) as handle:
            temporary=Path(handle.name)
            np.savez_compressed(handle,xi_sorted=samples,prefix_sum=prefix)
        temporary.replace(array_path)
        output[str(h)]=dict(file=array_path.name,sha256=_sha256(array_path),
            file_size_bytes=array_path.stat().st_size,unit='hour',crop_half_size_km=level['crop_half_size_km'],
            canonical_demand_orders=expected_count,canonical_od_count=level['od_market_count'],
            valid_timing_samples=len(samples),excluded_timing_samples=expected_count-len(samples),
            exclusions={k:v for k,v in counts[h].items() if k!='demand_window_rows'},
            demand_business_counts=dict(sorted(businesses[h].items())),
            valid_timing_samples_by_day=dict(sorted(days[h].items())),
            mean_hours=float(samples.mean()),mean_minutes=float(samples.mean()*60),
            median_minutes=float(np.median(samples)*60),p90_minutes=float(np.quantile(samples,.9)*60),
            p95_minutes=float(np.quantile(samples,.95)*60),
            prefix_total_hours=float(prefix[-1]),prefix_sum_length=len(prefix))
    manifest=dict(schema_version=CACHE_SCHEMA,raw_file=str(raw_path),raw_file_sha256=raw_hash,
        raw_file_size_bytes=raw_stat.st_size,raw_file_mtime_ns=raw_stat.st_mtime_ns,
        total_raw_rows=total_raw_rows,family_manifest=str(family_path),family_manifest_sha256=family_hash,
        source_module_sha256=_sha256(Path(__file__)),selected_days=sorted(selected_days),
        payment_hour_window=[start_hour,end_hour],business_lines='all canonical business lines',
        projection_center_lon=center_lon,projection_center_lat=center_lat,
        spatial_rule='both original endpoints inside the canonical local-km square, inclusive',
        timing_rule='same-day payment < fetch <= delivery; xi=(fetch-payment)/60 hours',
        event_meaning='observed composite payment-to-pickup lead time',
        demand_records_removed=False,distribution='empirical distribution with equal observation weights',
        scenario_flow_variables=False,levels=output,
        total_array_size_bytes=sum(record['file_size_bytes'] for record in output.values()))
    _atomic_json(manifest_path,manifest)
    return manifest


def load_empirical_timing(hubs):
    """Return sorted xi observations in hours, building the shared cache if needed."""
    manifest=build_timing_cache()
    key=str(int(hubs))
    if key not in manifest['levels']:
        raise ValueError(f'Unsupported Nanshan locker scale: {hubs}')
    array_path,_=_paths(hubs)
    record=manifest['levels'][key]
    if _sha256(array_path)!=record['sha256']:
        raise ValueError(f'Timing cache checksum mismatch for H{hubs}')
    with np.load(array_path,allow_pickle=False) as archive:
        samples=archive['xi_sorted'].copy()
        prefix=archive['prefix_sum']
        if (len(samples)!=record['valid_timing_samples'] or len(prefix)!=len(samples)+1
                or prefix[0]!=0. or np.any(np.diff(samples)<0)
                or not np.allclose(prefix[1:]-prefix[:-1],samples,atol=2e-11,rtol=0)):
            raise ValueError(f'Invalid sorted samples or prefix sums for H{hubs}')
    return samples


def timing_metadata(hubs):
    """Return the per-crop statistics and source hashes for result provenance."""
    manifest=build_timing_cache()
    record=dict(manifest['levels'][str(int(hubs))])
    record.update(raw_file_sha256=manifest['raw_file_sha256'],
                  family_manifest_sha256=manifest['family_manifest_sha256'],
                  cache_manifest=str(CACHE_DIR/'manifest.json'),timing_rule=manifest['timing_rule'])
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args()
    result=build_timing_cache(force=args.force)
    print(json.dumps(dict(cache=str(CACHE_DIR),total_array_size_bytes=result['total_array_size_bytes'],
                          levels={h:{k:r[k] for k in ('valid_timing_samples','mean_minutes','sha256')}
                                  for h,r in result['levels'].items()}),ensure_ascii=False,indent=2))
