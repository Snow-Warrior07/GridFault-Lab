import gzip
import csv
import json
from pathlib import Path
import numpy as np
import pytest
from gridfault.dataset import load,row_phasors
from gridfault.features import PROFILES
from gridfault.inference import predict_record
from gridfault.simulator import to_record

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def rows():return load(ROOT)


def test_group_and_scenario_partitions_do_not_overlap(rows):
    for key in ['group_id','scenario_id']:
        assignments={}
        for r in rows:
            assert assignments.setdefault(r[key],r['split'])==r['split']


def test_distance_bands_are_really_held_out(rows):
    for r in rows:
        if r['event']=='internal' and r['split']!='test':
            assert not (.20<=r['location_pu']<=.25 or .70<=r['location_pu']<=.75)


def test_no_fault_position_for_non_internal_events(rows):
    assert all(r['location_pu'] is None for r in rows if r['event']!='internal')


def test_no_target_or_simulator_settings_enter_features():
    forbidden={'event','fault_type','location_pu','actual_fault_type','resistance_ohm','snr_db','group_id','scenario_id','split'}
    for features in PROFILES.values():assert not forbidden.intersection(features)


def test_exported_models_preserve_public_schema_and_finite_parameters():
    for profile,features in PROFILES.items():
        b=json.loads((ROOT/'site'/'assets'/f'model-{profile}.json').read_text())
        assert b['features']==features
        assert np.isfinite(b['bounds']).all()
        assert b['families']['forest']['location']['interval_radius_km']>0


def test_healthy_and_external_sample_have_no_location(rows):
    bundle=json.loads((ROOT/'site/assets/model-phasor_double.json').read_text())
    for event in ['healthy','external']:
        row=next(r for r in rows if r['event']==event and r['split']=='test' and r['snr_db']=='clean')
        pred=predict_record(to_record(row_phasors(row)),ROOT/'site/assets',bundle=bundle)
        assert pred['event']==event
        assert pred['location_km'] is None
        assert pred['interval_km'] is None


def test_out_of_domain_measurements_abstain():
    from gridfault.simulator import simulate
    r=to_record(simulate()['phasors'])
    r['phasors']['s_ia']={'re':1000000,'im':1000000}
    p=predict_record(r,ROOT/'site/assets')
    assert p['quality']=='out_of_domain'
    assert p['event'] is None and p['location_km'] is None

