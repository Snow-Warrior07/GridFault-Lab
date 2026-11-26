from copy import deepcopy
import pytest
from gridfault.inference import InputError,choose_profile,validate_record
from gridfault.simulator import simulate,to_record
from gridfault.config import CHANNELS_S,CHANNELS_R


def record():return to_record(simulate()['phasors'])


def test_auto_fallback_is_measurement_aware():
    r=record()
    assert choose_profile(r)[0]=='phasor_double'
    r['synchronized']=False
    assert choose_profile(r)[0]=='rms_double'
    r['phasors']={k:v for k,v in r['phasors'].items() if k in CHANNELS_S}
    assert choose_profile(r)[0]=='rms_single'


def test_receiving_only_is_a_separately_trained_profile():
    r=record();r['phasors']={k:v for k,v in r['phasors'].items() if k in CHANNELS_R}
    assert choose_profile(r)[0]=='rms_receiving'


@pytest.mark.parametrize('key,value',[('line_id','L1-2'),('base_kv',230),('line_length_km',200),('schema_version',True),('synchronized','true')])
def test_unsupported_metadata_is_rejected(key,value):
    r=record();r[key]=value
    with pytest.raises(InputError):validate_record(r)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),'1.0',True])
def test_bad_measurements_are_rejected(value):
    r=record();r['phasors']['s_va']['re']=value
    with pytest.raises(InputError):validate_record(r)


def test_missing_measurements_are_not_zero_filled():
    r=record();r['phasors']={}
    with pytest.raises(InputError,match='No trained profile'):choose_profile(r)


def test_conflicting_rms_and_phasor_is_rejected():
    r=record();r['rms']={'s_va':99}
    with pytest.raises(InputError,match='disagree'):validate_record(r)


def test_unsynchronized_phasor_request_is_rejected():
    r=record();r['synchronized']=False
    with pytest.raises(InputError,match='unsynchronized'):choose_profile(r,'phasor_double')


def test_null_measurement_objects_are_rejected():
    r=record();r['rms']=None
    with pytest.raises(InputError,match='objects'):validate_record(r)


def test_overflowing_numeric_input_is_a_readable_error():
    r=record();r['phasors']['s_va']['re']=10**400
    with pytest.raises(InputError,match='Non-numeric'):validate_record(r)

