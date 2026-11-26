import numpy as np
import pytest
from gridfault.config import FAULT_TYPES,PHASE_TO_SEQUENCE
from gridfault.features import double_ended_location
from gridfault.simulator import Conditions,simulate


@pytest.mark.parametrize('fault',FAULT_TYPES)
@pytest.mark.parametrize('d',[.03,.37,.94])
def test_known_fault_location_and_kcl(fault,d):
    r=simulate(fault_type=fault,location_pu=d,resistance_ohm=25)
    p=double_ended_location(r['phasors'])
    assert r['kcl_residual']<1e-10
    assert p['location_pu']==pytest.approx(d,abs=1e-9)
    assert p['residual']<1e-9


@pytest.mark.parametrize('event',['healthy','external'])
def test_non_internal_fault_has_no_fault_position_or_differential_current(event):
    r=simulate(event=event)
    p=r['phasors']
    assert r['location_pu'] is None
    assert max(abs(p['s_i'+x]+p['r_i'+x]) for x in 'abc')<1e-10


def test_balanced_network_has_negligible_zero_sequence():
    p=simulate(event='healthy')['phasors']
    seq=PHASE_TO_SEQUENCE@np.array([p[k] for k in ['s_va','s_vb','s_vc']])
    assert abs(seq[0])<1e-10
    assert abs(seq[2])<1e-10


def test_terminal_reference_reversal():
    from gridfault.config import CHANNELS_S,CHANNELS_R
    p=simulate(location_pu=.23)['phasors']
    swapped={**dict(zip(CHANNELS_S,[p[k] for k in CHANNELS_R])),**dict(zip(CHANNELS_R,[p[k] for k in CHANNELS_S]))}
    assert double_ended_location(swapped)['location_pu']==pytest.approx(.77,abs=1e-9)


def test_line_parameter_uncertainty_is_actually_simulated():
    r=simulate(Conditions(line_x_scale=1.05),location_pu=.4)
    oracle=double_ended_location(r['phasors'],r['actual_z'])
    nominal=double_ended_location(r['phasors'])
    assert abs(oracle['location_pu']-.4)<1e-9
    from gridfault.config import NOMINAL_Z
    assert not np.allclose(r['actual_z'],NOMINAL_Z)
    assert abs(nominal['location_pu']-.4)>1e-8

