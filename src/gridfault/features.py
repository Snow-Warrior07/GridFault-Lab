"""The only feature path, shared by training, CLI, and browser parity fixtures."""
from __future__ import annotations

import numpy as np
from .config import CHANNELS, CHANNELS_R, CHANNELS_S, NOMINAL_Z, PHASE_TO_SEQUENCE

PHASOR_FEATURES = CHANNELS + [f"{k}_{part}" for k in CHANNELS for part in ["re","im"]] + [
    f"{group}_seq{seq}" for group in ["sv","si","rv","ri"] for seq in range(3)
] + ["diff_a", "diff_b", "diff_c", "diff_ratio"]
PROFILES = {"rms_single": CHANNELS_S, "rms_receiving": CHANNELS_R, "rms_double": CHANNELS, "phasor_double": PHASOR_FEATURES}


def extract(phasors: dict) -> dict:
    f = {k:float(abs(phasors[k])) for k in CHANNELS if k in phasors}
    for k,v in phasors.items():
        f[k+"_re"],f[k+"_im"] = float(v.real),float(v.imag)
    if all(k in phasors for k in CHANNELS):
        arrays = [np.array([phasors[k] for k in CHANNELS[i:i+3]]) for i in range(0,12,3)]
        for name,values in zip(["sv","si","rv","ri"],arrays):
            for seq,value in enumerate(PHASE_TO_SEQUENCE@values):
                f[f"{name}_seq{seq}"] = float(abs(value))
        diff = arrays[1]+arrays[3]
        f.update(zip(["diff_a","diff_b","diff_c"],map(float,abs(diff))))
        f["diff_ratio"] = float(np.linalg.norm(diff)/(np.linalg.norm(arrays[1])+np.linalg.norm(arrays[3])+1e-12))
    return f


def double_ended_location(phasors: dict, z: np.ndarray = NOMINAL_Z) -> dict:
    vs = np.array([phasors[k] for k in CHANNELS_S[:3]])
    is_ = np.array([phasors[k] for k in CHANNELS_S[3:]])
    vr = np.array([phasors[k] for k in CHANNELS_R[:3]])
    ir = np.array([phasors[k] for k in CHANNELS_R[3:]])
    a = z@(is_+ir)
    b = vs-vr+z@ir
    den = float(np.vdot(a,a).real)
    if den < 1e-12:
        return {"location_pu": None, "residual": None}
    m = float(np.vdot(a,b).real/den)
    residual = float(np.linalg.norm(b-m*a)/(np.linalg.norm(b)+1e-12))
    return {"location_pu":m, "residual":residual}


def single_ended_location(phasors: dict, fault_type: str) -> float | None:
    """Conventional reactance-loop baseline; explicitly affected by remote infeed/Rf."""
    v = np.array([phasors[k] for k in CHANNELS_S[:3]])
    i = np.array([phasors[k] for k in CHANNELS_S[3:]])
    if len(fault_type) == 2 and fault_type.endswith("G"):
        p = "ABC".index(fault_type[0])
        i0 = i.sum()/3
        # k0=(Z0-Z1)/Z1=2 when I0 is defined as one third of the residual.
        loop_v,loop_i = v[p],i[p]+2*i0
    else:
        a,b = ["ABC".index(p) for p in fault_type if p in "ABC"][:2]
        loop_v,loop_i = v[a]-v[b],i[a]-i[b]
    if abs(loop_i)<1e-10:
        return None
    return float((loop_v/loop_i).imag/.072)

