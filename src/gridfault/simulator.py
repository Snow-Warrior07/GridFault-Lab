"""Phase-domain steady-state short-circuit simulation on an adapted 9-bus network.

Faults are electrical shunts/phase connections, not handcrafted class templates.
All voltages are phase-to-neutral RMS phasors; terminal currents point INTO the line.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

from .config import (
    BENCHMARK_ID, BRANCHES, CHANNELS, LENGTH_KM, LINE_ID, LOADS,
    SEQUENCE_TO_PHASE, Z_BASE_OHM, phase_impedance,
)


@dataclass
class Conditions:
    load_scale: float = 1.0
    source_scale: float = 1.0
    source_angle_deg: float = 5.0
    voltage_scale: float = 1.0
    line_r_scale: float = 1.0
    line_x_scale: float = 1.0
    zero_ratio: float = 3.0
    load_unbalance: float = 0.0


def draw_conditions(rng: np.random.Generator) -> Conditions:
    return Conditions(
        load_scale=float(rng.uniform(.65, 1.35)), source_scale=float(rng.uniform(.75, 1.25)),
        source_angle_deg=float(rng.uniform(2, 9)), voltage_scale=float(rng.uniform(.97, 1.05)),
        line_r_scale=float(rng.uniform(.95, 1.05)), line_x_scale=float(rng.uniform(.95, 1.05)),
        zero_ratio=3.0, load_unbalance=float(rng.uniform(-.03, .03)),
    )


def _stamp_branch(y: np.ndarray, f: int, t: int, z: np.ndarray, b: float = 0.0) -> None:
    fs, ts = slice(3*f, 3*f+3), slice(3*t, 3*t+3)
    ys = np.linalg.inv(z)
    sh = np.eye(3) * (1j * b / 2)
    y[fs, fs] += ys + sh
    y[ts, ts] += ys + sh
    y[fs, ts] -= ys
    y[ts, fs] -= ys


def _connect(y: np.ndarray, a: int, b: int, conductance: float) -> None:
    y[a, a] += conductance
    y[b, b] += conductance
    y[a, b] -= conductance
    y[b, a] -= conductance


def _stamp_fault(y: np.ndarray, bus: int, fault_type: str, resistance_ohm: float, aux: int) -> None:
    phases = ["ABC".index(p) for p in fault_type if p in "ABC"]
    resistance = max(resistance_ohm / Z_BASE_OHM, 1e-5)
    nodes = [3*bus+p for p in phases]
    if len(nodes) == 1:
        y[nodes[0], nodes[0]] += 1 / resistance
    elif len(nodes) == 2 and not fault_type.endswith("G"):
        _connect(y, nodes[0], nodes[1], 1 / resistance)
    else:
        # Common junction: DLG has phase arms and a shared ground return.
        arm_r = resistance / 2 if fault_type.endswith("G") else resistance
        for node in nodes:
            _connect(y, node, aux, 1 / arm_r)
        if fault_type.endswith("G"):
            y[aux, aux] += 2 / resistance


def simulate(
    conditions: Conditions | None = None, fault_type: str | None = "AG",
    location_pu: float = .4, resistance_ohm: float = 10.0,
    event: str = "internal", external_bus: int = 9,
) -> dict:
    c = conditions or Conditions()
    if event not in {"internal", "healthy", "external"}:
        raise ValueError("Unknown event")
    if not 0 < location_pu < 1:
        raise ValueError("An internal fault must be strictly inside the line")
    if fault_type and fault_type not in {"AG","BG","CG","AB","BC","CA","ABG","BCG","CAG","ABC"}:
        raise ValueError("Unsupported fault type")
    if resistance_ohm <= 0:
        raise ValueError("Fault resistance must be positive")
    active_fault = event != "healthy"
    needs_aux = active_fault and (fault_type.endswith("G") and len(fault_type)>2 or fault_type == "ABC")
    n = 30 + int(needs_aux)  # 9 buses + the line's inserted fault node, plus optional common junction.
    y = np.zeros((n, n), complex)
    inj = np.zeros(n, complex)
    z_line = phase_impedance(.0085*c.line_r_scale, .072*c.line_x_scale, c.zero_ratio)
    for f, t, r, x, b in BRANCHES:
        if {f, t} == {7, 8}:
            continue
        _stamp_branch(y, f-1, t-1, phase_impedance(r, x, c.zero_ratio), b)
    _stamp_branch(y, 6, 9, z_line * location_pu)
    _stamp_branch(y, 9, 7, z_line * (1-location_pu))
    for bus, p, q in LOADS:
        sl = slice(3*(bus-1), 3*bus)
        factors = np.array([1+c.load_unbalance, 1-c.load_unbalance, 1.0])
        y[sl, sl] += np.diag((p-1j*q) * c.load_scale * factors)
    for bus, angle in [(1, 0), (2, c.source_angle_deg), (3, .6*c.source_angle_deg)]:
        sl = slice(3*(bus-1), 3*bus)
        z_src = phase_impedance(.015*c.source_scale, .18*c.source_scale, 2.0)
        ys = np.linalg.inv(z_src)
        ep = SEQUENCE_TO_PHASE @ np.array([0, c.voltage_scale*np.exp(1j*np.deg2rad(angle)), 0])
        y[sl, sl] += ys
        inj[sl] += ys @ ep
    if active_fault:
        _stamp_fault(y, 9 if event == "internal" else external_bus-1, fault_type, resistance_ohm, n-1)
    v = np.linalg.solve(y, inj)
    vs, vr, vf = v[18:21], v[21:24], v[27:30]
    is_ = np.linalg.solve(z_line*location_pu, vs-vf)
    ir = np.linalg.solve(z_line*(1-location_pu), vr-vf)
    return {
        "phasors": dict(zip(CHANNELS, np.concatenate([vs,is_,vr,ir]))),
        "event": event, "fault_type": fault_type if event == "internal" else None,
        "actual_fault_type": fault_type if active_fault else None,
        "location_pu": location_pu if event == "internal" else None,
        "resistance_ohm": resistance_ohm if active_fault else None,
        "external_bus": external_bus if event == "external" else None,
        "conditions": asdict(c), "actual_z": z_line,
        "kcl_residual": float(np.linalg.norm(y@v-inj) / max(np.linalg.norm(inj), 1e-12)),
    }


def perturb_phasors(phasors: dict, rng: np.random.Generator, snr_db: float | None = 35) -> dict:
    """SNR is defined in the phasor domain, NOT claimed to be raw-waveform SNR."""
    out = {}
    for start in range(0,12,3):
        names = CHANNELS[start:start+3]
        values = np.array([phasors[k] for k in names])
        if snr_db is not None:
            rms = max(float(np.sqrt(np.mean(abs(values)**2))), 1e-8)
            sd = rms * 10**(-snr_db/20) / np.sqrt(2)
            values = values + sd*(rng.normal(size=3)+1j*rng.normal(size=3))
        out.update(zip(names, values))
    return out


def to_record(phasors: dict, *, synchronized: bool = True) -> dict:
    return {
        "schema_version": 1, "benchmark_id": BENCHMARK_ID, "line_id": LINE_ID,
        "bus_s": 7, "bus_r": 8, "line_length_km": LENGTH_KM,
        "base_mva": 100, "base_kv": 345, "frequency_hz": 50,
        "units": "per_unit_rms", "synchronized": synchronized,
        "phasors": {k:{"re":float(v.real),"im":float(v.imag)} for k,v in phasors.items()},
    }

