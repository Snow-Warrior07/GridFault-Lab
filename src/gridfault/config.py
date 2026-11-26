"""Benchmark assumptions are explicit; case9 supplies positive-sequence data only."""
from __future__ import annotations

import numpy as np

SEED = 20261126
BENCHMARK_ID = "ieee9-adapted-7-8-v1"
LINE_ID = "L7-8"
LENGTH_KM = 120.0  # Research assumption; not a length supplied by MATPOWER.
BASE_MVA = 100.0
BASE_KV = 345.0
Z_BASE_OHM = BASE_KV**2 / BASE_MVA
FAULT_TYPES = ["AG", "BG", "CG", "AB", "BC", "CA", "ABG", "BCG", "CAG", "ABC"]
EVENTS = ["external", "healthy", "internal"]
CHANNELS_S = ["s_va", "s_vb", "s_vc", "s_ia", "s_ib", "s_ic"]
CHANNELS_R = ["r_va", "r_vb", "r_vc", "r_ia", "r_ib", "r_ic"]
CHANNELS = CHANNELS_S + CHANNELS_R
EVENT_THRESHOLD = 0.65
TYPE_THRESHOLD = 0.55

# MATPOWER case9 numbering -> the numbering in the supplied review's Figure 4.
BUS_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 9, 7: 8, 8: 7, 9: 5}
# f, t, positive-sequence resistance/reactance, total shunt susceptance, per unit.
CASE9_BRANCHES = [
    (1, 4, 0.0, .0576, 0.0), (4, 5, .017, .092, .158),
    (5, 6, .039, .17, .358), (3, 6, 0.0, .0586, 0.0),
    (6, 7, .0119, .1008, .209), (7, 8, .0085, .072, .149),
    (8, 2, 0.0, .0625, 0.0), (8, 9, .032, .161, .306),
    (9, 4, .01, .085, .176),
]
BRANCHES = [(BUS_MAP[f], BUS_MAP[t], r, x, b) for f, t, r, x, b in CASE9_BRANCHES]
LOADS = [(BUS_MAP[5], .90, .30), (BUS_MAP[7], 1.00, .35), (BUS_MAP[9], 1.25, .50)]
A120 = np.exp(2j * np.pi / 3)
SEQUENCE_TO_PHASE = np.array([[1, 1, 1], [1, A120**2, A120], [1, A120, A120**2]], complex)
PHASE_TO_SEQUENCE = np.linalg.inv(SEQUENCE_TO_PHASE)


def phase_impedance(r: float, x: float, zero_ratio: float = 3.0) -> np.ndarray:
    z1 = complex(r, x)
    return SEQUENCE_TO_PHASE @ np.diag([zero_ratio * z1, z1, z1]) @ PHASE_TO_SEQUENCE


NOMINAL_Z = phase_impedance(.0085, .072)


def benchmark_metadata() -> dict:
    return {
        "id": BENCHMARK_ID, "line_id": LINE_ID, "bus_s": 7, "bus_r": 8,
        "line_length_km": LENGTH_KM, "base_mva": BASE_MVA, "base_kv": BASE_KV,
        "frequency_hz": 50, "z_base_ohm": Z_BASE_OHM,
        "nominal_z_re": NOMINAL_Z.real.tolist(), "nominal_z_im": NOMINAL_Z.imag.tolist(),
        "source": "https://github.com/MATPOWER/matpower/blob/master/data/case9.m",
        "assumptions": [
            "Linear, fundamental-frequency, three-phase nodal circuit; not EMT or field data.",
            "Loads are constant admittances; generators are grounded Norton equivalents.",
            "Negative-sequence impedance equals positive; zero-sequence impedance is assumed 3x positive.",
            "The monitored line is a series-only model; its case9 charging is intentionally omitted.",
            "Other branches use pi shunts; no detailed transformer winding/grounding model.",
            "120 km is an assumed length; it does not follow from MATPOWER data.",
            "Only the registered 7-8 circuit, bases, and stated operating ranges are supported.",
        ],
    }

