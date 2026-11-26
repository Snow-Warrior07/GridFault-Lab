"""Generate independent operating-condition groups before measurements/augmentation."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np

from .config import CHANNELS, FAULT_TYPES, SEED, benchmark_metadata
from .features import PHASOR_FEATURES, extract
from .simulator import draw_conditions, perturb_phasors, simulate, to_record


def generate(root: Path, groups: int = 800, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    order = rng.permutation(groups)
    cuts = [int(.60*groups),int(.75*groups),int(.85*groups)]
    partitions = dict(zip(order,np.repeat(["train","validation","calibration","test"],np.diff([0,*cuts,groups]))))
    rows,examples=[],[]
    for group in range(groups):
        split = str(partitions[group])
        conditions = draw_conditions(rng)
        d=float(rng.uniform(.05,.95))
        # Deliberate held-out distance bands: stronger than merely using new random floats.
        if split != "test":
            while .20<=d<=.25 or .70<=d<=.75:
                d=float(rng.uniform(.05,.95))
        elif group%2==0:
            d=float(rng.uniform(.20,.25) if group%4==0 else rng.uniform(.70,.75))
        rf=float(np.exp(rng.uniform(np.log(.5),np.log(80))))
        events=[("healthy","AG",9)]+[("internal",f,9) for f in FAULT_TYPES]+[
            ("external",str(rng.choice(FAULT_TYPES)),9),
            ("external",str(rng.choice(FAULT_TYPES)),6),
        ]
        for index,(event,fault,bus) in enumerate(events):
            physical=simulate(conditions,fault,d,rf,event,bus)
            scenario_id=f"G{group:04d}-E{index:02d}"
            for variant,snr in enumerate([None,float(rng.choice([20,30,40,50]))]):
                measured=perturb_phasors(physical["phasors"],rng,snr)
                row={
                    "group_id":f"G{group:04d}","scenario_id":scenario_id,"split":split,
                    "variant":variant,"event":event,"fault_type":physical["fault_type"] or "",
                    "actual_fault_type":physical["actual_fault_type"] or "",
                    "location_pu":physical["location_pu"] if event=="internal" else "",
                    "resistance_ohm":physical["resistance_ohm"] or "",
                    "snr_db":"clean" if snr is None else snr,
                    "external_bus":physical["external_bus"] or "",
                    **physical["conditions"],**extract(measured),
                }
                rows.append(row)
                if split=="test" and variant==1 and not any(e["label"]==event+":"+(fault if event=="internal" else "") for e in examples):
                    examples.append({"id":scenario_id,"label":event+":"+(fault if event=="internal" else ""),
                                     "truth":{"event":event,"fault_type":row["fault_type"] or None,
                                              "location_km":d*120 if event=="internal" else None},
                                     "record":to_record(measured)})
        if (group+1)%100==0:
            print(f"Generated {group+1}/{groups} independent groups ({len(rows)} measurement records)",flush=True)
    (root/"data").mkdir(parents=True,exist_ok=True)
    dest=root/"data"/"dataset.csv.gz"
    # Deterministic gzip header makes the same seed/environment produce the same digest.
    import io
    buffer=io.StringIO(newline="")
    writer=csv.DictWriter(buffer,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows)
    dest.write_bytes(gzip.compress(buffer.getvalue().encode(),mtime=0))
    manifest={
        "seed":seed,"groups":groups,"records":len(rows),"physical_scenarios":groups*13,
        "partition_groups":dict(Counter(partitions.values())),
        "partition_records":dict(Counter(r["split"] for r in rows)),
        "event_counts":dict(Counter(r["event"] for r in rows)),
        "fault_types":FAULT_TYPES,"variants":["clean","20/30/40/50 dB complex-phasor noise"],
        "sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),"benchmark":benchmark_metadata(),
        "held_out_distance_bands_pu":[[.20,.25],[.70,.75]],
        "splitting":"Operating-condition groups assigned before events, windows, or noisy copies are generated.",
        "features":PHASOR_FEATURES,
        "not_features":["event","fault_type","actual_fault_type","location_pu","resistance_ohm","snr_db","conditions","scenario_id"],
    }
    (root/"data"/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    (root/"data"/"examples.json").write_text(json.dumps(examples,indent=2),encoding="utf-8")
    return manifest


def load(root: Path) -> list[dict]:
    with gzip.open(root/"data"/"dataset.csv.gz","rt",encoding="utf-8",newline="") as f:
        rows=list(csv.DictReader(f))
    for row in rows:
        for k in PHASOR_FEATURES:
            row[k]=float(row[k])
        row["location_pu"]=float(row["location_pu"]) if row["location_pu"] else None
    return rows


def row_phasors(row: dict) -> dict:
    return {k:complex(row[k+"_re"],row[k+"_im"]) for k in CHANNELS}

