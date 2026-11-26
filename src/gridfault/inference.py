"""Strict input validation, measurement-aware routing, uncertainty and abstention."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from .config import BENCHMARK_ID, CHANNELS, LENGTH_KM, LINE_ID, EVENT_THRESHOLD, TYPE_THRESHOLD
from .features import PROFILES, double_ended_location, extract
from .portable import probabilities, raw_predict


class InputError(ValueError):
    pass


def validate_record(record: dict) -> tuple[dict,dict,list[str]]:
    if not isinstance(record,dict):
        raise InputError("Input must be a JSON object")
    required={"schema_version":1,"benchmark_id":BENCHMARK_ID,"line_id":LINE_ID,"bus_s":7,"bus_r":8,
              "base_kv":345,"base_mva":100,"frequency_hz":50,"line_length_km":120,"units":"per_unit_rms"}
    for k,value in required.items():
        if record.get(k)!=value or isinstance(record.get(k),bool):
            raise InputError(f"Unsupported or missing {k}; this model requires {value!r}")
    if not isinstance(record.get("synchronized"),bool):
        raise InputError("synchronized must explicitly be true or false")
    rms,phasors={},{}
    supplied=record.get("phasors",{})
    given_rms=record.get("rms",{})
    if not isinstance(supplied,dict) or not isinstance(given_rms,dict):
        raise InputError("rms and phasors must be objects")
    for k,value in supplied.items():
        if k not in CHANNELS:
            raise InputError(f"Unknown measurement channel: {k}")
        if not isinstance(value,dict) or set(value)!={"re","im"}:
            raise InputError(f"{k} requires re and im")
        try:
            if any(isinstance(value[t],bool) or not isinstance(value[t],(int,float)) for t in ["re","im"]):
                raise ValueError
            z=complex(float(value["re"]),float(value["im"]))
        except (TypeError,ValueError,OverflowError):
            raise InputError(f"Non-numeric phasor {k}") from None
        if not np.isfinite([z.real,z.imag]).all():
            raise InputError(f"Non-finite phasor {k}")
        phasors[k]=z;rms[k]=float(abs(z))
    for k,value in given_rms.items():
        if k not in CHANNELS:
            raise InputError(f"Unknown measurement channel: {k}")
        try:
            if isinstance(value,bool) or not isinstance(value,(int,float)):raise ValueError
            v=float(value)
        except (TypeError,ValueError,OverflowError):
            raise InputError(f"Non-numeric RMS {k}") from None
        if not np.isfinite(v) or v<0:
            raise InputError(f"{k} must be a finite non-negative RMS magnitude")
        if k in rms and abs(v-rms[k])>max(.001,.02*rms[k]):
            raise InputError(f"RMS and phasor disagree for {k}")
        rms[k]=v
    f=extract(phasors);f.update(rms)
    eligible=[p for p,cols in PROFILES.items() if all(k in f for k in cols) and (p!="phasor_double" or record["synchronized"])]
    return f,phasors,eligible


def choose_profile(record: dict, requested: str = "auto") -> tuple[str,dict,dict,list[str]]:
    f,phasors,eligible=validate_record(record)
    if requested=="auto":
        if not eligible:
            missing=[k for k in PROFILES["rms_single"] if k not in f]
            raise InputError("No trained profile available. Missing sending-end channels: "+", ".join(missing))
        profile=eligible[-1]
    else:
        if requested not in PROFILES:
            raise InputError("Unknown measurement profile")
        if requested not in eligible:
            missing=[k for k in PROFILES[requested] if k not in f]
            detail="unsynchronized terminal data" if requested=="phasor_double" and not record["synchronized"] else ", ".join(missing)
            raise InputError("Requested profile unavailable: "+detail)
        profile=requested
    return profile,f,phasors,eligible


def predict_record(record: dict, models_root: Path, profile: str = "auto", family: str = "forest", *, bundle: dict | None = None) -> dict:
    profile,f,phasors,eligible=choose_profile(record,profile)
    bundle=bundle or json.loads((models_root/f"model-{profile}.json").read_text(encoding="utf-8"))
    if family not in bundle["families"]:
        raise InputError(f"{family} is not trained for {profile}")
    model=bundle["families"][family]
    x=np.array([[f[k] for k in bundle["features"]]])
    bounds=np.array(bundle["bounds"])
    outside=[name for name,value,limits in zip(bundle["features"],x[0],bounds) if value<limits[0] or value>limits[1]]
    result={"profile":profile,"family":family,"eligible_profiles":eligible,"event":None,"fault_type":None,
            "location_km":None,"interval_km":None,"quality":"accepted","messages":[],"model_version":bundle["version"]}
    if outside:
        result.update(quality="out_of_domain",messages=["Outside training support: "+", ".join(outside[:6])])
        return result
    ep=probabilities(model["event"],x)[0]
    idx=int(np.argmax(ep));event=model["event"]["classes"][idx]
    result["event_probabilities"]=dict(zip(model["event"]["classes"],map(float,ep)))
    if ep[idx]<EVENT_THRESHOLD:
        result.update(quality="uncertain",messages=["Event confidence below the validated decision threshold; diagnosis withheld."])
        return result
    result["event"]=event
    if event!="internal":
        result["messages"]=["No in-line fault location applies to this event."]
        return result
    tp=probabilities(model["type"],x)[0]
    ti=int(np.argmax(tp))
    result["type_probabilities"]=dict(zip(model["type"]["classes"],map(float,tp)))
    if tp[ti]>=TYPE_THRESHOLD:
        result["fault_type"]=model["type"]["classes"][ti]
    else:
        result["messages"].append("Fault type is uncertain; type label withheld.")
    if model["location"]["kind"]=="physics":
        physical=double_ended_location(phasors)
        m=physical["location_pu"]
        result["physics_residual"]=physical["residual"]
        if m is None or physical["residual"]>model["location"]["max_residual"]:
            result["messages"].append("Electrical consistency check failed; location withheld.")
            return result
    else:
        m=float(raw_predict(model["location"],x)[0,0])
    if not np.isfinite(m) or not 0<=m<=1:
        result["messages"].append("Predicted position is outside the monitored line; location withheld.")
        return result
    d=m*LENGTH_KM
    q=model["location"]["interval_radius_km"]
    result.update(location_km=d,interval_km=[max(0,d-q),min(LENGTH_KM,d+q)],location_method=model["location"]["kind"])
    if not record["synchronized"]:
        result["messages"].append("Phasor-based location is disabled; using the trained RMS fallback.")
    return result

