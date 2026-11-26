"""Reproducible training, validation-only temperature fitting, and held-out evaluation."""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import scipy
import sklearn
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, recall_score

from .anfis import fit_anfis, predict_anfis
from .config import CHANNELS, EVENT_THRESHOLD, FAULT_TYPES, LENGTH_KM, SEED, benchmark_metadata
from .dataset import generate, load, row_phasors
from .features import PROFILES, double_ended_location, single_ended_location
from .inference import predict_record
from .portable import export_forest, probabilities
from .simulator import to_record


def _temperature(model: dict, raw: np.ndarray, labels: np.ndarray) -> None:
    logits=np.log(np.maximum(raw,1e-12)) if model["kind"]=="forest" else raw
    classes=model["classes"]
    targets=np.array([classes.index(str(t)) for t in labels])
    def objective(log_t):
        p=softmax(logits/np.exp(log_t),axis=1)
        return float(-np.log(np.maximum(p[np.arange(len(p)),targets],1e-12)).mean())
    result=minimize_scalar(objective,bounds=(np.log(.25),np.log(4)),method="bounded")
    model["temperature"]=float(np.exp(result.x))


def _interval(errors: list[float], coverage: float = .90) -> float:
    if not errors:
        raise RuntimeError("No eligible calibration faults for a location interval")
    rank=min(len(errors),math.ceil((len(errors)+1)*coverage))
    return float(np.sort(errors)[rank-1])


def _ece(p: np.ndarray, truth: np.ndarray, classes: list[str]) -> float:
    confidence=p.max(axis=1)
    predicted=np.array(classes)[p.argmax(axis=1)]
    total=0.0
    bins=np.minimum((confidence*10).astype(int),9)
    for index in range(10):
        mask=bins==index
        if mask.any():
            total+=float(mask.mean()*abs((predicted[mask]==truth[mask]).mean()-confidence[mask].mean()))
    return total


def _location_metrics(truth: list, pred: list) -> dict:
    a,b=np.array(truth,float),np.array(pred,float)
    if len(a)==0:return {"n":0}
    errors=abs(a-b)
    return {"n":len(a),"mae_km":float(errors.mean()),"rmse_km":float(np.sqrt(np.mean((a-b)**2))),
            "p95_absolute_error_km":float(np.quantile(errors,.95)),
            "mean_error_percent_line":float(errors.mean()/LENGTH_KM*100)}


def _classification_metrics(truth, probabilities_, classes) -> dict:
    pred=np.array(classes)[np.argmax(probabilities_,axis=1)]
    return {"n":len(truth),"accuracy":float(accuracy_score(truth,pred)),
            "macro_f1":float(f1_score(truth,pred,labels=classes,average="macro",zero_division=0)),
            "recall":dict(zip(classes,map(float,recall_score(truth,pred,labels=classes,average=None,zero_division=0)))),
            "confusion_matrix":confusion_matrix(truth,pred,labels=classes).tolist(),"classes":classes,
            "expected_calibration_error_10_bins":_ece(probabilities_,np.array(truth),classes)}


def _evaluate(bundle: dict, family: str, rows: list[dict], models_root: Path) -> tuple[dict,list[dict]]:
    models=bundle["families"][family]
    x=np.array([[r[f] for f in bundle["features"]] for r in rows])
    event_p=probabilities(models["event"],x)
    is_internal=np.array([r["event"]=="internal" for r in rows])
    type_p=probabilities(models["type"],x[is_internal])
    metrics={"event":_classification_metrics([r["event"] for r in rows],event_p,models["event"]["classes"]),
             "fault_type_oracle_internal":_classification_metrics([r["fault_type"] for r in rows if r["event"]=="internal"],type_p,models["type"]["classes"])}
    predictions=[];actual_d=[];estimated_d=[];covered=[];false_locations=0;noise_buckets={};correct_types=0
    for row in rows:
        record=to_record(row_phasors(row))
        result=predict_record(record,models_root,bundle["profile"],family,bundle=bundle)
        predictions.append(result)
        if row["event"]=="internal":
            if result.get("fault_type")==row["fault_type"]:correct_types+=1
            if result["location_km"] is not None:
                truth=row["location_pu"]*LENGTH_KM
                actual_d.append(truth);estimated_d.append(result["location_km"])
                lo,hi=result["interval_km"]
                covered.append(lo<=truth<=hi)
                noise_buckets.setdefault(str(row["snr_db"]),[]).append(abs(truth-result["location_km"]))
        elif result["location_km"] is not None:
            false_locations+=1
    internal_count=int(is_internal.sum())
    nofault_count=len(rows)-internal_count
    accepted=sum(r["event"] is not None for r in predictions)
    healthy=[p for p,r in zip(predictions,rows) if r["event"]=="healthy"]
    internal=[p for p,r in zip(predictions,rows) if r["event"]=="internal"]
    metrics["pipeline"]={
        "total_records":len(rows),"event_acceptance_rate":accepted/len(rows),
        "correct_event_among_accepted":sum(p["event"]==r["event"] for p,r in zip(predictions,rows) if p["event"] is not None)/max(accepted,1),
        "internal_event_recall_including_abstentions":sum(p["event"]=="internal" for p in internal)/max(internal_count,1),
        "healthy_false_internal_rate":sum(p["event"]=="internal" for p in healthy)/max(len(healthy),1),
        "correct_fault_type_fraction_all_internal":correct_types/max(internal_count,1),
        "location_availability_on_internal":len(actual_d)/max(internal_count,1),
        "false_in_line_location_count":false_locations,"non_internal_records":nofault_count,
        "abstention_or_domain_rejection_count":len(rows)-accepted,
        "interval_target_coverage":.90,"interval_empirical_test_coverage":float(np.mean(covered)) if covered else None,
    }
    metrics["location_accepted_internal"]=_location_metrics(actual_d,estimated_d)
    metrics["noise_location"]={k:{"n":len(v),"mae_km":float(np.mean(v))} for k,v in noise_buckets.items()}
    return metrics,predictions


def train(root: Path, groups: int = 800, rules: int = 100, regenerate: bool = True) -> dict:
    started=time.perf_counter()
    if regenerate or not (root/"data"/"dataset.csv.gz").exists():
        generate(root,groups)
    rows=load(root)
    split={name:[r for r in rows if r["split"]==name] for name in ["train","validation","calibration","test"]}
    group_sets={s:{r["group_id"] for r in rr} for s,rr in split.items()}
    for a in group_sets:
        for b in group_sets:
            if a!=b:assert not group_sets[a]&group_sets[b]
    assets=root/"site"/"assets"
    assets.mkdir(parents=True,exist_ok=True)
    all_metrics={};fixtures=[];bundles={}
    for profile_index,(profile,features) in enumerate(PROFILES.items()):
        print("Training",profile,flush=True)
        x={s:np.array([[r[f] for f in features] for r in rr]) for s,rr in split.items()}
        internal={s:np.array([r["event"]=="internal" for r in rr]) for s,rr in split.items()}
        event_y={s:np.array([r["event"] for r in rr]) for s,rr in split.items()}
        type_y={s:np.array([r["fault_type"] for r in rr if r["event"]=="internal"]) for s,rr in split.items()}
        d_y={s:np.array([r["location_pu"] for r in rr if r["event"]=="internal"]) for s,rr in split.items()}
        options=dict(n_estimators=40,max_depth=13,min_samples_leaf=3,n_jobs=-1,random_state=SEED+profile_index)
        event_est=RandomForestClassifier(**options,class_weight="balanced_subsample").fit(x["train"],event_y["train"])
        type_est=RandomForestClassifier(**options,class_weight="balanced_subsample").fit(x["train"][internal["train"]],type_y["train"])
        event_model=export_forest(event_est,True);type_model=export_forest(type_est,True)
        _temperature(event_model,event_est.predict_proba(x["validation"]),event_y["validation"])
        _temperature(type_model,type_est.predict_proba(x["validation"][internal["validation"]]),type_y["validation"])
        if profile=="phasor_double":
            location_model={"kind":"physics","max_residual":.35}
        else:
            loc_est=RandomForestRegressor(**options).fit(x["train"][internal["train"]],d_y["train"])
            location_model=export_forest(loc_est,False)
        forests={"event":event_model,"type":type_model,"location":location_model}
        families={"forest":forests}
        if profile=="rms_single":
            am={}
            for target,labels,xx in [
                ("event",event_y["train"],x["train"]),
                ("type",type_y["train"],x["train"][internal["train"]]),
            ]:
                classes=list(map(str,np.unique(labels)))
                yy=np.eye(len(classes))[[classes.index(str(v)) for v in labels]]
                print(f"  ANFIS {target}: {rules} rules; hybrid fitting",flush=True)
                m=fit_anfis(xx,yy,rules=rules,epochs=2,seed=SEED)
                m["classes"]=classes
                vx=x["validation"] if target=="event" else x["validation"][internal["validation"]]
                vy=event_y["validation"] if target=="event" else type_y["validation"]
                _temperature(m,predict_anfis(m,vx),vy)
                am[target]=m
            print("  ANFIS distance: hybrid fitting",flush=True)
            am["location"]=fit_anfis(x["train"][internal["train"]],d_y["train"],rules=rules,epochs=2,seed=SEED)
            families["anfis"]=am
        low=x["train"].min(axis=0);high=x["train"].max(axis=0)
        margin=np.maximum((high-low)*.15,1e-5)
        bundle={"version":"0.1.0","profile":profile,"features":features,
                "bounds":np.column_stack([low-margin,high+margin]).tolist(),"benchmark":benchmark_metadata(),"families":families}
        # Intervals use ONLY the calibration partition, after the fixed eligibility gates.
        for family,model in families.items():
            model["location"]["interval_radius_km"]=0.0
            errors=[]
            for row in split["calibration"]:
                if row["event"]!="internal":continue
                pred=predict_record(to_record(row_phasors(row)),assets,profile,family,bundle=bundle)
                if pred["location_km"] is not None:
                    errors.append(abs(pred["location_km"]-row["location_pu"]*LENGTH_KM))
            model["location"]["interval_radius_km"]=_interval(errors)
            model["location"]["calibration_accepted_internal_records"]=len(errors)
        (assets/f"model-{profile}.json").write_text(json.dumps(bundle,separators=(",",":"),allow_nan=False),encoding="utf-8")
        bundles[profile]=bundle
        all_metrics[profile]={}
        for family in families:
            metrics,predictions=_evaluate(bundle,family,split["test"],assets)
            all_metrics[profile][family]=metrics
            for index in np.linspace(0,len(split["test"])-1,min(24,len(split["test"])),dtype=int):
                fixtures.append({"profile":profile,"family":family,"record":to_record(row_phasors(split["test"][index])),"expected":predictions[index]})
            print(profile,family,json.dumps(metrics["location_accepted_internal"]),flush=True)
    # Physics baselines use true class only for the explicitly labelled single-ended oracle comparator.
    truth=[];single=[];double=[]
    for row in split["test"]:
        if row["event"]!="internal":continue
        p=row_phasors(row)
        sm=single_ended_location(p,row["fault_type"])
        dm=double_ended_location(p)["location_pu"]
        if sm is not None and dm is not None:
            truth.append(row["location_pu"]*120);single.append(sm*120);double.append(dm*120)
    manifest=json.loads((root/"data"/"manifest.json").read_text())
    report={"version":"0.1.0","generated_at":datetime.now(timezone.utc).isoformat(),
            "environment":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__},
            "dataset":manifest,"profiles":all_metrics,
            "physics_baselines_oracle_internal":{"single_ended_reactance_oracle_type":_location_metrics(truth,single),
                                                  "double_ended_ungated":_location_metrics(truth,double)},
            "training_elapsed_seconds":time.perf_counter()-started,
            "limitations":["Synthetic fundamental-frequency benchmark only; no field or EMT validation claimed.",
                           "Calibration intervals target 90%; empirical held-out coverage is reported, not guaranteed under shift.",
                           "Only the configured line, bases, source/load ranges, and input conventions are supported.",
                           "Noise levels are complex-phasor-domain SNR, not raw waveform SNR.",
                           "Inference latency excludes acquisition of any raw measurement window."]}
    (assets/"evaluation.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    (root/"tests"/"fixtures").mkdir(parents=True,exist_ok=True)
    (root/"tests"/"fixtures"/"parity.json").write_text(json.dumps(fixtures,allow_nan=False),encoding="utf-8")
    examples=json.loads((root/"data"/"examples.json").read_text())
    (assets/"examples.json").write_text(json.dumps(examples,indent=2),encoding="utf-8")
    (root/"results").mkdir(exist_ok=True)
    (root/"results"/"evaluation.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print("Complete. Elapsed seconds:",round(time.perf_counter()-started,1),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument("--groups",type=int,default=800)
    parser.add_argument("--rules",type=int,default=100)
    parser.add_argument("--reuse-data",action="store_true")
    args=parser.parse_args()
    train(args.root,args.groups,args.rules,not args.reuse_data)


if __name__=="__main__":main()

