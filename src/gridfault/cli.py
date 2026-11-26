from __future__ import annotations
import argparse
import json
from pathlib import Path
from .inference import InputError, predict_record
from .simulator import simulate,to_record


def main():
    p=argparse.ArgumentParser(description="GridFault Lab research prototype")
    sub=p.add_subparsers(dest="command",required=True)
    sample=sub.add_parser("sample",help="Write a physically simulated example input")
    sample.add_argument("--fault",default="AG")
    sample.add_argument("--distance",type=float,default=.4,help="Per-unit distance from bus 7")
    sample.add_argument("--output",type=Path,default=Path("sample.json"))
    pred=sub.add_parser("predict")
    pred.add_argument("input",type=Path)
    pred.add_argument("--models",type=Path,default=Path("site/assets"))
    pred.add_argument("--profile",default="auto")
    pred.add_argument("--family",default="forest")
    a=p.parse_args()
    try:
        if a.command=="sample":
            a.output.write_text(json.dumps(to_record(simulate(fault_type=a.fault,location_pu=a.distance)["phasors"]),indent=2),encoding="utf-8")
            print(str(a.output))
        else:
            result=predict_record(json.loads(a.input.read_text(encoding="utf-8")),a.models,a.profile,a.family)
            print(json.dumps(result,indent=2,allow_nan=False))
    except (InputError,ValueError,OSError) as e:
        p.exit(2,f"Input/model error: {e}\n")


if __name__=="__main__":main()

