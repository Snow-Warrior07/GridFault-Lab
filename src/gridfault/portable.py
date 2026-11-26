"""JSON-only trained models. No uploaded pickle/joblib files are deserialized."""
from __future__ import annotations
import numpy as np
from .anfis import predict_anfis


def export_forest(estimator, classification: bool) -> dict:
    trees=[]
    for e in estimator.estimators_:
        t=e.tree_
        nodes=[]
        for i in range(t.node_count):
            if t.children_left[i]==-1:
                value=t.value[i].ravel().astype(float)
                if classification:
                    value=value/max(value.sum(),1e-12)
                    nodes.append([-1,0,-1,-1,value.tolist()])
                else:
                    nodes.append([-1,0,-1,-1,float(value[0])])
            else:
                nodes.append([int(t.feature[i]),float(t.threshold[i]),int(t.children_left[i]),int(t.children_right[i]),None])
        trees.append(nodes)
    return {"kind":"forest","classification":classification,"trees":trees,
            "classes":list(map(str,estimator.classes_)) if classification else None}


def raw_predict(model: dict, x: np.ndarray) -> np.ndarray:
    x=np.atleast_2d(x).astype(float)
    if model["kind"]=="anfis":
        return predict_anfis(model,x)
    # sklearn tree comparisons cast feature vectors to float32, also reproduced in JS.
    xf=x.astype(np.float32)
    outputs=[]
    for values in xf:
        acc=None
        for tree in model["trees"]:
            n=0
            while tree[n][0]!=-1:
                node=tree[n]
                n=node[2] if values[node[0]]<=node[1] else node[3]
            leaf=np.atleast_1d(tree[n][4]).astype(float)
            acc=leaf.copy() if acc is None else acc+leaf
        outputs.append(acc/len(model["trees"]))
    return np.array(outputs)


def probabilities(model: dict, x: np.ndarray) -> np.ndarray:
    scores=raw_predict(model,x)
    if model["kind"]=="forest":
        scores=np.log(np.maximum(scores,1e-12))
    scores=scores/model.get("temperature",1.0)
    scores-=scores.max(axis=1,keepdims=True)
    p=np.exp(scores)
    return p/p.sum(axis=1,keepdims=True)

