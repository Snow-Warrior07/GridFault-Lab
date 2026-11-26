"""First-order Gaussian Sugeno ANFIS with hybrid least-squares/premise learning.

K-means initializes premises. Each hybrid epoch fits ridge consequents, then updates
Gaussian centers/log-widths using the exact MSE gradient. Not a MATLAB bit-for-bit clone.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits


def _weights(x: np.ndarray, centers: np.ndarray, widths: np.ndarray) -> np.ndarray:
    logw = -.5*np.sum(((x[:,None,:]-centers[None,:,:])/widths[None,:,:])**2,axis=2)
    logw -= logw.max(axis=1,keepdims=True)
    w = np.exp(logw)
    return w/w.sum(axis=1,keepdims=True)


def predict_anfis(model: dict, x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    xn = (x-np.array(model["mean"]))/np.array(model["scale"])
    w = _weights(xn,np.array(model["centers"]),np.array(model["widths"]))
    extended = np.column_stack([np.ones(len(xn)),xn])
    coeff = np.array(model["coefficients"])
    return np.einsum("nk,nj,kjo->no",w,extended,coeff)


def fit_anfis(x: np.ndarray, y: np.ndarray, rules: int = 100, epochs: int = 3, seed: int = 0) -> dict:
    x,y = np.asarray(x,float),np.asarray(y,float)
    if y.ndim==1:
        y=y[:,None]
    mean,scale = x.mean(axis=0),np.maximum(x.std(axis=0),1e-6)
    xn=(x-mean)/scale
    rules=min(rules,len(xn))
    with threadpool_limits(limits=1):
        km=KMeans(n_clusters=rules,n_init=1,max_iter=70,random_state=seed).fit(xn)
        centers=km.cluster_centers_.copy()
        distances=np.sqrt(np.sum((centers[:,None,:]-centers[None,:,:])**2,axis=2))
        distances[distances==0]=np.inf
        nearest=np.min(distances,axis=1)
        nearest=np.where(np.isfinite(nearest),nearest,1.0)
        widths=np.repeat(np.clip(nearest/np.sqrt(x.shape[1]),.35,2)[:,None],x.shape[1],axis=1)
        extended=np.column_stack([np.ones(len(xn)),xn])
        losses=[]
        for epoch in range(epochs+1):
            w=_weights(xn,centers,widths)
            design=(w[:,:,None]*extended[:,None,:]).reshape(len(xn),-1)
            reg=Ridge(alpha=.03,fit_intercept=False,solver="cholesky").fit(design,y)
            coeff=np.asarray(reg.coef_).T.reshape(rules,x.shape[1]+1,y.shape[1])
            local=np.einsum("nj,kjo->nko",extended,coeff)
            pred=np.einsum("nk,nko->no",w,local)
            error=pred-y
            losses.append(float(np.mean(error**2)))
            if epoch==epochs:
                break
            dl=np.einsum("no,nko->nk",2*error/(len(xn)*y.shape[1]),local-pred[:,None,:])*w
            delta=xn[:,None,:]-centers[None,:,:]
            dc=np.einsum("nk,nkp->kp",dl,delta/(widths[None,:,:]**2))
            ds=np.einsum("nk,nkp->kp",dl,(delta/widths[None,:,:])**2)
            centers -= .03*np.clip(dc,-1,1)
            widths=np.clip(widths*np.exp(-.03*np.clip(ds,-1,1)),.2,3.0)
    return {"kind":"anfis","rules":rules,"epochs":epochs,"mean":mean.tolist(),
            "scale":scale.tolist(),"centers":centers.tolist(),"widths":widths.tolist(),
            "coefficients":coeff.tolist(),"training_mse_by_epoch":losses}

