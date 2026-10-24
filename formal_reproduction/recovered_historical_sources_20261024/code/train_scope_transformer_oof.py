from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_unified_transformer_fusion import FusionModel
from train_scope_strict_oof_stack import read_frames, read_hits, fold_features
from build_strict_oof_dynamic_stack import load_edges
from feature_profiles import fit_profile_groups, transformer_modalities

ROOT = Path.cwd(); FEAT = Path(os.environ.get("PROTOESS_SCOPE_FEATURE_ROOT", ROOT / "eukaryote_model_datasets" / "scope_feature_sets_v2")); MAN = ROOT / "eukaryote_model_datasets" / "unified_v3_experiment_manifests_pre95human"; JOINT = MAN / "joint" / "all_15_species"
# Preserve the historical Transformer output tree unless a reproducibility run
# explicitly supplies an isolated destination.
OUT = Path(os.environ.get("TRANSFORMER_OOF_OUT", ROOT / "eukaryote_model_results" / "scope_transformer_oof"))


def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def split_groups(X, k):
    a = [X[:, 0:128], X[:, 128:256], X[:, 256:287], X[:, 287:415], X[:, 415:543], X[:, 543:558], X[:, 558:573], X[:, 573:582]]
    cur = np.concatenate([X[:, 582:591+k], X[:, 591+k+21:591+k+36]], axis=1)
    return a + [cur, X[:, 591+k:591+k+21]]


def train_predict(train_groups, ytrain, eval_groups, yeval, predict_groups, seed, artifact_path: Path | None = None):
    """Fit with early stopping selected solely within the training rows.

    ``eval_groups`` and ``yeval`` are prediction-only.  They are retained in
    this interface so all historical callers continue to work, but never
    participate in epoch selection or preprocessing estimation.
    """
    del yeval
    seed_all(seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(os.environ.get("TRANSFORMER_BATCH", "64")); max_epochs = int(os.environ.get("TRANSFORMER_MAX_EPOCHS", "80")); patience = int(os.environ.get("TRANSFORMER_PATIENCE", "12"))
    fit_idx, stop_idx = next(StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed + 1000).split(np.zeros(len(ytrain)), ytrain))

    def scale(train, *others):
        scaled_train=[]; scaled_others=[[] for _ in others]
        for position, tr in enumerate(train):
            mu=tr.mean(axis=0); sd=tr.std(axis=0).clip(min=1e-6); scaled_train.append(((tr-mu)/sd).astype(np.float32))
            for out, other in zip(scaled_others, others): out.append(((other[position]-mu)/sd).astype(np.float32))
        return scaled_train, scaled_others

    def build(groups):
        dims=[item.shape[1] for item in groups]; model=FusionModel(dims).to(device); pos=(len(ytrain)-ytrain.sum())/max(ytrain.sum(),1)
        return dims, model, nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos],device=device)), torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4), torch.amp.GradScaler("cuda",enabled=device.type=="cuda")

    def predict(model, groups, dims):
        out=[]; model.eval(); loader=DataLoader(TensorDataset(*[torch.from_numpy(x) for x in groups]),batch_size=max(256,batch_size*2))
        with torch.no_grad():
            for values in loader:
                xs=[item.to(device) for item in values]; out.append(torch.sigmoid(model(xs,[torch.ones(len(xs[0]),device=device) for _ in xs])).cpu().numpy())
        return np.concatenate(out)

    def train_epoch(model, loss_fn, optimizer, scaler, groups, targets, dims):
        loader=DataLoader(TensorDataset(*[torch.from_numpy(x) for x in groups],torch.from_numpy(targets.astype(np.float32))),batch_size=batch_size,shuffle=True); model.train()
        for values in loader:
            xs=[item.to(device) for item in values[:len(dims)]]; target=values[-1].to(device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=device.type=="cuda"):
                loss=loss_fn(model(xs,[torch.ones(len(target),device=device) for _ in xs]),target)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()

    selection_train, (selection_stop,) = scale([item[fit_idx] for item in train_groups], [item[stop_idx] for item in train_groups])
    dims, model, loss_fn, optimizer, scaler = build(selection_train); best_epoch=1; best_auc=-np.inf; wait=0
    for epoch in range(1, max_epochs + 1):
        train_epoch(model, loss_fn, optimizer, scaler, selection_train, ytrain[fit_idx], dims)
        auc=roc_auc_score(ytrain[stop_idx], predict(model, selection_stop, dims))
        if auc > best_auc: best_auc=auc; best_epoch=epoch; wait=0
        else: wait += 1
        if wait >= patience: break
    full_train, (scaled_eval, scaled_predict) = scale(train_groups, eval_groups, predict_groups)
    dims, model, loss_fn, optimizer, scaler = build(full_train)
    for _ in range(best_epoch): train_epoch(model, loss_fn, optimizer, scaler, full_train, ytrain, dims)
    if artifact_path is not None:
        artifact_path = Path(artifact_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "group_dims": dims,
            "scaler_mean": [group.mean(axis=0).astype(np.float32) for group in train_groups],
            "scaler_std": [group.std(axis=0).clip(min=1e-6).astype(np.float32) for group in train_groups],
            "best_epoch": int(best_epoch),
            "seed": int(seed),
        }, artifact_path)
    return predict(model, scaled_eval, dims), predict(model, scaled_predict, dims), best_epoch


def main():
    seed_all(); OUT.mkdir(parents=True,exist_ok=True); joint={s:pd.read_csv(JOINT/f"{s}_metadata.tsv",sep="\t",dtype={"record_id":str}) for s in ("train","val","test")}; only=os.environ.get("TRANSFORMER_ONLY",""); force=os.environ.get("TRANSFORMER_FORCE","0")=='1'; jobs=[]
    for scope in ("joint","clade","within_species","loso"):
        for p in sorted((FEAT/scope).iterdir()):
            if p.is_dir() and (p/"DONE.json").exists() and (not only or f"{scope}:{p.name}"==only): jobs.append((scope,p.name))
    active_species=set()
    for scope,name in jobs: active_species.update(read_frames(scope,name)["train"].species.astype(str).unique())
    edges=load_edges(joint,active_species)
    for scope,name in jobs:
        out=OUT/scope/name; out.mkdir(parents=True,exist_ok=True)
        if (out/"metrics.json").exists() and not force: continue
        frames=read_frames(scope,name); X={s:np.asarray(np.load(FEAT/scope/name/f"X_{s}.npy"),dtype=np.float32) for s in frames}; feature_names=json.loads((FEAT/scope/name/"feature_names.json").read_text(encoding="utf-8")); profile_path=os.environ.get("PROTOESS_FEATURE_PROFILE_PATH",""); profile=json.loads(Path(profile_path).read_text(encoding="utf-8")) if profile_path else None; y={s:frames[s].label.eq("essential").astype(np.int8).to_numpy() for s in frames}; ids=frames["train"].record_id.astype(str).to_numpy(); k=int(X["train"].shape[1]-582-9-21-15); hits=read_hits(scope,name,frames); oof=np.zeros(len(ids),dtype=np.float32)
        for fold,(ii,hh) in enumerate(StratifiedKFold(3,shuffle=True,random_state=42).split(ids,y["train"])):
            inner=set(ids[ii]); xi=fold_features(X["train"][ii],frames,hits[hits.scope_split.eq("train")],edges,inner,ids[ii],582+9+k); xh=fold_features(X["train"][hh],frames,hits[hits.scope_split.eq("train")],edges,inner,ids[hh],582+9+k); gi,gh,gn=fit_profile_groups(xi,xh,feature_names,profile,seed=42+fold); pg=train_predict(transformer_modalities(gi,gn),y["train"][ii],transformer_modalities(gh,gn),y["train"][hh],transformer_modalities(gh,gn),42+fold)[0]; oof[hh]=pg
        gtrain,geval,gn=fit_profile_groups(X["train"],X["val"],feature_names,profile,seed=42); train_modalities=transformer_modalities(gtrain,gn); val_modalities=transformer_modalities(geval,gn); gtrain_test,gtest,gn_test=fit_profile_groups(X["train"],X["test"],feature_names,profile,seed=42); test_modalities=transformer_modalities(gtest,gn_test)
        _,pv,_=train_predict(train_modalities,y["train"],val_modalities,y["val"],val_modalities,42); _,pt,_=train_predict(train_modalities,y["train"],val_modalities,y["val"],test_modalities,42)
        result={"scope":scope,"name":name,"device":"cuda" if torch.cuda.is_available() else "cpu","oof_AUROC":float(roc_auc_score(y["train"],oof)),"val_AUROC":float(roc_auc_score(y["val"],pv)),"test_AUROC":float(roc_auc_score(y["test"],pt))}; np.save(out/"oof_prob.npy",oof);np.save(out/"val_prob.npy",pv);np.save(out/"test_prob.npy",pt);(out/"metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8");print(scope,name,result,flush=True)


if __name__=='__main__': main()
