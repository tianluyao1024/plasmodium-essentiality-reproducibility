"""Legacy tree-only application helper.

This module remains importable because the annotation-exclusion and application
workflows reuse its deterministic application-feature functions.  Its former
``train_application`` routine assembles a four-branch tree-only score and is
not the manuscript's formal five-branch OOF model.  Formal whole-proteome
scoring is performed by ``run_ncomms_full_stack_application_baseline.py``;
the executable entry point below is deliberately disabled to avoid conflating
the legacy helper with the formal analysis.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from train_scope_strict_oof_stack import lgbm, cat, aggregate_mm

ROOT=Path.cwd(); FEAT=ROOT/'eukaryote_model_datasets/scope_feature_sets_v2'; MAN=ROOT/'eukaryote_model_datasets/unified_v3_experiment_manifests_pre95human'; ALIGN=ROOT/'eukaryote_model_datasets/features/unified_v3_aligned_features'; APP=ROOT/'eukaryote_model_datasets/genomewide_target_application/random_species_label_split_seed42'; APPF=ROOT/'eukaryote_model_datasets/features/genomewide_target_application'; RUN=Path(os.environ.get('FORMAL_APPLICATION_FEATURE_ROOT', ROOT/'eukaryote_model_datasets/genomewide_target_application/final_application_features')); OUT=ROOT/'eukaryote_model_results/final_target_prediction'; OUT.mkdir(parents=True,exist_ok=True)
ALIASES={'Plasmodium_berghei_ANKA':'Plasmodium_berghei','Plasmodium_falciparum_3D7':'Plasmodium_falciparum','Plasmodium_knowlesi_H':'Plasmodium_knowlesi'}
GROUPS=['esm2','prott5','basic_seq','go_ipr_pfam','kmer_graph','homology30','homology50','string_network']

def app_arrays():
    out={}
    for g in GROUPS:
        p=APPF/g
        f=next((x for x in p.glob('app_*.npy') if 'ids' not in x.name),None)
        if f is None: raise FileNotFoundError(p)
        out[g]=np.asarray(np.load(f,mmap_mode='r'),dtype=np.float32)
    out['kmer_graph']=out['kmer_graph'][:,:512]
    out['ppi_network']=np.asarray(np.load(APPF/'string_network/app_emb.npy',mmap_mode='r'),dtype=np.float32)
    out['ppi_prop']=np.asarray(np.load(APPF/'string_label_propagation/app_emb.npy',mmap_mode='r'),dtype=np.float32)
    return out

def aligned_group(g):
    return np.asarray(np.load(ALIGN/g/'train_emb_aligned.npy',mmap_mode='r'),dtype=np.float32), np.load(ALIGN/g/'train_present.npy',mmap_mode='r').astype(bool)

def app_static(scope,name,frames,aa):
    master=np.load(ALIGN/'master_train_ids.npy',allow_pickle=True).astype(str); pos={x:i for i,x in enumerate(master)}
    train_ids=frames['train'].record_id.astype(str).to_numpy(); idx=np.array([pos[x] for x in train_ids],dtype=np.int64)
    parts=[]
    for g in GROUPS:
        tr,mask=aligned_group(g); raw=tr[idx].copy(); valid=mask[idx]
        mean=np.where(valid[:,None],raw,0).sum(0)/np.maximum(valid.sum(0),1); centered=np.where(valid[:,None],raw-mean,0); std=np.sqrt((centered**2).sum(0)/np.maximum(valid.sum(0),1)).clip(1e-6)
        z=np.asarray(aa[g],dtype=np.float32).copy(); z=np.nan_to_num(z,nan=0,posinf=0,neginf=0); z=(z-mean)/std
        if raw.shape[1]>128:
            svd=TruncatedSVD(n_components=128,n_iter=4,random_state=42); svd.fit(np.nan_to_num(raw,nan=0)); z=svd.transform(z).astype(np.float32)
        parts.append(z)
    species=sorted(frames['train'].species.astype(str).unique()); spix={s:i for i,s in enumerate(species)}; one=np.zeros((len(aa['esm2']),len(species)),np.float32)
    app_species=APP_META.species.astype(str).map(lambda x: ALIASES.get(x,x)).to_numpy()
    for i,s in enumerate(app_species):
        if s in spix: one[i,spix[s]]=1
    parts.append(one); parts.append(aa['ppi_network'][:,:9]); return np.concatenate(parts,axis=1)

def app_ppi_dynamic(frames,aa):
    out=np.zeros((len(APP_META),15),np.float32); idx={rid:i for i,rid in enumerate(APP_META.record_id.astype(str))}
    tax={'Homo_sapiens':9606,'Plasmodium_berghei_ANKA':5823,'Plasmodium_falciparum_3D7':36329,'Plasmodium_knowlesi_H':5851}
    for sp,t in tax.items():
        z=np.load(RUN/f'ppi_{t}.npz',allow_pickle=True); ids=z['ids'].astype(str); net=z['network']; pr=z['propagation']
        for i,rid in enumerate(ids):
            j=idx[rid]
            for k in range(3): out[j,k*5:k*5+5]=[pr[i,k*4],pr[i,k*4+1],pr[i,k*4+2],pr[i,k*4+3],net[i,k]]
    return out

def legacy_train_tree_only_application(scope,name,aa,mm_hits,ppi_dyn):
    base=MAN/scope/name; frames={s:pd.read_csv(base/f'{s}_metadata.tsv',sep='\t',dtype={'record_id':str}).fillna('') for s in ('train','val','test')}; X=np.asarray(np.load(FEAT/scope/name/'X_train.npy',mmap_mode='r'),dtype=np.float32); y=frames['train'].label.eq('essential').astype(int).to_numpy(); appx=app_static(scope,name,frames,aa)
    train_ids=frames['train'].record_id.astype(str).to_numpy(); target=set(train_ids); h=mm_hits[mm_hits.target.isin(target)].copy(); h['identity']=h.pident.astype(float)/100; h['coverage']=h.alnlen.astype(float)/h[['qlen','tlen']].astype(float).max(axis=1); h['score']=h.identity*h.coverage
    labels=dict(zip(train_ids,y.astype(float))); tsp=dict(zip(train_ids,frames['train'].species.astype(str))); qsp=dict(zip(APP_META.record_id.astype(str),APP_META.species.astype(str).map(lambda x:ALIASES.get(x,x)))); app_ids=APP_META.record_id.astype(str).to_numpy(); dynmm=aggregate_mm(h[h['query'].isin(set(app_ids))],app_ids,labels,tsp,qsp)
    appx=np.concatenate([appx,dynmm,ppi_dyn],axis=1)
    if appx.shape[1]!=X.shape[1]: raise RuntimeError(f'{scope}/{name}: app {appx.shape} train {X.shape}')
    m1=lgbm(X,y,42); m2=cat(X,y,42); p1=m1.predict_proba(appx)[:,1]; p2=m2.predict_proba(appx)[:,1]
    exp=np.zeros(len(appx),np.float32); prior=np.full(len(appx),y.mean(),np.float32)
    for sp in frames['train'].species.astype(str).unique():
        ii=np.flatnonzero(frames['train'].species.astype(str).to_numpy()==sp); jj=np.flatnonzero(APP_META.species.astype(str).map(lambda x:ALIASES.get(x,x)).to_numpy()==sp)
        if len(ii)>=30 and len(jj) and len(np.unique(y[ii]))==2:
            ex=lgbm(X[ii],y[ii],123); exp[jj]=ex.predict_proba(appx[jj])[:,1]
            prior[jj]=y[ii].mean()
    exp[(exp==0)]=p1[(exp==0)]
    oof=np.load(ROOT/'eukaryote_model_results/scope_strict_oof_stack'/scope/name/'oof_base.npy'); meta=LogisticRegression(C=1,max_iter=2000,class_weight='balanced',random_state=42).fit(oof,y)
    prob=meta.predict_proba(np.column_stack([p1,p2,exp,prior]))[:,1]
    return prob,float(json.loads((ROOT/'eukaryote_model_results/scope_strict_oof_stack'/scope/name/'metrics.json').read_text())['threshold'])

APP_META=pd.read_csv(APP/'app_metadata.tsv',sep='\t',dtype={'record_id':str}).fillna('')
def main():
    aa=app_arrays(); mm=pd.read_csv(RUN/'app_vs_train.tsv',sep='\t',header=None,names=['query','target','pident','alnlen','evalue','bits','qlen','tlen'],dtype={'query':str,'target':str}); ppi=app_ppi_dynamic({},aa); jobs=[('joint','all_15_species','joint_score'),('clade','protozoa','protozoa_score'),('within_species','Plasmodium_berghei','pberg_within_score'),('within_species','Plasmodium_falciparum','pfal_within_score'),('within_species','Plasmodium_knowlesi','pknow_within_score'),('within_species','Homo_sapiens','human_score')]
    out=APP_META[['record_id','species','gene_id','protein_length']].copy()
    for s,n,col in jobs:
        p,t=legacy_train_tree_only_application(s,n,aa,mm,ppi); out[col]=p; out[col+'_threshold']=t
    out.to_csv(OUT/'final_application_model_scores.tsv',sep='\t',index=False); print('completed',out.shape)
if __name__=='__main__':
    raise SystemExit(
        "This legacy four-branch helper is not an executable formal pipeline. "
        "Use train_species_matched_five_branch_oof.py for formal evaluation "
        "or run_ncomms_full_stack_application_baseline.py for the five-branch atlas."
    )
