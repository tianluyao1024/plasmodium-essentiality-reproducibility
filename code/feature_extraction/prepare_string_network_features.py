from __future__ import annotations

import gzip
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT=Path.cwd(); DATA="dedup95_strict_fixed"; F=ROOT/"eukaryote_model_datasets"/"features"/DATA
S=ROOT/"eukaryote_model_datasets"/DATA/"random_species_label_split_seed42"; D=ROOT/"downloads"/"string_v12"
OUT=F/"string_network"; OUT.mkdir(parents=True,exist_ok=True); D.mkdir(parents=True,exist_ok=True)
MM=ROOT/"tools"/"mmseqs2"/"mmseqs"/"bin"/"mmseqs.exe"
EXCLUDE={"Danio_rerio","Candida_albicans__A21","Bombyx_mori"}
TAX={"Arabidopsis_thaliana":3702,"Caenorhabditis_elegans":6239,"Candida_albicans__SC5314":237561,
"Candida_glabrata__CBS138":284593,"Drosophila_melanogaster":7227,"Homo_sapiens":9606,"Mus_musculus":10090,
"Oryza_sativa_japonica":39947,"Pichia_pastoris___Komagataella_phaffii__species_level":644223,
"Plasmodium_berghei":5823,"Plasmodium_falciparum":36329,"Plasmodium_knowlesi":5851,
"Saccharomyces_cerevisiae__species_level":4932,"Schizosaccharomyces_pombe__species_level":284812,"Toxoplasma_gondii":432359}
NAMES=["string_mapped","string_map_identity","string_map_coverage","degree_score400","degree_score700","degree_score900",
"weighted_degree","mean_edge_score","max_edge_score"]

def download(url,path):
 if path.exists() and path.stat().st_size>100:return
 part=path.with_suffix(path.suffix+".part"); headers={}
 mode="wb"
 if part.exists() and part.stat().st_size:
  headers["Range"]=f"bytes={part.stat().st_size}-";mode="ab"
 with requests.get(url,stream=True,timeout=120,headers=headers) as r:
  if r.status_code==200 and mode=="ab":mode="wb"
  r.raise_for_status()
  with part.open(mode) as fh:
   for chunk in r.iter_content(1024*1024):
    if chunk:fh.write(chunk)
 part.replace(path)

def read_fasta(path):
 d={};k=None;s=[]
 with path.open(encoding="utf-8") as fh:
  for line in fh:
   line=line.strip()
   if line.startswith(">"):
    if k is not None:d[k]="".join(s)
    k=line[1:].split()[0];s=[]
   else:s.append(line)
 if k is not None:d[k]="".join(s)
 return d

def network_stats(path):
 vals={}
 def get(k):
  if k not in vals:vals[k]=[0,0,0,0.0,0.0,0.0]
  return vals[k]
 with gzip.open(path,"rt",encoding="utf-8") as fh:
  next(fh)
  for line in fh:
   a,b,score=line.split();sc=float(score)
   for k in (a,b):
    x=get(k);x[0]+=sc>=400;x[1]+=sc>=700;x[2]+=sc>=900;x[3]+=sc/1000;x[4]+=1;x[5]=max(x[5],sc/1000)
 return {k:[v[0],v[1],v[2],v[3],v[3]/max(v[4],1),v[5]] for k,v in vals.items()}

def main():
 allseq={}
 for split in ("train","val","test"):allseq.update(read_fasta(S/f"{split}.protein.fasta"))
 b=F/"basic_seq"; names0=json.loads((b/"species_to_id.json").read_text());idname={int(v):k for k,v in names0.items()}
 split_data={}
 for split in ("train","val","test"):
  ids=np.load(b/f"{split}_ids.npy",allow_pickle=True).astype(str);sp=np.load(b/f"{split}_species.npy").astype(int);names=np.asarray([idname[x] for x in sp]);keep=~np.isin(names,list(EXCLUDE));split_data[split]=(ids[keep],names[keep])
 all_ids=np.concatenate([x[0] for x in split_data.values()]);all_species=np.concatenate([x[1] for x in split_data.values()]);feat={}
 for species,tax in TAX.items():
  seqgz=D/f"{tax}.protein.sequences.v12.0.fa.gz";links=D/f"{tax}.protein.links.v12.0.txt.gz"
  download(f"https://stringdb-downloads.org/download/protein.sequences.v12.0/{seqgz.name}",seqgz)
  download(f"https://stringdb-downloads.org/download/protein.links.v12.0/{links.name}",links)
  target=D/f"{tax}.protein.sequences.v12.0.fa"
  if not target.exists():
   with gzip.open(seqgz,"rb") as src,target.open("wb") as dst:
    while chunk:=src.read(1024*1024):dst.write(chunk)
  q=D/f"query_{tax}.fa"
  ids=all_ids[all_species==species]
  if not q.exists():
   with q.open("w",encoding="utf-8") as fh:
    for rid in ids:fh.write(f">{rid}\n{allseq[rid]}\n")
  hit=D/f"map_{tax}.tsv"
  if not hit.exists():subprocess.run([str(MM),"easy-search",str(q),str(target),str(hit),str(D/f"tmp_map_{tax}"),"--format-output","query,target,pident,alnlen,evalue,bits,qlen,tlen","--max-seqs","1","-s","7.5","--threads","8"],check=True)
  h=pd.read_csv(hit,sep="\t",header=None,names=["q","t","pid","aln","e","bits","qlen","tlen"]);h["cov"]=h["aln"]/h[["qlen","tlen"]].max(axis=1);h=h.sort_values(["q","bits"],ascending=[True,False]).drop_duplicates("q").set_index("q")
  stats=network_stats(links);mapped=0
  for rid in ids:
   if rid not in h.index:feat[rid]=[0,0,0,0,0,0,0,0,0];continue
   r=h.loc[rid]; st=stats.get(str(r.t),[0,0,0,0,0,0]);good=float(r.pid)>=70 and float(r["cov"])>=.6;mapped+=good
   feat[rid]=[float(good),float(r.pid)/100,float(r["cov"]),*st] if good else [0,float(r.pid)/100,float(r["cov"]),0,0,0,0,0,0]
  print(species,"mapped",mapped,"of",len(ids),flush=True)
 for split,(ids,names) in split_data.items():
  arr=np.asarray([feat.get(i,[0]*9) for i in ids],dtype=np.float32);np.save(OUT/f"{split}_emb.npy",arr);np.save(OUT/f"{split}_ids.npy",ids)
 (OUT/"feature_names.json").write_text(json.dumps(NAMES,indent=2),encoding="utf-8")
if __name__=="__main__":main()
