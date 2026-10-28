"""Recalculate target recovery with exact integer decile cutoffs.

This is a post-score registry analysis; it does not fit or validate a model.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

SPECIES={'Plasmodium_falciparum':'Pf','Plasmodium_berghei':'Pb','Plasmodium_knowlesi':'Pk'}

def recalculate(audit):
    required={'record_id','species','loso_score','target_set','is_target'}
    if not required.issubset(audit.columns):
        raise ValueError(f'Missing columns: {sorted(required-set(audit.columns))}')
    selected=audit.loc[audit.target_set.eq('Tier 1+2 target-supported registry')].copy()
    if selected.empty or selected.species.isna().any() or not selected.is_target.isin([True,False]).all():
        raise ValueError('Target flags must be non-null Booleans')
    rows=[]
    for species, group in selected.groupby('species',sort=False):
        if species not in SPECIES: raise ValueError(f'Unknown malaria species: {species}')
        if group.record_id.isna().any() or group.record_id.duplicated().any() or not np.isfinite(group.loso_score).all():
            raise ValueError(f'Invalid identifiers or scores: {species}')
        ranked=group.sort_values('loso_score',ascending=False,kind='mergesort').reset_index(drop=True)
        n=len(ranked); total=int(ranked.is_target.sum())
        if total==0: raise ValueError(f'No mapped targets: {species}')
        for k in range(1,11):
            size=(n*k+9)//10
            hits=int(ranked.iloc[:size].is_target.sum())
            odds,p=fisher_exact([[hits,size-hits],[total-hits,n-size-(total-hits)]],alternative='greater')
            rows.append({'species_short':SPECIES[species],'species':species,'top_fraction':k/10,'held_labelled_genes':n,'selected_genes':size,
                         'mapped_target_genes':total,'target_hits':hits,'target_recall':hits/total,
                         'fisher_odds_ratio':odds,'fisher_p_greater':p})
    return pd.DataFrame(rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target_gene_audit',type=Path)
    parser.add_argument('output_tsv',type=Path)
    args=parser.parse_args()
    if args.output_tsv.exists(): raise FileExistsError(args.output_tsv)
    data=pd.read_csv(args.target_gene_audit,sep='\t')
    result=recalculate(data)
    result.to_csv(args.output_tsv,sep='\t',index=False)
    print(f'Wrote {len(result)} rows to {args.output_tsv}; source order breaks score ties.')
if __name__=='__main__': main()
