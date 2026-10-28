# Figure 6d exact-decile correction (review only)

This addendum corrects a floating-point cutoff error in the v7 review candidate. It is not a final archive, a new model fit, or a blind evaluation. The v7 bundle and its source files remain unchanged.

For P. knowlesi, exact integer ceilings select 1,224 instead of 1,225 records at 30%, and 2,856 instead of 2,857 at 70%. The excluded records are non-targets, so target-hit counts and plotted recall do not change. Fisher one-sided enrichment P values become 0.008836105284403978 and 2.2679519449758756e-08 respectively. Other candidate issues and the pending fresh 100-repeat audit still apply.

## Reproduce

Install numpy, pandas, and scipy in a Python environment, then run from the repository root:

```sh
python formal_reproduction/fig6d_exact_decile_review_20261028/fig6_exact_decile_portable.py release/v20261024_fivefold_nine_route_candidate_v7/source_data/Fig6/target_gene_audit.tsv output.tsv
```

The script reads the frozen target-supported registry, ranks by the saved LOSO score with a stable sort, and uses integer ceiling for each decile. It refuses to overwrite an existing output. Compare all 30 rows against `Fig6d_exact_decile_REVIEW.tsv`. This is a post-score computational prioritization analysis, not experimental validation.
