# Data dictionary

## `predictions.tsv`

`species`, `gene_id`, `score`, `binary_call`, `threshold`, `model_version`, `feature_version`, `screen_condition`, `evidence_class`. The score and binary call are the frozen matched-species computational predictions.

## Other files

- `loso_metrics.tsv`: four held-species source-only LOSO metrics, including Tg; retrospective and not blind.
- `source_composition_metrics.tsv`: two malaria sources versus the same sources plus Tg, on identical held-out records; sensitivity comparison, not causal evidence.
- `pk_dual_screen_gene_pairs.tsv`: matched PKNH/PKA1H gene-pair screen labels, continuous scores and source-only projection scores.
- `pk_dual_screen_summary.tsv`: audited denominators, discordance rate and continuous-score association.
- `s4_label_exchange_runs.tsv` and `s4_label_exchange_summary.tsv`: source-training label-exchange sensitivity; validation and held labels remain unchanged.

All models and analyses are retrospective. Read `MODEL_CARD.md` before interpreting values.
