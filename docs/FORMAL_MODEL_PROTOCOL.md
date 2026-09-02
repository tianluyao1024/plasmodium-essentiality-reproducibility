# Formal model and deployment model protocol

## Scope and boundary

The public resource includes four source-labelled species: *Plasmodium falciparum*, *P. berghei*, *P. knowlesi* and *Toxoplasma gondii*. For each it provides processed labelled feature matrices, class-separated CDS FASTA records, practical deployment models and training/prediction scripts. *T. gondii* is included to reproduce its source/held-out role in the apicomplexan transfer analyses; it is not a member of the final three-species malaria atlas.

The public repository does not contain the manuscript, genome-wide atlas calls, target-ranking tables, docking outputs, figure source data or unpublished analysis workbooks.

## Locked formal manuscript model

The manuscript's central model is a five-input OOF stack: LightGBM, CatBoost, a species-aware expert, a fold-local species-prevalence branch and a compact Transformer feature-integration branch. L2-regularised logistic regression combines OOF probabilities. Every dynamic feature, preprocessing parameter and threshold is fitted from the corresponding training partition only.

The static molecular representation contains 591 columns: 128 ESM2, 128 ProtT5, 31 sequence descriptors, 128 annotation, 128 k-mer/graph, 15 MMseqs30, 15 MMseqs50 and 18 label-free STRING columns. In a species-matched analysis it is combined with one species indicator, 21 fold-local MMseqs2-neighbourhood and 15 fold-local STRING/PPI-neighbourhood columns, giving 628 columns. Clade/LOSO source representations use four one-hot species columns.

The complete formal source is public in [`../formal_reproduction/`](../formal_reproduction/). The confidential fixed input bundle and split manifests are supplied to editors/reviewers during peer review and will be publicly deposited on publication.

## Practical deployment models

`models/` contains two-base-learner LightGBM/CatBoost OOF deployment ensembles trained from the public matrices. These models are used for transparent reuse and are deliberately distinct from the locked five-branch stack; they must not be used to recalculate the manuscript's held-out metrics, atlas calls or target-ranking results.
