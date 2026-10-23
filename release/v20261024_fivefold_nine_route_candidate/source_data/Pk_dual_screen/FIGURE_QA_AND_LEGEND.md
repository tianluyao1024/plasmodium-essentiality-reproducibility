# Dual-screen P. knowlesi supplement: working figure

Status: provisional. This figure is not yet approved for submission and does not replace any original main-figure panel.

## Figure legend draft

**Retrospective comparison of two P. knowlesi screens and projected source-only predictions.** All panels use the same 3,482 unique, one-to-one-mapped gene pairs with definite calls in both screens and a held PKNH label. **a,** Elsworth hybrid model score (HMS) and Oberstaller mutagenesis index score (MIS+) for each pair; colours indicate agreement of the published categorical calls. The Spearman rank correlation is shown; the two scores are not on an interchangeable scale. **b,** Counts and percentages of the 3,482 pairs by the two screens' categorical calls. **c,** Receiver operating characteristic curves for human-red-cell Oberstaller calls and predictions from two source-only LOSO models, trained respectively on Pf + Pb and Pf + Pb + Tg. PKNH predictions are projected to the mapped PKA1H orthologues; the models do not use A1H-specific features. **d,** Empirical cumulative distributions of absolute distance from the threshold chosen on the relevant source validation data, by screen agreement and source combination. This margin is not a calibrated uncertainty estimate.

The two screens differ in parasite strain and experimental procedures as well as red-cell conditions; observed differences cannot be attributed solely to the host cell. Human-screen data and some results were previously inspected. The analysis is retrospective, is not a prospective blinded validation, and is restricted to the stated matched subset. Other-species training labels can include homologues of held proteins, and global unlabelled clustering includes held-species sequences. No claim of strict sequence-level cold start is made.

## Data and quality checks

- `source_gene_pairs.tsv` contains all 3,482 plotted pairs, both screen calls and continuous scores, both model scores, source thresholds and absolute margins. No plot-specific row sampling was used.
- The four categorical cells are 1,410, 369, 66 and 1,637. There are 435 discordant pairs (12.49%).
- Spearman rho is 0.7616. The separately computed gene-pair bootstrap 95% interval is 0.7471–0.7745; it assumes independence between gene pairs and is not drawn as an error bar.
- ROC AUROCs are recomputed from the plotted pairs and matched to the audited human-screen table. The two ROC curves share the same human-screen labels; the figure does not claim a significant difference between curves.
- Panel d shows empirical distributions of all selected genes without a hypothesis-test annotation. PKNH-to-PKA1H mapping, selection rules and upstream input hashes are recorded in the accompanying manifest.
- Python/matplotlib was used for all exports. The source preflight passed 13 checks and issued one DPI detection warning because DPI is supplied through export options; the TIFF was independently checked at 600 × 600 dpi, 4,081 × 3,330 pixels. SVG contains 74 text elements, and PDF text extraction succeeded. The scatter points and heatmap are rasterized within vector exports; text and axes remain vector.
- The PNG was visually inspected for clipping and overlaps at the working layout. Journal-specific final sizing, colour accessibility and final submission QA remain outstanding.
