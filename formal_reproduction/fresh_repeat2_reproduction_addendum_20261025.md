# Fresh paired-refit repeat 2 audit addendum

**Status: append-only review-candidate evidence; not a formal archive release or journal submission.** This note documents one additional end-to-end paired refit, repeat 2, and a partial aggregation over three of the 100 historical repeats. It does not replace the frozen v7 candidate or revise its test metrics.

## Repeat 2 reproduction

The recovered five-fold runner was executed for *Plasmodium falciparum*, *P. berghei* and *P. knowlesi*, using the same 80% class-stratified training subset and five-fold allocation for the full 628-feature model and the GO/IPR/Pfam-masked comparison. All six OOF probability arrays and all six application prediction files match the archived repeat-2 files byte for byte. Replaying the saved meta-models reproduces application scores with a maximum absolute error of `1.11e-16`. The manifests report that validation and test labels were not used for fitting or threshold selection.

Across the 455 frozen annotation-retained directional groups, both model variants recovered the reference call pattern in 339 groups for repeat 2, matching the archived repeat-2 summary. This is a single paired refit, not an independent biological screen, prospective prediction or stability estimate.

The machine-readable audit is `fresh_repeat2_reproduction_audit_20261025.json` (SHA-256 `edc9685985f90c702cde6d6a83789bac68043cadf69704c1ff3d6971bc139346`). The paired-summary SHA-256 is `5ddbe99411a24d5858424720d5a5c784f4935d67c0327ed312678eee7c533da7`; the source-snapshot SHA-256 is `08be5ba0d8b03d33e46c8c951a84fa181b22ca7e883d5c2be5f359e3458697b6`.

## Partial 3/100 aggregation

Repeats 0, 1 and 2 have now been freshly reproduced; the other 97 archived repeats remain integrity-audited but were not freshly refit. The three-repeat partial aggregation across 455 groups has a median recovery frequency of `2/3`. Its manifest deliberately sets `final_stable_groups` to null. It must not be interpreted as the historical 184/455 result at the 80/100 rule or as a completed stability analysis.

The aggregate manifest SHA-256 is `a381d9e628aa130e378ce6dcd11884f39c5165be631059ecfff58458f1305b62`. The historical 100-repeat result remains an integrity-audited saved-output analysis; the 184 groups were not recalculated from this partial set.

## Scope and limitations

The paired refits are sensitivity analyses distinct from the frozen five-fold matched-species test evaluation. Historical model-route development inspected Pk test outcomes, so neither the refits nor the frozen metrics are described as blind. Application predictions include records used to fit each refit and are descriptive; they are not an independent performance estimate. Required raw 628-feature matrices and parts of the workspace remain external, so these artifacts do not make the workflow a portable end-to-end model rerun.

The public review-candidate ZIP and canonical manifest remain unchanged. This addendum is not a permanent archive DOI, formal release tag or submission to a journal.

