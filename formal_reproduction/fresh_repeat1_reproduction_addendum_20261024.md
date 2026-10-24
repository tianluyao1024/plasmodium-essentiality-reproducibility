# Fresh paired-refit repeat-1 audit addendum

Date: 2026-10-24  
Scope: one fresh end-to-end rerun of paired repeat 1, not a fresh 100-repeat reconstruction.

The isolated repeat-1 workflow covered *Plasmodium falciparum*, *P. berghei* and *P. knowlesi*, each with the full-feature and GO/IPR/Pfam-masked variants. All six OOF arrays and all six application-prediction tables were byte-identical to their archived repeat-1 outputs. Saved application models reproduced their stored scores with maximum absolute error at or below 1.11e-16. Each within-species variant pair used the same selected training records and five-fold assignments. Run manifests state that validation/test labels were not used for fitting or threshold selection.

The repeat-1 summary covers 455 candidate groups, with 336 recovered by both variants; its SHA-256 is `1094aa9dd4732bc1e70a37dcb4cf3a893c603a47bfcff84451e5c247036813ca`. The machine-readable report records the hashes for each output and has zero audit errors.

This is a follow-up to the integrity audit of the 100 archived repeat outputs. Only repeats 0 and 1 have been freshly reproduced end to end; the 184/455 result at the historical 80/100 rule remains an archived 100-repeat result, not a new 100-fit rerun. This addendum does not change the frozen matched-species test metrics or make any result a blind external validation.

The public candidate ZIP is not rewritten by this addendum. The fresh repeat-1 audit report is in `fresh_repeat1_reproduction_audit_20261024.json` in the same directory.
