# Recovered paired-refit source snapshot

This snapshot preserves the source code for the historical 100-repeat paired directional-stability workflow. The driver performs five-fold refits on 80% stratified samples of the species-specific training records, comparing the full feature set with a GO/IPR/Pfam-excluded variant. It is separate from the frozen five-fold nine-route main evaluation and does not alter the manuscript's headline test metrics.

## Verification status

The saved-result audit verified all 100 repeat summaries (455 candidate groups per repeat), with no missing repeats and no input/output hash mismatches. The audit did not read validation or test labels. See `saved_output_integrity_report_20261024.json` and `source_manifest.json`.

This is an integrity audit of archived outputs, not a fresh rerun of all 100 model fits. Repeats 0 and 1 have been freshly reproduced end to end in the current revision work; the remaining 98 are archived-only. The 100-repeat stability result, including the 184/455 group summary, must therefore be described as an audited saved-output analysis, not as 100 newly reproduced fits. The append-only repeat-1 reproduction report and scope addendum are published separately on the same review-candidate branch.

## Source and execution boundaries

The `code/` directory contains the 21-file local dependency closure of the original runner, summarizer, aggregator and saved-output audit. `source_manifest.json` records byte counts and SHA-256 values for every file and the frozen source-snapshot/summarizer hashes.

The original scripts retain historical Windows workspace defaults and depend on external feature matrices, metadata, source-scope files, PPI data and the established project layout; these inputs are not included here. The runner is resumable and may start a missing fit if invoked against an incomplete original output tree. Do not run it against the archived workspace unless you intend that behavior. This snapshot is provenance, not a portable one-command reproduction bundle.

The scripts and audit report do not contain manuscript test/validation outcomes as fitting inputs. The paired-refit analysis is a post-score stability analysis and does not establish experimental essentiality or lineage causation.

