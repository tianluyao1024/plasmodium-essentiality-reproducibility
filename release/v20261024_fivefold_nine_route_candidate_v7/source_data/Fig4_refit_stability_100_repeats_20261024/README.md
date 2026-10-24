# Fig. 4 paired-refit stability evidence

This package contains the 100 saved paired-refit recovery tables (455 directional candidate groups per repeat), sanitized per-repeat manifests, and the aggregate recovery table used for the frozen Fig. 4d analysis.

The verified rule is recovery of the same reference directional pattern by both the full and annotation-excluded model variants. A group is called stable when recovery is at least 80/100. The archived results reproduce 184 stable groups. `audit_report.json` records the value- and hash-level comparison against the aggregate and frozen Fig. 4 table.

The package does not refit models. The fitting inputs and source snapshots were audited separately, but the exact orchestration command used for all 100 fitting repeats is not pinned in the run manifests. These files therefore support audit of saved stability outputs, not a claim of end-to-end repeat refitting. The analysis is not an independent validation set and provides no experimental confirmation.
