# Recovered historical analysis sources (2026-10-24)

This folder preserves source files recovered from the local working repository and verified against the pre-run source snapshot for the 2026-10-17 paired-refit pilot. The 15 snapshot-listed code files match all 15 SHA-256 values in that snapshot; the snapshot as a whole contains 77 input/code files, all 77 still present locally with matching hashes. Two later aggregation scripts are also preserved with the hashes recorded by the completed 100-repeat aggregation and final table manifest.

## Scope and limits

- These are historical source snapshots, not a portable reproduction environment. Several scripts use machine-specific paths, and the 77 input files are not redistributed in this folder.
- The 15-file snapshot includes the nine-route application/pilot code as it existed on 2026-10-17. It is not a claim that the current v6 frozen model has been refit from source here.
- The exact 100-repeat stability table builder and paired-repeat aggregator are included. The `efa89...` hash in the final table manifest identifies `build_final_stability_tables_20261023.py`; it is not the 100-repeat model-fitting runner.
- The exact model-fitting runner version used for all 100 repeats is not identified by the repeat manifests. The current working copy of `run_directional_resampling_refit_stability.py` has a different hash from the final-table builder and is not represented as a verified historical runner here.
- Refitting the nine-route frozen model from source feature matrices, reproducing all paper analyses, and promoting these files into the formal v6 release bundle remain separate tasks. The v6 candidate manifest and website payload are unchanged by this supporting archive.

## Verification

`source_hashes.tsv` records the expected byte size and SHA-256 of each archived source file. The pilot partition unit tests are included. To run those tests from this directory:

```powershell
Push-Location code
python -m unittest test_nine_route_refit_partitions_20261017.py
Pop-Location
```

The upstream pre-run snapshot manifest and the local 77/77 input hash audit remain in the authors' working files; no local absolute paths are embedded here.
