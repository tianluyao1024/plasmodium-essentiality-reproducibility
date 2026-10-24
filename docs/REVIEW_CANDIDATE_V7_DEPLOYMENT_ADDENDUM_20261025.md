# v7 review-candidate deployment addendum

Date: 2026-10-25

This addendum records a website deployment that occurred after the v7 package README was prepared. It updates deployment status only; the frozen model outputs, prediction tables, candidate ZIP, and canonical analysis manifest are unchanged.

## Deployment verification

- The ProtoEss Atlas serves `v20261024_fivefold_nine_route_candidate_v7` in the separate **Revision candidate** tab at https://www.protoessatlas.com.
- The production app points to the versioned v7 data directory. The production and public Streamlit health endpoints returned HTTP 200 with `ok`; the app test passed with zero exceptions.
- All 25 files listed by the site-specific v7 payload manifest passed SHA-256 verification in both the integrated source and production data directories.
- The v7 canonical manifest SHA-256 remains `9a7a0541f673cc7eb339f405d63fea05da21dc40ab6fdc6e08158243237e7d3a`. The convenience review bundle remains SHA-256 `92b8bb50cd9b0a22fbc3df4887db7be3e2d1c859686d0bca18fd59095b257baf`.

## Scope and status

The v7 directory README is retained as the original build-time snapshot; its predeployment sentence is superseded by this addendum and the repository-level README. The v6 candidate and legacy atlas remain preserved. This is still a public review candidate, not a formal archive release or journal submission. No candidate-specific DOI or release tag exists, the official Nature Portfolio Reporting Summary and portal declarations remain open, and no journal submission has been made.

The five-fold nine-route model uses the frozen 628 raw features and original label-stratified split. Matched-species test results remain retrospective because P. knowlesi outcomes were historically inspected. The P. knowlesi human-red-cell analysis remains a retrospective homolog projection, not an A1H sequence-level or blind external prediction. Saved refit outputs are integrity-audited; repeats 0, 1 and 2 were freshly reproduced, while the other 97 remain archived-only. Repeat 2 is documented in the [machine-readable audit](../formal_reproduction/fresh_repeat2_reproduction_audit_20261025.json) and [scope addendum](../formal_reproduction/fresh_repeat2_reproduction_addendum_20261025.md). The partial 3/100 aggregation is not a final stability estimate.
