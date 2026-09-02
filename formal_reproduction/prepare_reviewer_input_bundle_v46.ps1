param(
    [string]$Destination = 'D:\ProtoEss_Reviewer_Inputs_v46'
)

$ErrorActionPreference = 'Stop'
$Workspace = $PSScriptRoot
$Data = Join-Path $Workspace 'eukaryote_model_datasets'
$Results = Join-Path $Workspace 'eukaryote_model_results'

New-Item -ItemType Directory -Force -Path $Destination | Out-Null

function Copy-Tree([string]$Source, [string]$Target) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Target) | Out-Null
    & robocopy $Source $Target /E /COPY:DAT /DCOPY:DAT /R:2 /W:2 /NP /NFL /NDL /NJH /NJS
    if ($LASTEXITCODE -gt 7) { throw "robocopy failed ($LASTEXITCODE): $Source" }
}

# The bundle supports the central matched-species and Tg-augmentation analyses.
# It intentionally omits manuscripts, figures, final results and deployment assets.
foreach ($Species in @('Plasmodium_falciparum', 'Plasmodium_berghei', 'Plasmodium_knowlesi')) {
    Copy-Tree (Join-Path $Data "scope_feature_sets_v2\within_species\$Species") (Join-Path $Destination "eukaryote_model_datasets\scope_feature_sets_v2\within_species\$Species")
    Copy-Tree (Join-Path $Data "unified_v3_experiment_manifests_pre95human\within_species\$Species") (Join-Path $Destination "eukaryote_model_datasets\unified_v3_experiment_manifests_pre95human\within_species\$Species")
}
Copy-Tree (Join-Path $Data 'scope_feature_sets_v2\clade\protozoa') (Join-Path $Destination 'eukaryote_model_datasets\scope_feature_sets_v2\clade\protozoa')
Copy-Tree (Join-Path $Data 'unified_v3_experiment_manifests_pre95human\clade\protozoa') (Join-Path $Destination 'eukaryote_model_datasets\unified_v3_experiment_manifests_pre95human\clade\protozoa')
Copy-Tree (Join-Path $Data 'unified_v3_experiment_manifests_pre95human\joint\all_15_species') (Join-Path $Destination 'eukaryote_model_datasets\unified_v3_experiment_manifests_pre95human\joint\all_15_species')
Copy-Tree (Join-Path $Data 'features\unified_v3_current_mmseqs_neighbors') (Join-Path $Destination 'eukaryote_model_datasets\features\unified_v3_current_mmseqs_neighbors')
Copy-Tree (Join-Path $Results 'strict_oof_dynamic_stack\mapped_edges') (Join-Path $Destination 'eukaryote_model_results\strict_oof_dynamic_stack\mapped_edges')
# Application-side matrices and the cached all-versus-training mappings are
# required for the genome-scale atlas stage.  They are processed derivatives of
# public resources and remain in the confidential reviewer bundle only.
Copy-Tree (Join-Path $Data 'genomewide_target_application') (Join-Path $Destination 'eukaryote_model_datasets\genomewide_target_application')
Copy-Tree (Join-Path $Data 'features\genomewide_target_application') (Join-Path $Destination 'eukaryote_model_datasets\features\genomewide_target_application')
Copy-Tree 'D:\eukaryote_model_runs_go_ipr_pfam\genomewide_target_application\final_application_features' (Join-Path $Destination 'application_feature_cache')
# Fixed after-scoring context tables and GO resources are needed for the atlas
# and directional-interpretation stages.  They contain identifiers and public
# annotations, not manuscript text, figures or final submission artefacts.
$Context = Join-Path $Destination 'eukaryote_model_results\reviewer_context'
New-Item -ItemType Directory -Force -Path $Context | Out-Null
Import-Csv (Join-Path $Workspace 'protozoa_human_sci_analysis\reviewer_revision\sensitive_homology_priority_closure\Complete_atlas_15591_genes_host_closed.tsv') -Delimiter "`t" |
    Select-Object species,gene_id,homology_group_id |
    Export-Csv (Join-Path $Context 'orthogroup_membership.tsv') -Delimiter "`t" -NoTypeInformation -Encoding utf8
Copy-Item (Join-Path $Workspace 'protozoa_human_sci_analysis\reviewer_revision\host_risk_revised_atlas\all_three_plasmodium_genes_host_risk_revised.tsv') (Join-Path $Context 'all_three_plasmodium_genes_host_risk_revised.tsv') -Force
Copy-Item (Join-Path $Workspace 'protozoa_human_sci_analysis\reviewer_revision\ncomms_submission_v44\Source_Data\Supplementary_FigS5a_target_registry_members.tsv') (Join-Path $Context 'target_registry_members.tsv') -Force
Copy-Tree (Join-Path $Workspace 'downloads\go_current') (Join-Path $Destination 'downloads\go_current')

$files = Get-ChildItem -LiteralPath $Destination -Recurse -File | Sort-Object FullName
$rows = foreach ($file in $files) {
    [PSCustomObject]@{
        relative_path = $file.FullName.Substring($Destination.Length).TrimStart('\')
        bytes = $file.Length
        sha256 = ''
    }
}
$rows | Export-Csv -LiteralPath (Join-Path $Destination 'input_manifest_sha256.csv') -NoTypeInformation -Encoding utf8
$metadata = [PSCustomObject]@{
    bundle = 'ProtoEss reviewer-only core processed inputs v46'
    purpose = 'reproduce species-matched OOF, protozoan LOSO/Tg-augmentation analyses and the genome-scale atlas'
    created_utc = (Get-Date).ToUniversalTime().ToString('o')
    file_count = $rows.Count
    total_bytes = [long](($rows | Measure-Object -Property bytes -Sum).Sum)
    access = 'confidential peer-review material; not a public repository release'
}
$metadata | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Destination 'bundle_metadata.json') -Encoding utf8
Write-Output (Join-Path $Destination 'bundle_metadata.json')
