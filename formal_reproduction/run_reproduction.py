from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

PACKAGE = Path(__file__).resolve().parent
CODE = PACKAGE / 'code'
STAGES = {
    'matched': ['train_species_matched_five_branch_oof.py'],
    'transfer': ['train_clade_loso_full_stack.py', 'run_tg_shared_feature_loso_control.py'],
    'robustness': ['run_cluster_blocked_homology_baseline.py', 'run_protozoa_label_uncertainty.py', 'run_ncomms_annotation_excluded.py'],
    'pharmacology': ['run_loso_essentiality_conditioned_pharmacology.py', 'run_canonical_three_species_conserved_target_screen.py', 'run_exact_triplet_eligibility_matched_null.py'],
    'atlas': ['run_ncomms_full_stack_application_baseline.py', 'build_inner_validation_essentiality_atlas.py'],
    'directionality': ['run_ncomms_directional_audits.py', 'run_directional_resampling_refit_stability.py'],
}

def main() -> None:
    parser = argparse.ArgumentParser(description='Run the reviewer-accessible ProtoEss reproduction workflow.')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=[*STAGES, 'all'], default='all')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    input_root = Path(cfg['formal_input_root']).expanduser().resolve()
    application_feature_root = Path(cfg.get('application_feature_root', input_root / 'application_feature_cache')).expanduser().resolve()
    output_root = Path(cfg['output_root']).expanduser().resolve()
    feature_root = input_root / 'eukaryote_model_datasets'
    download_root = input_root / 'downloads'
    edge_cache = input_root / 'eukaryote_model_results' / 'strict_oof_dynamic_stack' / 'mapped_edges'
    missing = [] if feature_root.exists() and (download_root.exists() or edge_cache.exists()) else [
        'eukaryote_model_datasets',
        'downloads or eukaryote_model_results/strict_oof_dynamic_stack/mapped_edges',
    ]
    if args.stage in ('atlas', 'all') and not (application_feature_root / 'app_vs_train.tsv').exists():
        missing.append('application_feature_root/app_vs_train.tsv')
    if missing:
        raise SystemExit('Input root is incomplete: ' + '; '.join(missing))
    output_root.mkdir(parents=True, exist_ok=True)
    # Several legacy entry points launch sibling scripts by filename while
    # deliberately treating cwd as the immutable data root.  Short-lived
    # wrappers keep that data-root contract while resolving every sibling from
    # this self-contained reviewer package.
    wrappers = []
    for source in CODE.glob('*.py'):
        wrapper = input_root / source.name
        if wrapper.exists():
            continue
        wrapper.write_text("import runpy\nrunpy.run_path(" + repr(str(source)) + ", run_name='__main__')\n", encoding='utf-8')
        wrappers.append(wrapper)
    env = os.environ.copy()
    env.update({
        'PYTHONPATH': str(CODE) + os.pathsep + env.get('PYTHONPATH', ''),
        'PROTOESS_REPRODUCTION_CONFIG': str(args.config.resolve()),
        'PROTOESS_REPRODUCTION_OUTPUT': str(output_root),
        'NCOMMS_CANONICAL_FIVE_BRANCH_ROOT': str(output_root / 'matched_species_five_branch'),
        'NCOMMS_ANNOTATION_EXCLUDED_OUT': str(output_root / 'annotation_excluded'),
        'NCOMMS_FULL_APPLICATION_OUT': str(output_root / 'full_application'),
        'NCOMMS_FULL_GENOMEWIDE_TABLE': str(output_root / 'full_application' / 'full_formal_stack_genomewide_all_three.tsv'),
        'NCOMMS_EXCLUDED_GENOMEWIDE_TABLE': str(output_root / 'annotation_excluded' / 'functional_annotation_excluded_genomewide_all_three.tsv'),
        'NCOMMS_DIRECTIONAL_OUT': str(output_root / 'directional_audits'),
        'NCOMMS_DIRECTIONAL_INPUT_ROOT': str(output_root / 'directional_audits'),
        'NCOMMS_DIRECTIONAL_REFIT_OUT': str(output_root / 'directional_refits'),
        'FORMAL_APPLICATION_FEATURE_ROOT': str(application_feature_root),
        'PROTOESS_FORMAL_INPUT_ROOT': str(input_root),
        'PROTOESS_REVIEW_CONTEXT_ROOT': str(input_root / 'eukaryote_model_results' / 'reviewer_context'),
        'TRANSFORMER_BATCH': str(cfg.get('transformer_batch_size', 64)),
    })
    stages = list(STAGES) if args.stage == 'all' else [args.stage]
    commands = []
    for stage in stages:
        for script in STAGES[stage]:
            command = [sys.executable, str(CODE / script)]
            if script == 'train_species_matched_five_branch_oof.py':
                command.extend(['--output-root', str(output_root / 'matched_species_five_branch')])
            commands.append(command)
    manifest = {'config': str(args.config.resolve()), 'input_root': str(input_root), 'output_root': str(output_root), 'stages': stages, 'commands': commands, 'python': sys.version}
    (output_root / 'reproduction_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    try:
        for command in commands:
            print('RUN', ' '.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=input_root, env=env, check=True)
    finally:
        for wrapper in wrappers:
            wrapper.unlink(missing_ok=True)

if __name__ == '__main__':
    main()
