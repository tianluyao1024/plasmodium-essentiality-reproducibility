"""Build the v57 Nature Communications package with directional-context Fig. 5.

The package is assembled from v56 frozen outputs.  This build adds no newly
fitted model and does not alter the matched-species atlas or LOSO predictions.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
import zipfile

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches
from docx.text.paragraph import Paragraph
from docx.oxml import OxmlElement


ROOT = Path(__file__).resolve().parent
V56 = ROOT / "protozoa_human_sci_analysis" / "reviewer_revision" / "ncomms_submission_v56_loso_full_workflow_validation"
OUT = ROOT / "protozoa_human_sci_analysis" / "reviewer_revision" / "ncomms_submission_v57_directional_context_fig6"
FIG = OUT / "Main_Figures"
SOURCE = OUT / "Source_Data"
UPLOAD = OUT / "Upload_Files_v57"


def ignored(_: str, names: list[str]) -> set[str]:
    excluded = {
        "_qc_png",
        "_render_v56",
        "_render_v56_final",
        "Main_manuscript_with_all_figures_v56.docx",
        "Main_manuscript_with_all_figures_v56.pdf",
        "Manuscript.docx",
        "01_Main_manuscript_with_figures.docx",
    }
    return {name for name in names if name in excluded or name.endswith(".zip")}


def stage_v56() -> None:
    """Copy reusable files while intentionally excluding old main-manuscript assets."""
    OUT.mkdir(parents=True, exist_ok=True)
    for item in V56.iterdir():
        if item.name in {
            "_qc_png",
            "_render_v56",
            "_render_v56_final",
            "Main_manuscript_with_all_figures_v56.docx",
            "Main_manuscript_with_all_figures_v56.pdf",
            "Manuscript.docx",
        } or item.suffix == ".zip":
            continue
        dest = OUT / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True, ignore=ignored)
        else:
            shutil.copy2(item, dest)

    # Recover the new Figure 5 source-data names if an earlier incremental build
    # applied a broad figure-number rename.  These names are not LOSO inputs.
    contextual_names = [
        ("Figure6_figure_notes.md", "Figure5_figure_notes.md"),
        ("Figure6a_external_orthology_support.tsv", "Figure5a_external_orthology_support.tsv"),
        ("Figure6b_sequence_identity_matched_comparison.tsv", "Figure5b_sequence_identity_matched_comparison.tsv"),
        ("Figure6c_GO_enrichment.tsv", "Figure5c_GO_enrichment.tsv"),
        ("Figure6d_group_level_evidence_matrix.tsv", "Figure5d_group_level_evidence_matrix.tsv"),
    ]
    for stale_name, restored_name in contextual_names:
        stale = SOURCE / stale_name
        restored = SOURCE / restored_name
        if stale.exists() and not restored.exists():
            stale.replace(restored)
        elif stale.exists() and restored.exists():
            # A prior incremental package build left a duplicate contextual
            # source-data file with the old number.  It is regenerated below
            # under the canonical Figure 5 name.
            stale.unlink()

    # Renumber only the old LOSO Figure 5 assets.  The new contextual Figure 5
    # also has source-data names beginning with Figure5, so a wildcard rename
    # would corrupt the source-data/figure correspondence.
    for path in sorted(FIG.glob("Fig5_LOSO*")):
        path.replace(path.with_name(path.name.replace("Fig5_LOSO", "Fig6_LOSO")))
    loso_source_names = [
        "Fig5_LOSO_full_workflow_run_manifest.json",
        "Fig5a_LOSO_target_label_naive_target_ranking.tsv",
        "Fig5b_LOSO_label_length_matched_target_control.tsv",
        "Figure5a_LOSO_binary_performance.tsv",
        "Figure5b_c_target_enrichment_and_member_concordance.tsv",
        "Figure5d_LOSO_target_cumulative_recovery.tsv",
    ]
    for name in loso_source_names:
        path = SOURCE / name
        if path.exists():
            destination = (
                name.replace("Fig5_LOSO", "Fig6_LOSO")
                .replace("Fig5a_", "Fig6a_")
                .replace("Fig5b_", "Fig6b_")
                .replace("Figure5", "Figure6")
            )
            path.replace(SOURCE / destination)


def insert_before(anchor: Paragraph, text: str = "", style=None) -> Paragraph:
    new_p = OxmlElement("w:p")
    anchor._p.addprevious(new_p)
    paragraph = Paragraph(new_p, anchor._parent)
    if style is not None:
        paragraph.style = style
    if text:
        paragraph.add_run(text)
    return paragraph


def copy_paragraph_properties(target: Paragraph, source: Paragraph) -> None:
    if source._p.pPr is not None:
        target._p.insert(0, deepcopy(source._p.pPr))


def replace_runs(paragraph: Paragraph, old: str, new: str) -> None:
    matched = False
    for run in paragraph.runs:
        if old in run.text:
            run.text = run.text.replace(old, new)
            matched = True
    if not matched and old in paragraph.text:
        # Captions and normal prose in the current source each keep this text in one run.
        # This fallback preserves the paragraph style when a string spans runs.
        paragraph.text = paragraph.text.replace(old, new)


def find_paragraph(doc: Document, prefix: str) -> Paragraph:
    for p in doc.paragraphs:
        if p.text.strip().startswith(prefix):
            return p
    raise ValueError(f"Paragraph not found: {prefix}")


def add_figure5_section(doc: Document) -> None:
    anchor = find_paragraph(doc, "Held-species essentiality calls recapitulate target enrichment")
    old_caption = find_paragraph(doc, "Fig. 6 | Frozen held-species")

    heading = insert_before(anchor, "Comparative sequence and functional context of refit-stable directional orthogroups", style=anchor.style)
    heading.runs[0].bold = True
    insert_before(
        anchor,
        "The directional layer was defined before contextual annotation was inspected. Forty-two of the 46 refit-stable exact Pf-Pb-Pk triplets were independently supported by a shared OrthoMCL 6.21 group (Fig. 5a and Supplementary Fig. S8). Compared with 46 exact E/E/E triplets matched one-to-one for mean protein length, directional triplets had lower mean pairwise global amino-acid identity (median 0.611 versus 0.747; two-sided Mann-Whitney P = 0.016; Benjamini-Hochberg FDR = 0.049; Fig. 5b and Supplementary Table S9). This comparison situates the stable directional calls in comparative sequence space; it does not redefine their call patterns or rank them by divergence.",
    )
    insert_before(
        anchor,
        "Three Gene Ontology terms passed the prespecified multiple-testing threshold: cytoplasmic dynein complex, dynein intermediate-chain binding and microtubule (two groups for each term; FDR = 0.048; Fig. 5c and Supplementary Table S10). The all-group matrix exposes both the species call states and the independent orthology/refit evidence for every directional triplet (Fig. 5d). Release-to-release mapping coverage was insufficient for a parallel formal enrichment test of domain, localisation or life-cycle annotations; these fields remain available as record-level context in Supplementary Tables S5-S8 rather than being promoted to a group-level signal.",
    )

    # Append an image paragraph, then move it to the exact Results location.
    shape = doc.add_picture(str(FIG / "Fig5_refit_stable_directional_comparative_context.png"), width=Inches(6.25))
    image_p = doc.paragraphs[-1]
    image_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    anchor._p.addprevious(image_p._p)

    caption = insert_before(anchor)
    copy_paragraph_properties(caption, old_caption)
    first = caption.add_run("Fig. 5 | Refit-stable directional orthogroups show comparative sequence and functional context. ")
    first.bold = True
    caption.add_run(
        "a, Independent OrthoMCL 6.21 support for the 46 refit-stable exact Pf-Pb-Pk triplets; support requires all three model genes to occur in a shared external group. "
        "b, Mean pairwise global amino-acid identity for all 46 directional triplets and 46 exact E/E/E triplets matched one-to-one for mean protein length. Violin envelopes show distributions, points show individual triplets and horizontal lines show medians; the two-sided Mann-Whitney P value was adjusted across three prespecified comparisons by Benjamini-Hochberg correction. "
        "c, GO terms with FDR <= 0.05 in the 46 directional groups relative to 4,183 exact triplets; each term annotates two directional groups. "
        "d, All 46 groups ordered by call pattern and sequence identity. In Pf/Pb/Pk call tracks, dark indicates predicted essential and pale indicates predicted non-essential; the OrthoMCL track shows external support and the refit track encodes reference-pattern recovery frequency across 100 refits. These downstream comparative annotations do not change a gene-level essentiality score or atlas call. Source data are provided with this paper."
    )


def update_docx() -> Path:
    doc = Document(V56 / "Main_manuscript_with_all_figures_v56.docx")
    # Existing Fig. 5 is the held-species workflow validation and becomes Fig. 6.
    for paragraph in doc.paragraphs:
        replace_runs(paragraph, "Fig. 5", "Fig. 6")
        replace_runs(paragraph, "Figure 5", "Figure 6")
    add_figure5_section(doc)
    out_path = OUT / "Main_manuscript_with_all_figures_v57.docx"
    doc.save(out_path)
    shutil.copy2(out_path, OUT / "Manuscript.docx")
    return out_path


def update_markdown() -> None:
    text = (V56 / "Manuscript.md").read_text(encoding="utf-8")
    text = text.replace("Fig. 5", "Fig. 6").replace("Figure 5", "Figure 6")
    section = """
### Comparative sequence and functional context of refit-stable directional orthogroups

The directional layer was fixed before contextual annotation was inspected. Forty-two of the 46 refit-stable exact Pf-Pb-Pk triplets were independently supported by a shared OrthoMCL 6.21 group (Fig. 5a and Supplementary Fig. S8). Compared with 46 exact E/E/E triplets paired one-to-one for mean protein length, directional triplets had lower mean pairwise global amino-acid identity (median 0.611 versus 0.747; two-sided Mann-Whitney P = 0.016; Benjamini-Hochberg FDR = 0.049; Fig. 5b and Supplementary Table S9). The comparison places stable directionality in comparative sequence space; it does not redefine their call patterns or rank them by divergence.

Three Gene Ontology terms passed the prespecified multiple-testing threshold: cytoplasmic dynein complex, dynein intermediate-chain binding and microtubule (two groups for each term; FDR = 0.048; Fig. 5c and Supplementary Table S10). The all-group matrix shows the species call states with external orthology and refit evidence for every directional triplet (Fig. 5d). Release-to-release mapping coverage was insufficient for parallel formal domain, localisation and life-cycle enrichment tests; those fields remain as record-level context in Supplementary Tables S5-S8.

"""
    marker = "### Held-species essentiality calls recapitulate target enrichment"
    if marker not in text:
        raise ValueError("Results insertion marker missing from Manuscript.md")
    text = text.replace(marker, section + marker, 1)
    old_legend = "**Fig. 6 | Frozen held-species essentiality predictions recapitulate known-target enrichment.**"
    new_legend = """**Fig. 5 | Refit-stable directional orthogroups show comparative sequence and functional context.** **a**, Independent OrthoMCL 6.21 support for the 46 refit-stable exact Pf-Pb-Pk triplets; support requires all three model genes to occur in a shared external group. **b**, Mean pairwise global amino-acid identity for all 46 directional triplets and 46 exact E/E/E triplets paired one-to-one for mean protein length. Violin envelopes show distributions, points show individual triplets and horizontal lines show medians; the two-sided Mann-Whitney P value was adjusted across three prespecified comparisons by Benjamini-Hochberg correction. **c**, GO terms with FDR <= 0.05 in the 46 directional groups relative to 4,183 exact triplets; each term annotates two directional groups. **d**, All 46 groups ordered by call pattern and sequence identity. In Pf/Pb/Pk call tracks, dark indicates predicted essential and pale indicates predicted non-essential; the OrthoMCL track shows external support and the refit track encodes reference-pattern recovery frequency across 100 refits. These downstream comparative annotations do not change a gene-level essentiality score or atlas call. Source data are provided with this paper.

**Fig. 6 | Frozen held-species essentiality predictions recapitulate known-target enrichment.**"""
    if old_legend not in text:
        raise ValueError("Expected held-species figure legend marker missing")
    text = text.replace(old_legend, new_legend, 1)
    (OUT / "Manuscript.md").write_text(text, encoding="utf-8")


def update_supporting_files() -> None:
    (OUT / "README_submission_package.md").write_text(
        "# Nature Communications submission package v57\n\n"
        "This package preserves the frozen five-branch matched-species atlas (8,735 calls across 15,591 malaria-parasite genes), the frozen held-species essentiality-to-target concordance analysis, and adds a main-text comparative-context figure for the 46 refit-stable directional orthogroups. No classifier was retrained, no probability was re-scored and no operating threshold was reselected.\n\n"
        "Figure 5 reports independent OrthoMCL support, length-matched sequence identity, multiple-testing-controlled GO signals and group-level refit evidence for the directional set. Figure 6 evaluates all labelled held-species genes for Pf, Pb, Pk and Tg using source-validation thresholds fixed before target evaluation. Pf, Pb and Pk target annotations are applied only after frozen scores are generated.\n",
        encoding="utf-8",
    )
    supp = OUT / "Supplementary_Information.md"
    supp_text = supp.read_text(encoding="utf-8")
    if "Main-text Fig. 5" not in supp_text:
        supp_text += (
            "\n\n### Main-text directional-context display\n\n"
            "Main-text Fig. 5 integrates the previously supplementary external OrthoMCL audit, length-matched sequence comparison and BH-controlled GO results for the 46 refit-stable directional triplets. Supplementary Figs. S8, S11 and S12 retain the complete source-level detail; domain, localisation and life-cycle records remain in Supplementary Tables S5-S8 because release-to-release mapping coverage did not meet the prespecified enrichment threshold.\n"
        )
        supp.write_text(supp_text, encoding="utf-8")
    # Ensure the public reviewer-code directory contains the exact new figure/package scripts.
    reviewer = OUT / "Reviewer_Code"
    reviewer.mkdir(exist_ok=True)
    for name in [
        "make_v57_directional_context_figure.py",
        "build_v57_directional_context_submission.py",
        "run_v56_loso_full_workflow_validation.py",
        "make_v56_loso_full_workflow_figure.py",
    ]:
        src = ROOT / name
        if src.exists():
            shutil.copy2(src, reviewer / name)


def zip_tree(zip_path: Path, roots: list[Path]) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for root in roots:
            if root.is_file():
                archive.write(root, root.name)
            else:
                for path in sorted(root.rglob("*")):
                    if path.is_file() and not path.name.endswith(".zip") and "_render" not in path.parts and "_qc" not in path.parts:
                        archive.write(path, str(path.relative_to(OUT)))


def create_upload_and_archives(doc_path: Path) -> None:
    UPLOAD.mkdir(exist_ok=True)
    source_zip = UPLOAD / "08_Source_Data_v57.zip"
    code_zip = UPLOAD / "09_Reviewer_Code_v57.zip"
    zip_tree(source_zip, [SOURCE])
    zip_tree(code_zip, [OUT / "Reviewer_Code"])

    copy_map = {
        "01_Main_manuscript_with_figures.docx": doc_path,
        "02_Cover_letter.docx": OUT / "Cover_letter.docx",
        "03_Supplementary_Information.docx": OUT / "Supplementary_Information.docx",
        "04_Supplementary_Tables.docx": OUT / "Supplementary_Tables.docx",
        "05_Supplementary_Data_S1.xlsx": OUT / "Supplementary_Data_S1.xlsx",
        "06_Supplementary_Data_S2_essentiality_atlas.xlsx": OUT / "Supplementary_Data_S2_Leakage_Safe_Essentiality_Atlas.xlsx",
        "07_Supplementary_Data_S3_post_score_evidence_context.xlsx": OUT / "Supplementary_Data_S3_Post_Score_Evidence_Context.xlsx",
        "10_Reporting_Summary.docx": OUT / "Nature_Portfolio_Reporting_Summary_Completed.docx",
        "11_ML_Checklist.docx": OUT / "Machine_Learning_Checklist_Completed.docx",
    }
    for dest_name, src in copy_map.items():
        shutil.copy2(src, UPLOAD / dest_name)
    for number, basename in [
        (12, "Fig1_study_design.tiff"),
        (13, "Fig2_species_matched_performance.tiff"),
        (14, "Fig3_target_label_naive_transfer_controls.tiff"),
        (15, "Fig4_atlas_and_directionality.tiff"),
        (16, "Fig5_refit_stable_directional_comparative_context.tiff"),
        (17, "Fig6_LOSO_full_workflow_target_validation.tiff"),
    ]:
        shutil.copy2(FIG / basename, UPLOAD / f"{number:02d}_{basename}")
    (UPLOAD / "UPLOAD_MANIFEST.md").write_text(
        "# Upload manifest: Nature Communications v57\n\n"
        "1. `01_Main_manuscript_with_figures.docx` - integrated manuscript with six embedded main figures.\n"
        "2. `02_Cover_letter.docx` - cover letter.\n"
        "3. `03_Supplementary_Information.docx` - supplementary methods and figure legends.\n"
        "4. `04_Supplementary_Tables.docx` - supplementary tables.\n"
        "5. `05_Supplementary_Data_S1.xlsx` - record-level provenance.\n"
        "6. `06_Supplementary_Data_S2_essentiality_atlas.xlsx` - frozen matched-species atlas.\n"
        "7. `07_Supplementary_Data_S3_post_score_evidence_context.xlsx` - post-score evidence context.\n"
        "8. `08_Source_Data_v57.zip` - source data for Figures 1-6, including Fig. 5 directional context and Fig. 6 frozen LOSO analysis.\n"
        "9. `09_Reviewer_Code_v57.zip` - analysis and packaging code.\n"
        "10. `10_Reporting_Summary.docx` - Nature Portfolio reporting summary.\n"
        "11. `11_ML_Checklist.docx` - machine-learning checklist.\n"
        "12-17. High-resolution TIFFs for Figures 1-6.\n",
        encoding="utf-8",
    )

    pdf = OUT / "Main_manuscript_with_all_figures_v57.pdf"
    roots = [
        doc_path,
        pdf,
        OUT / "Cover_letter.docx",
        OUT / "Supplementary_Information.docx",
        OUT / "Supplementary_Tables.docx",
        OUT / "Supplementary_Data_S1.xlsx",
        OUT / "Supplementary_Data_S2_Leakage_Safe_Essentiality_Atlas.xlsx",
        OUT / "Supplementary_Data_S3_Post_Score_Evidence_Context.xlsx",
        OUT / "Main_Figures",
        OUT / "Supplementary_Figures",
        OUT / "Source_Data",
        OUT / "Reviewer_Code",
        UPLOAD,
        OUT / "README_submission_package.md",
        OUT / "Author_Metadata_for_Submission.md",
    ]
    zip_tree(OUT / "ncomms_submission_v57_directional_context_fig6.zip", roots)


def main() -> None:
    stage_v56()
    update_markdown()
    update_supporting_files()
    doc_path = update_docx()
    print(doc_path)


if __name__ == "__main__":
    main()
