"""Build the v58 Nature Communications package after final reviewer polishing.

The build reuses frozen v57 results.  It removes duplicated directional-context
reporting from the Fig. 4 Results section, clarifies the Fig. 5/6 legends, and
creates an integrated Supplementary Information document containing S1-S12.
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


ROOT = Path(__file__).resolve().parent
V57 = ROOT / "protozoa_human_sci_analysis" / "reviewer_revision" / "ncomms_submission_v57_directional_context_fig6"
OUT = ROOT / "protozoa_human_sci_analysis" / "reviewer_revision" / "ncomms_submission_v58_final_polish"
FIG = OUT / "Main_Figures"
SUPP_FIG = OUT / "Supplementary_Figures"
SOURCE = OUT / "Source_Data"
UPLOAD = OUT / "Upload_Files_v58"


def ignored(_: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name.startswith("_render") or name.startswith("_qc") or name.endswith(".zip")
    }


def stage_v57() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(V57, OUT, ignore=ignored)
    for stale in [
        OUT / "Main_manuscript_with_all_figures_v57.docx",
        OUT / "Main_manuscript_with_all_figures_v57.pdf",
        OUT / "Manuscript.docx",
        OUT / "ncomms_submission_v57_directional_context_fig6.zip",
    ]:
        if stale.exists():
            stale.unlink()
    shutil.rmtree(OUT / "Upload_Files_v57", ignore_errors=True)


def replace_text(paragraph: Paragraph, text: str) -> None:
    paragraph.clear()
    paragraph.add_run(text)


def find_prefix(doc: Document, prefix: str) -> Paragraph:
    for paragraph in doc.paragraphs:
        if paragraph.text.strip().startswith(prefix):
            return paragraph
    raise ValueError(f"Could not find paragraph beginning: {prefix}")


def replace_prefix(doc: Document, prefix: str, text: str) -> Paragraph:
    paragraph = find_prefix(doc, prefix)
    replace_text(paragraph, text)
    return paragraph


def replace_figure_before_caption(doc: Document, caption: Paragraph, image_path: Path) -> None:
    """Replace the immediately preceding embedded figure while preserving position."""
    previous = caption._p.getprevious()
    if previous is not None:
        previous.getparent().remove(previous)
    doc.add_picture(str(image_path), width=None)
    picture = doc.paragraphs[-1]
    picture.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption._p.addprevious(picture._p)


FIG5_CAPTION = (
    "Fig. 5 | Refit-stable directional orthogroups show comparative sequence and functional context. "
    "a, Independent OrthoMCL 6.21 support for the 46 refit-stable exact Pf-Pb-Pk triplets; support requires all three model genes to occur in a shared external group. "
    "b, Mean pairwise global amino-acid identity for all 46 directional triplets and 46 exact E/E/E triplets paired one-to-one for mean protein length. Violin envelopes show distributions, points show individual triplets and horizontal lines show medians; the two-sided Mann-Whitney P value was adjusted across three prespecified comparisons by Benjamini-Hochberg correction. "
    "c, GO terms with FDR <= 0.05 in the 46 directional groups relative to 4,183 exact triplets; each term annotates two directional groups. "
    "d, All 46 groups ordered by call pattern and sequence identity. In the Pf/Pb/Pk tracks, dark cells denote predicted-essential calls and pale cells predicted-non-essential calls. Green denotes independent OrthoMCL support; blue intensity denotes reference-pattern recovery frequency across 100 refits. Source data are provided with this paper."
)


FIG6_CAPTION = (
    "Fig. 6 | Frozen held-species essentiality predictions recapitulate known-target enrichment. "
    "a, Binary essentiality performance on all labelled genes of each held species; this panel provides the held-species prediction context for the target-enrichment analyses in b-d. The operating threshold was selected from source-species validation data before held-species evaluation. "
    "b, Haldane-Anscombe-corrected odds ratios for known-target enrichment in observed-essential and LOSO-predicted-essential sets. Points and horizontal bars show odds ratios and 95% confidence intervals; the dashed line denotes no enrichment. ‘Direct’ denotes mapped direct Pf targets; ‘supported’ denotes the full target-supported registry. "
    "c, Fraction of mapped known targets classified as essential by observed labels or frozen LOSO calls. "
    "d, Cumulative recovery of registry-supported genes at every 10% increment of the frozen LOSO ranking. Target annotations were applied after score generation and were not used for fitting, label-derived feature construction, preprocessing or threshold selection. Tg is included in a as a distant held-out essentiality-transfer audit; it is not included in b-d because no harmonised tiered Tg target registry was prespecified for this comparison."
)


def revise_results_markdown(text: str) -> str:
    old_open = (
        "We then resolved 4,183 exact one-to-one-to-one Pf-Pb-Pk orthogroups using reciprocal best-hit MMseqs2 components. "
        "An independent OrthoMCL 6.21 analysis assigned 3,804 of the 4,183 exact triplets to a shared OrthoMCL group; 42 of 46 refit-stable directional groups received the same external support (Fig. 4e; Supplementary Fig. S8). "
        "This analysis supports the orthology context while preserving PlasmoDB-68/MMseqs2 triplets as the primary analysis units; it is not used to redefine membership in the refit-stable subset. "
    )
    new_open = (
        "We then resolved 4,183 exact one-to-one-to-one Pf-Pb-Pk orthogroups using reciprocal best-hit MMseqs2 components. "
    )
    if old_open not in text:
        raise ValueError("Expected directional-results opening was not found")
    text = text.replace(old_open, new_open, 1)
    old_context = (
        "The refit-stable subset refines, rather than replaces, the 535 full-model and 328 annotation-retained directional sets: it prioritises high-margin discordant triplets that persist after annotation exclusion and repeated refitting. "
        "The Pk directional retention audit is shown in Supplementary Fig. S9. Post hoc GO analysis of the 46 stable groups against the 4,183-triplet background identified cytoplasmic dynein complex, dynein intermediate-chain binding and microtubule annotations (two groups each; Benjamini-Hochberg FDR = 0.048; Supplementary Fig. S11). "
        "Relative to 46 exact E/E/E triplets matched one-to-one by mean protein length, the directional groups had lower triplet-wide global amino-acid identity (median 0.611 versus 0.747; two-sided Mann-Whitney P = 0.016, Benjamini-Hochberg-adjusted P = 0.049; Supplementary Fig. S12 and Supplementary Table S9). "
        "Broad OrthoFinder context was available for 40 directional and 43 control groups. Its distribution differed after multiple-testing correction (adjusted P = 0.050), although both group medians were zero; it is therefore retained as evolutionary context rather than evidence that paralogy explains the directional calls. "
        "A release-to-release mapping audit provided too few high-confidence UniProt transfers for formal domain, localisation or harmonised life-cycle enrichment. Those member-level fields and their denominators are reported explicitly in Supplementary Tables S5-S11, preventing missing annotation from being treated as a biological absence. "
        "Together, these layers focus matched-stage perturbation experiments on a small, robustness-selected set of candidate directional dependencies."
    )
    new_context = (
        "The refit-stable subset provides a focused layer within the 535 full-model and 328 annotation-exclusion-retained directional sets. "
        "The Pk directional retention audit is shown in Supplementary Fig. S9; comparative sequence, orthology and annotation context for the 46 refit-stable groups is presented in the following section."
    )
    if old_context not in text:
        raise ValueError("Expected repeated directional-context paragraph was not found")
    text = text.replace(old_context, new_context, 1)
    text = text.replace(
        "The comparison places stable directionality in comparative sequence space; it does not redefine their call patterns or rank them by divergence.",
        "The comparison places the stable directional calls in comparative sequence space independently of the score-based membership definition.",
        1,
    )
    text = text.replace(
        "Release-to-release mapping coverage was insufficient for parallel formal domain, localisation and life-cycle enrichment tests; those fields remain as record-level context in Supplementary Tables S5-S8.",
        "Release-to-release mapping coverage did not support parallel formal enrichment tests of domain, localisation or life-cycle fields. These annotations are reported as record-level context in Supplementary Tables S5-S8.",
        1,
    )
    text = text.replace("**Fig. 4 | Genome-scale atlas resolves shared and directional predicted-essentiality patterns.** **a**, Matched-species predicted-essential calls across 15,591 malaria-parasite genes. **b**, Quantitative filtering of exact Pf-Pb-Pk triplets. **c**, High-margin directional call patterns. **d**, Reference-pattern recovery across 100 class-stratified refits; the dashed line is the 80% retention criterion. **e**, Independent OrthoMCL support. Directional groups are hypotheses for experimental follow-up.",
                        "**Fig. 4 | Genome-scale atlas resolves shared and directional predicted-essentiality patterns.** **a**, Matched-species predicted-essential calls across 15,591 malaria-parasite genes. **b**, Quantitative filtering of exact Pf-Pb-Pk triplets. **c**, High-margin directional call patterns. **d**, Reference-pattern recovery across 100 class-stratified refits; the dashed line is the 80% retention criterion. Directional groups are hypotheses for experimental follow-up.", 1)
    text = text.replace("**Fig. 5 | Refit-stable directional orthogroups show comparative sequence and functional context.**", f"**{FIG5_CAPTION}**", 1)
    # The replacement above wraps the full legend; discard the former continuation.
    start = text.find(f"**{FIG5_CAPTION}**")
    if start >= 0:
        next_legend = text.find("**Fig. 6 |", start)
        if next_legend < 0:
            raise ValueError("Figure 6 legend marker was not found")
        text = text[:start] + FIG5_CAPTION + "\n\n" + text[next_legend:]
    old_fig6_start = "**Fig. 6 | Frozen held-species essentiality predictions recapitulate known-target enrichment.**"
    start = text.find(old_fig6_start)
    if start < 0:
        raise ValueError("Figure 6 legend marker was not found")
    text = text[:start] + FIG6_CAPTION + "\n"
    text = text.replace("### Pharmacological concordance supports interpretation after scoring", "**Pharmacological context after species-matched scoring.**", 1)
    return text


def update_manuscript() -> Path:
    md_path = OUT / "Manuscript.md"
    md_path.write_text(revise_results_markdown(md_path.read_text(encoding="utf-8")), encoding="utf-8")

    # The staged tree intentionally excludes versioned manuscript artefacts.
    # Read the frozen integrated v57 source directly, then write a new v58 copy.
    doc = Document(V57 / "Main_manuscript_with_all_figures_v57.docx")
    replace_prefix(
        doc,
        "We then resolved 4,183 exact one-to-one-to-one Pf-Pb-Pk orthogroups",
        "We then resolved 4,183 exact one-to-one-to-one Pf-Pb-Pk orthogroups using reciprocal best-hit MMseqs2 components. A strong directional pattern required discordant matched-species binary calls and an absolute threshold-centred margin of at least 0.10 for every member. This yielded 535 exact orthogroups with high-margin discordant predicted-essentiality patterns under the five-branch stack. Random re-pairing of species-specific margins produced a mean of 2,211.8 high-margin discordant triplets, compared with 535 among the exact orthogroups. This negative control shows that exact orthology constrains the directionality definition; it does not establish a biological mechanism. Although concordant patterns predominated, the 46 retained groups displayed reproducible directional patterns after increasingly stringent controls. Annotation-excluded refitting retained 328 of these groups with the same directional pattern, and 46 groups recovered the reference pattern in at least 80% of 100 class-stratified refits (Fig. 4b,d). Across the refit analysis, the median reference-pattern recovery frequency was 0.45; the 46-group subset is therefore an annotation-independent, refit-stable hypothesis set rather than an unbiased census of directional essentiality. We report the wider 535- and 328-group sets alongside it. Membership is defined by threshold pattern and margin; the normalised directional-difference score (N-DDS) is used only as a scale-aware display coordinate.",
    )
    replace_prefix(
        doc,
        "The refit-stable subset refines, rather than replaces",
        "The refit-stable subset provides a focused layer within the 535 full-model and 328 annotation-exclusion-retained directional sets. The Pk directional retention audit is shown in Supplementary Fig. S9; comparative sequence, orthology and annotation context for the 46 refit-stable groups is presented in the following section.",
    )
    replace_prefix(
        doc,
        "The directional layer was defined before contextual annotation",
        "The directional layer was defined before contextual annotation was inspected. Forty-two of the 46 refit-stable exact Pf-Pb-Pk triplets were independently supported by a shared OrthoMCL 6.21 group (Fig. 5a and Supplementary Fig. S8). Compared with 46 exact E/E/E triplets matched one-to-one for mean protein length, directional triplets had lower mean pairwise global amino-acid identity (median 0.611 versus 0.747; two-sided Mann-Whitney P = 0.016; Benjamini-Hochberg FDR = 0.049; Fig. 5b and Supplementary Table S9). The comparison places the stable directional calls in comparative sequence space independently of the score-based membership definition.",
    )
    replace_prefix(
        doc,
        "Three Gene Ontology terms passed the prespecified",
        "Three Gene Ontology terms passed the prespecified multiple-testing threshold: cytoplasmic dynein complex, dynein intermediate-chain binding and microtubule (two groups for each term; FDR = 0.048; Fig. 5c and Supplementary Table S10). The all-group matrix shows the species call states with external orthology and refit evidence for every directional triplet (Fig. 5d). Release-to-release mapping coverage did not support parallel formal enrichment tests of domain, localisation or life-cycle fields. These annotations are reported as record-level context in Supplementary Tables S5-S8.",
    )
    replace_prefix(doc, "Fig. 4 | Genome-scale atlas", "Fig. 4 | Genome-scale atlas resolves shared and directional predicted-essentiality patterns. a, Matched-species predicted-essential calls across 15,591 malaria-parasite genes. b, Quantitative filtering of exact Pf-Pb-Pk triplets. c, High-margin directional call patterns. d, Reference-pattern recovery across 100 class-stratified refits; the dashed line is the 80% retention criterion. Directional groups are hypotheses for experimental follow-up.")
    fig5_caption = replace_prefix(doc, "Fig. 5 | Refit-stable directional", FIG5_CAPTION)
    replace_figure_before_caption(doc, fig5_caption, FIG / "Fig5_refit_stable_directional_comparative_context.png")
    replace_prefix(doc, "Fig. 6 | Frozen held-species", FIG6_CAPTION)
    pharmacology_heading = find_prefix(doc, "Pharmacological concordance supports interpretation after scoring")
    replace_text(pharmacology_heading, "Pharmacological context after species-matched scoring.")
    pharmacology_heading.style = doc.styles["Normal"]
    if pharmacology_heading.runs:
        pharmacology_heading.runs[0].bold = True

    out = OUT / "Main_manuscript_with_all_figures_v58.docx"
    doc.save(out)
    shutil.copy2(out, OUT / "Manuscript.docx")
    return out


def update_supplementary_markdown() -> None:
    path = OUT / "Supplementary_Information.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace("Supplementary Figs. S1-S11", "Supplementary Figs. S1-S12")
    text = text.replace("## Supplementary Fig. S11 | Post hoc biological interpretation of refit-stable directional groups", "## Supplementary Fig. S11 | Supporting Gene Ontology audit for refit-stable directional groups")
    text = text.replace("## Supplementary Fig. S12 | Coverage-audited biological context of refit-stable directional orthogroups", "## Supplementary Fig. S12 | Supporting diagnostics for the matched sequence comparison")
    text = text.replace(
        "Mean triplet-wide global amino-acid identity is lower in the 46 refit-stable directional groups than in 46 length-matched E/E/E controls (median 0.611 versus 0.747; two-sided Mann-Whitney P = 0.016; Benjamini-Hochberg-adjusted P = 0.049). Independent OrthoFinder paralog context differs in distribution after correction, although both medians are zero and the available mappings are reported rather than treated as a mechanism. The protein-length panel confirms the matching design.",
        "This figure provides the matching and mapping diagnostics for the sequence-identity comparison reported in Fig. 5b. It shows the individual identity values, OrthoFinder paralog context and protein-length balance for the 46 directional and 46 length-matched E/E/E triplets.",
    )
    text = text.replace("defined in the frozen v46 lineage", "defined in the frozen five-branch analysis")
    path.write_text(text, encoding="utf-8")


SUPPLEMENTARY_PNGS = {
    "Supplementary Fig. S1": "Supplementary_FigS1_feature_branch_SHAP.png",
    "Supplementary Fig. S2": "Supplementary_FigS2_essentiality_matched_pharmacology.png",
    "Supplementary Fig. S3": "Supplementary_FigS3_Tg_source_augmentation.png",
    "Supplementary Fig. S4": "Supplementary_FigS4_training_label_uncertainty.png",
    "Supplementary Fig. S5": "Supplementary_FigS5_pharmacology_audit.png",
    "Supplementary Fig. S6": "Supplementary_FigS6_directionality_definition_sensitivity.png",
    "Supplementary Fig. S7": "Supplementary_FigS7_unlabelled_atlas_audit.png",
    "Supplementary Fig. S8": "Supplementary_FigS8_independent_orthology_support.png",
    "Supplementary Fig. S9": "Supplementary_FigS9_Pk_directional_audit.png",
    "Supplementary Fig. S10": "Supplementary_FigS10_exact_triplet_eligibility_matched_null.png",
    "Supplementary Fig. S11": "Supplementary_FigS11_biological_interpretation.png",
    "Supplementary Fig. S12": "Supplementary_FigS12_refit_stable_biological_context.png",
}


def add_picture_after(doc: Document, paragraph: Paragraph, path: Path) -> None:
    # Keep every supplementary multi-panel figure within the Word text block;
    # natural image dimensions can otherwise clip the rightmost panel in PDF.
    doc.add_picture(str(path), width=Inches(6.65))
    picture_paragraph = doc.paragraphs[-1]
    picture_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph._p.addnext(picture_paragraph._p)


def update_supplementary_docx() -> Path:
    doc = Document(OUT / "Supplementary_Information.docx")
    replace_prefix(doc, "This file contains Supplementary Figs. S1-S11", "This file contains Supplementary Figs. S1-S12, their legends and the associated supporting notes. All downstream analyses use the frozen 8,735-call matched-species atlas, 4,183 exact Pf-Pb-Pk triplets and 46 refit-stable directional groups. None refits a classifier, changes a validation-selected threshold or changes an atlas call.")
    replace_prefix(doc, "Supplementary Fig. S11 | Post hoc", "Supplementary Fig. S11 | Supporting Gene Ontology audit for refit-stable directional groups")
    replace_prefix(doc, "Supplementary Fig. S12 | Coverage-audited", "Supplementary Fig. S12 | Supporting diagnostics for the matched sequence comparison")
    replace_prefix(doc, "a, Global amino-acid identity", "This figure provides the matching and mapping diagnostics for the sequence-identity comparison reported in Fig. 5b. It shows the individual identity values, OrthoFinder paralog context and protein-length balance for the 46 directional and 46 length-matched E/E/E triplets.")
    replace_prefix(doc, "This analysis interrogates the 46 refit-stable", "This analysis interrogates the 46 refit-stable directional exact Pf-Pb-Pk orthogroups defined in the frozen five-branch analysis. Candidate membership required agreement of the full and functional-annotation-excluded stacks and recovery of the reference call pattern in at least 80 of 100 class-stratified refits. No classifier was retrained for this note, and none of the analyses below changes a matched-species atlas call.")

    # Keep the final supporting diagnostic together with its legend rather than
    # stranding the heading at the foot of the preceding page.
    find_prefix(doc, "Supplementary Fig. S12").paragraph_format.page_break_before = True

    # Earlier supplementary drafts contained only a subset of embedded figures.
    # Remove those legacy drawings before inserting one current PNG per S1-S12.
    for paragraph in list(doc.paragraphs):
        if paragraph._p.xpath(".//w:drawing"):
            paragraph._element.getparent().remove(paragraph._element)

    for prefix, figure_name in SUPPLEMENTARY_PNGS.items():
        heading = find_prefix(doc, prefix)
        paragraphs = doc.paragraphs
        index = next(i for i, paragraph in enumerate(paragraphs) if paragraph._p is heading._p)
        following = next((paragraph for paragraph in paragraphs[index + 1 :] if paragraph.text.strip()), heading)
        add_picture_after(doc, following, SUPP_FIG / figure_name)

    out = OUT / "Supplementary_Information_v58_complete.docx"
    doc.save(out)
    shutil.copy2(out, OUT / "Supplementary_Information.docx")
    return out


def zip_tree(zip_path: Path, roots: list[Path]) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for root in roots:
            if root.is_file():
                archive.write(root, root.name)
                continue
            for path in sorted(root.rglob("*")):
                if path.is_file() and not path.name.endswith(".zip") and "_render" not in path.parts:
                    archive.write(path, str(path.relative_to(OUT)))


def package(main_doc: Path) -> None:
    reviewer = OUT / "Reviewer_Code"
    for name in ["make_v58_directional_context_figure.py", "build_v58_final_polish_submission.py"]:
        shutil.copy2(ROOT / name, reviewer / name)
    (OUT / "README_submission_package.md").write_text(
        "# Nature Communications submission package v58\n\n"
        "This final display-item revision preserves the frozen five-branch matched-species atlas (8,735 calls across 15,591 malaria-parasite genes), the frozen held-species essentiality-to-target concordance analysis, and the 46 refit-stable directional orthogroups. No classifier was retrained, no probability was re-scored and no operating threshold was reselected.\n\n"
        "Figure 4 reports atlas-scale calls and directional filtering. Figure 5 provides independent orthology, matched sequence and Gene Ontology context for the refit-stable directional set. Figure 6 provides held-species prediction context and target-label-naive pharmacological concordance. The Supplementary Information document contains all S1-S12 figures together with their legends and notes.\n",
        encoding="utf-8",
    )
    UPLOAD.mkdir(exist_ok=True)
    source_zip = UPLOAD / "08_Source_Data_v58.zip"
    code_zip = UPLOAD / "09_Reviewer_Code_v58.zip"
    zip_tree(source_zip, [SOURCE])
    zip_tree(code_zip, [reviewer])
    files = {
        "01_Main_manuscript_with_figures.docx": main_doc,
        "02_Cover_letter.docx": OUT / "Cover_letter.docx",
        "03_Supplementary_Information.docx": OUT / "Supplementary_Information.docx",
        "04_Supplementary_Tables.docx": OUT / "Supplementary_Tables.docx",
        "05_Supplementary_Data_S1.xlsx": OUT / "Supplementary_Data_S1.xlsx",
        "06_Supplementary_Data_S2_essentiality_atlas.xlsx": OUT / "Supplementary_Data_S2_Leakage_Safe_Essentiality_Atlas.xlsx",
        "07_Supplementary_Data_S3_post_score_evidence_context.xlsx": OUT / "Supplementary_Data_S3_Post_Score_Evidence_Context.xlsx",
        "10_Reporting_Summary.docx": OUT / "Nature_Portfolio_Reporting_Summary_Completed.docx",
        "11_ML_Checklist.docx": OUT / "Machine_Learning_Checklist_Completed.docx",
    }
    for target, source in files.items():
        shutil.copy2(source, UPLOAD / target)
    for number, name in enumerate([
        "Fig1_study_design.tiff", "Fig2_species_matched_performance.tiff", "Fig3_target_label_naive_transfer_controls.tiff",
        "Fig4_atlas_and_directionality.tiff", "Fig5_refit_stable_directional_comparative_context.tiff", "Fig6_LOSO_full_workflow_target_validation.tiff",
    ], start=12):
        shutil.copy2(FIG / name, UPLOAD / f"{number:02d}_{name}")
    (UPLOAD / "UPLOAD_MANIFEST.md").write_text(
        "# Upload manifest: Nature Communications v58\n\n"
        "Upload the integrated manuscript, Supplementary Information, Supplementary Tables, three workbooks, source-data and reviewer-code archives, reporting documents, and the six TIFF figure files. The Supplementary Information document contains S1-S12 and their legends.\n",
        encoding="utf-8",
    )
    pdf = OUT / "Main_manuscript_with_all_figures_v58.pdf"
    roots = [main_doc, pdf, OUT / "Cover_letter.docx", OUT / "Supplementary_Information.docx", OUT / "Supplementary_Tables.docx", OUT / "Supplementary_Data_S1.xlsx", OUT / "Supplementary_Data_S2_Leakage_Safe_Essentiality_Atlas.xlsx", OUT / "Supplementary_Data_S3_Post_Score_Evidence_Context.xlsx", FIG, SUPP_FIG, SOURCE, reviewer, UPLOAD, OUT / "README_submission_package.md", OUT / "Author_Metadata_for_Submission.md"]
    zip_tree(OUT / "ncomms_submission_v58_final_polish.zip", roots)


def main() -> None:
    stage_v57()
    # Regenerate the reviewer-polished Figure 5 after staging so it is retained
    # in the v58 package rather than the inherited v57 rendering.
    from make_v58_directional_context_figure import main as make_figure5

    make_figure5()
    update_supplementary_markdown()
    main_doc = update_manuscript()
    update_supplementary_docx()
    package(main_doc)
    print(main_doc)


if __name__ == "__main__":
    main()
