from pathlib import Path
import re, shutil, zipfile, hashlib, json
from docx import Document
from docx.shared import Pt, RGBColor
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'protozoa_human_sci_analysis/reviewer_revision/ncomms_submission_v60_3_submission_polish'
OUT = Path('D:/ProtoEss_NatureCommunications_v61_supervisor')
UP = OUT / 'Upload_Files'
UP.mkdir(parents=True, exist_ok=True)
changes = []

def replace(p, old, new):
    if old not in p.text:
        return
    # Replace across run boundaries while preserving all unaffected runs and math.
    for _ in range(p.text.count(old)):
        start = p.text.index(old)
        end = start + len(old)
        pos = 0
        placed = False
        for r in p.runs:
            text = r.text
            a, b = pos, pos + len(text)
            pos = b
            if b <= start or a >= end:
                continue
            prefix = text[:max(0, start-a)]
            suffix = text[max(0, end-a):] if end < b else ''
            r.text = prefix + (new if not placed else '') + suffix
            placed = True
        if not placed:
            raise RuntimeError(old)
        changes.append([old,new])

d = Document('D:/Main_manuscript_with_all_figures_v60_2.docx')
pairs = [
    ('Eqs. (9)-(11)', 'Eqs. (11)-(13)'),
    ('Eqs. (9)\u2013(11)', 'Eqs. (11)\u2013(13)'),
    ('best-hit score in Eq. (12)', 'best-hit score in Eq. (14)'),
    ('In Eq. (12), q and t', 'In Eq. (14), q and t'),
    ('supported by  by', 'supported by'),
    ('supported by by', 'supported by'),
    ('context.The', 'context. The'),
    ('for this comparison for this comparison', 'for this comparison'),
    ('broad cross-species vulnerability', 'conserved predicted-essentiality patterns'),
    ('mechanistic follow-up', 'experimental follow-up'),
    ('resulting after scoring calls', 'resulting predicted-essential calls'),
    ('reported as after scoring concordance', 'reported as post-score concordance'),
    ('paired bootstrap intervals for the difference in log odds ratio are reported in Fig. 6b and Source Data', 'odds ratios and confidence intervals are shown in Fig. 6b; paired bootstrap intervals for the difference in log odds ratio are provided in Source Data'),
    ('Treating these analyses separately prevents an inflated claim of universal prediction and makes the transfer boundary itself a biological and methodological result.', 'Together, these analyses distinguish within-species inference from cross-species transfer and identify how source composition affects prediction in each recipient species.'),
]
for p in d.paragraphs:
    for a,b in pairs:
        replace(p,a,b)
    if p.text.startswith('Fig. 4 |'):
        replace(p,'Directional groups are hypotheses', 'e, Independent OrthoMCL support for 3,804 of 4,183 exact triplets and 42 of 46 refit-stable directional groups. Directional groups are hypotheses')
    if p.text.startswith('The evidence landscape was recipient- and analysis-specific.'):
        replace(p,p.text,'Directional candidates showed no additional known-target enrichment, indicating that comparative predicted-essentiality patterns and pharmacological concordance capture different aspects of the evidence landscape.')
    if p.style.name.startswith('Heading') or p.style.name == 'Title':
        for r in p.runs:
            r.font.color.rgb = RGBColor(0,0,0)
            r.font.underline = False
        pp = p._p.find(qn('w:pPr'))
        if pp is not None:
            for border in list(pp.findall(qn('w:pBdr'))):
                pp.remove(border)
    if p.text.startswith('Fig. '):
        p.paragraph_format.keep_together = True
        p.paragraph_format.keep_with_next = False
    if p._p.xpath('.//w:drawing'):
        p.paragraph_format.keep_with_next = True
main = UP / '01_Main_manuscript_with_figures.docx'
d.save(main)

c = Document('D:/Cover_letter.docx')
for p in c.paragraphs:
    for a,b in [
        ('shared biological vulnerabilities across malaria species', 'conserved predicted-essentiality patterns across malaria species'),
        ('lineage-specific genetic requirements', 'lineage-associated predicted-essentiality differences'),
        ('providing independent concordance', 'providing external-to-training concordance'),
        ('these results show that cross-species essentiality cannot be reduced to simple propagation from closely related homologues, nor can additional evolutionary diversity be assumed to improve prediction uniformly', 'these results demonstrate recipient-dependent effects of source composition and quantify performance relative to the tested homology baselines'),
    ]:
        replace(p,a,b)
    for r in p.runs:
        r.bold = False
c.save(UP / '02_Cover_letter.docx')

files = {
 'Supplementary_Information.docx':'03_Supplementary_Information.docx',
 'Supplementary_Tables.docx':'04_Supplementary_Tables.docx',
 'Supplementary_Data_S1.xlsx':'05_Supplementary_Data_S1.xlsx',
 'Supplementary_Data_S2_Leakage_Safe_Essentiality_Atlas.xlsx':'06_Supplementary_Data_S2_essentiality_atlas.xlsx',
 'Supplementary_Data_S3_Post_Score_Evidence_Context.xlsx':'07_Supplementary_Data_S3_post_score_evidence_context.xlsx',
 'Nature_Portfolio_Reporting_Summary_Completed.docx':'10_Reporting_Summary_Response_Companion.docx',
 'Machine_Learning_Checklist_Completed.docx':'11_Machine_Learning_Checklist.docx',
}
for src,dst in files.items():
    shutil.copy2(BASE/src,UP/dst)
for folder,dest in [('Source_Data','08_Source_Data.zip'),('Reviewer_Code','09_Reviewer_Code.zip')]:
    with zipfile.ZipFile(UP/dest,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted((BASE/folder).rglob('*')):
            if f.is_file() and '__pycache__' not in f.parts:
                z.write(f,f.relative_to(BASE/folder))
        if folder == 'Reviewer_Code':
            z.write(Path(__file__),'submission_build/build_supervisor_submission_v61.py')

# These are exactly the image bytes in the supervisor manuscript.
figs = OUT / 'Main_Figures_embedded_sources'
figs.mkdir(exist_ok=True)
for i,shape in enumerate(d.inline_shapes,1):
    rid = shape._inline.graphic.graphicData.pic.blipFill.blip.embed
    part = d.part.related_parts[rid]
    suffix = Path(str(part.partname)).suffix
    (figs/f'Fig{i}{suffix}').write_bytes(part.blob)

metadata = '\n\n'.join(p.text for p in d.paragraphs[:6] if p.text.strip())
(OUT/'Author_Metadata.md').write_text(metadata,encoding='utf-8')
(OUT/'Changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2),encoding='utf-8')
text = '\n'.join(p.text for p in d.paragraphs)
assert 'Eqs. (9)-(11)' not in text and 'best-hit score in Eq. (12)' not in text
assert len(d.inline_shapes)==6
math_count = len(d._element.xpath('.//m:oMath'))
assert math_count >= 16, math_count
qc = {'figures':6,'word_math_objects':math_count,'base':'Supervisor manuscript supplied on 2026-09-14','model_retraining':False,'atlas_calls':8735}
(OUT/'Package_QC.json').write_text(json.dumps(qc,indent=2),encoding='utf-8')
(OUT/'README.md').write_text('''# ProtoEss Nature Communications submission package v61

The main manuscript and cover letter use the supervisor-edited files as their base. Formula cross-references, the Fig. 4e legend, the Fig. 6b statistical cross-reference, duplicate text and selected interpretation wording have been corrected. Six embedded figures and native Word equations are retained. Frozen data and analysis code are inherited from v60.3; no models or scientific results were recalculated.

## Upload files

Upload 01 as the manuscript, 02 as the cover letter, 03-04 as supplementary information, 05-07 as supplementary data, 08 as source data, and 09 as reviewer code using the corresponding types available in the portal. Files 10-11 contain reporting/checklist responses. The six images extracted from the main document are retained separately for consistency checking; the main manuscript already contains these figures.

## Required author completion

The response-companion DOCX is not the official Nature Portfolio Reporting Summary. A verified completed official smart PDF was not available in the inherited package. Complete and save the official form before submitting it where requested: https://www.nature.com/documents/nr-reporting-summary.pdf . Do not upload this README or the companion as a substitute for that form.

Use Author_Metadata.md to enter all six authors and both corresponding authors consistently. Author declarations, funding, contributions and reporting-form responses require final author verification. This package has not been uploaded or submitted to the journal, GitHub or Zenodo.

The original supervisor files on D: have not been modified. Changes.json records text changes. PDF previews are generated from the revised DOCX for layout review.
''',encoding='utf-8')
print(OUT)
print(qc)
