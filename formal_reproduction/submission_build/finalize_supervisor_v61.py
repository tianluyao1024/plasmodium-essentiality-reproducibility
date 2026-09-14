from pathlib import Path
import zipfile, hashlib, json
from docx import Document

OUT=Path('D:/ProtoEss_NatureCommunications_v61_supervisor')
UP=OUT/'Upload_Files'
path=UP/'01_Main_manuscript_with_figures.docx'
d=Document(path)
shape=d.inline_shapes[5]
rid=shape._inline.graphic.graphicData.pic.blipFill.blip.embed
png=OUT/'Main_Figures_embedded_sources/Fig6_LOSO_full_workflow_target_validation.png'
d.part.related_parts[rid]._blob=png.read_bytes()
d.save(path)
(OUT/'Main_Figures_embedded_sources/Fig6.png').write_bytes(png.read_bytes())
with zipfile.ZipFile(UP/'09_Reviewer_Code.zip','a',zipfile.ZIP_DEFLATED) as z:
    z.write(Path(__file__).with_name('make_v60_3_fig6.py'),'submission_build/make_v60_3_fig6.py')
    z.write(__file__,'submission_build/finalize_supervisor_v61.py')
print('Updated Fig6 in manuscript and reviewer code')
