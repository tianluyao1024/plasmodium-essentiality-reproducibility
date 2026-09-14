from pathlib import Path
from copy import deepcopy
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

p=Path('D:/ProtoEss_NatureCommunications_v61_supervisor/Upload_Files/01_Main_manuscript_with_figures.docx')
d=Document(p)
for math in d._element.xpath('.//m:oMath')[14:16]:
    for run in list(math.iter(qn('m:r'))):
        t=run.find(qn('m:t'))
        if t is not None and t.text and 'OR' in t.text:
            parent=run.getparent()
            idx=parent.index(run)
            for j,char in enumerate(t.text):
                r=OxmlElement('m:r')
                rp=OxmlElement('m:rPr')
                normal=OxmlElement('m:nor')
                rp.append(normal)
                r.append(rp)
                txt=OxmlElement('m:t')
                txt.text=char
                r.append(txt)
                parent.insert(idx+j,r)
            parent.remove(run)
# Remove inherited decorative title rule from styles as well.
for style in d.styles:
    for border in list(style.element.iter(qn('w:pBdr'))):
        border.getparent().remove(border)
d.save(p)
