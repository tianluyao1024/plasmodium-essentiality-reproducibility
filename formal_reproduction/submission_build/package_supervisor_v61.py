from pathlib import Path
import zipfile, shutil, hashlib, json

out=Path('D:/ProtoEss_NatureCommunications_v61_supervisor')
preview=out/'PDF_previews'
preview.mkdir(exist_ok=True)
shutil.copy2('D:/ProtoEss_v61_QA/main_final/01_Main_manuscript_with_figures.pdf',preview/'Main_manuscript.pdf')
shutil.copy2('D:/ProtoEss_v61_QA/cover/02_Cover_letter.pdf',preview/'Cover_letter.pdf')
with zipfile.ZipFile(out/'Upload_Files/09_Reviewer_Code.zip','a',zipfile.ZIP_DEFLATED) as z:
    z.write(Path(__file__).with_name('repair_v61_math.py'),'submission_build/repair_v61_math.py')
    z.write(__file__,'submission_build/package_supervisor_v61.py')
checks={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.rglob('*')) if f.is_file() and f.name!='SHA256SUMS.json'}
(out/'SHA256SUMS.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
dest=out.with_suffix('.zip')
with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
    for f in sorted(out.rglob('*')):
        if f.is_file():
            z.write(f,f.relative_to(out.parent))
with zipfile.ZipFile(dest) as z:
    assert z.testzip() is None
print(dest)
print(f'{dest.stat().st_size/1024/1024:.1f} MB')
