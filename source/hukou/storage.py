"""Atomic local state, collision-safe copy, and privacy-safe operational logging."""
import json,os,re,shutil,tempfile,logging
from pathlib import Path
from .pipeline import file_hash
from .models import Household

def atomic_json(path: Path,data: object) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            json.dump(data,stream,ensure_ascii=False,indent=2)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name): os.unlink(name)

def safe_stem(value: str) -> str:
    value=re.sub(r'[<>:"/\\|?*\x00-\x1f]','_',value).strip(' .')[:100]
    if not value or re.fullmatch(r'(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])',value): value='_'+value
    return value

def copy_pdf(h: Household,output: Path,naming: str='name') -> Path:
    """Commit a verified complete copy atomically; never expose a partial PDF."""
    source=Path(h.source_pdf)
    if file_hash(source)!=h.source_hash:
        raise OSError('输入文件在处理期间发生变化')
    reliable=h.rename_reliable and h.status=='ok'
    folder=output/('已识别PDF' if reliable else '待复核PDF')
    folder.mkdir(parents=True,exist_ok=True)
    base=safe_stem(h.householder if reliable else source.stem)
    if reliable and naming=='name_number': base+='_'+safe_stem(h.household_no)
    suffix=safe_stem(h.household_no) if reliable and h.household_no else h.source_hash[:10]
    for i in range(10000):
        stem=base if i==0 else f'{base}_{suffix}'+('' if i==1 else f'_{i}')
        target=folder/f'{stem}.pdf'
        if target.exists():
            if file_hash(target)==h.source_hash: return target
            continue
        temporary=None
        try:
            fd,temporary=tempfile.mkstemp(prefix='.hukou-copy-',suffix='.tmp',dir=folder)
            with os.fdopen(fd,'wb') as dst:
                with source.open('rb') as src: shutil.copyfileobj(src,dst)
                dst.flush(); os.fsync(dst.fileno())
            if file_hash(Path(temporary))!=h.source_hash or file_hash(source)!=h.source_hash:
                raise OSError('输入文件在处理期间发生变化')
            # Windows rename is atomic and fails if the destination exists.
            if os.name=='nt': os.rename(temporary,target)
            else: os.link(temporary,target)
            return target
        except FileExistsError: continue
        finally:
            if temporary: Path(temporary).unlink(missing_ok=True)
    raise OSError('输出同名文件过多')

class PrivacyFilter(logging.Filter):
    def filter(self,record: logging.LogRecord) -> bool:
        message=record.getMessage()
        record.msg=re.sub(r'(?<!\d)(\d{2})[\d\s-]{12,24}([\dXx]{4})(?!\d)',r'\1************\2',message)
        record.args=()
        return True
