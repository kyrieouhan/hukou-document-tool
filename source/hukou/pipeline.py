"""Sequential, cancellable PDF processing with local-only OCR."""
from pathlib import Path
from dataclasses import asdict
import time,json,hashlib,logging
import re
import fitz
import cv2
import numpy as np
from .config import Settings
from .ocr import RapidEngine,OcrEngine,Token
from .imaging import orient,deskew,detect_regions,classify,resize,rotate,recover_cover_regions,NON_MEMBER_KINDS,region_tokens
from .parser import parse_cover,parse_member,resolve_household,cluster_households
from .candidates import FieldReader
from .models import Household,Issue
from .validators import validate
from .identity_evidence import extract_identities,corroborate_damaged_members

LOG=logging.getLogger('hukou')

class Cancelled(Exception): pass

def file_hash(path: Path) -> str:
    sha=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): sha.update(block)
    return sha.hexdigest()

def _page_fallback_region(image: np.ndarray,width: int=1500):
    """Whole-page stand-in region when contours miss a clearly printed card."""
    from .imaging import Region,deskew
    oriented,_angle=deskew(image)
    h,w=oriented.shape[:2]
    scale=min(1.,width/w)
    crop=cv2.resize(oriented,None,fx=scale,fy=scale) if scale<1 else oriented.copy()
    ch,cw=crop.shape[:2]
    table_h=int(ch*.78);top=int(ch*.10)
    quad=[[0.,0.],[float(cw-1),0.],[float(cw-1),float(ch-1)],[0.,float(ch-1)]]
    return Region(crop,quad,[0,0,w,h],table_h,table_top=top,extended_image=crop)

def _fallback_card_regions(image: np.ndarray,probe_tokens: list,width: int=1500):
    """Split a borderless page into card crops using text-row clusters."""
    from .imaging import Region
    if not probe_tokens:return []
    h,w=image.shape[:2]
    # probe_tokens were read from a resized view; map y back to source pixels.
    probe_h=max(max(p[1] for p in t.box) for t in probe_tokens) if probe_tokens else h
    # Cluster on source-scale y using each token's box in the probe image
    # scaled by the ratio between source height and probe content height.
    ordered=sorted(probe_tokens,key=lambda t:t.y)
    clusters=[[ordered[0]]]
    for t in ordered[1:]:
        if t.y-clusters[-1][-1].y>probe_h*.12:
            clusters.append([t])
        else:
            clusters[-1].append(t)
    # Map probe boxes to source coordinates via the whole-page bbox.
    # probe was resize(image,1900): scale=min(1,1900/max(h,w)).
    scale=min(1.,1900/max(h,w))
    inv=1./scale if scale>0 else 1.
    regions=[]
    for cluster in clusters:
        if len(cluster)<5:continue
        x1=min(min(p[0] for p in t.box) for t in cluster)*inv
        y1=min(min(p[1] for p in t.box) for t in cluster)*inv
        x2=max(max(p[0] for p in t.box) for t in cluster)*inv
        y2=max(max(p[1] for p in t.box) for t in cluster)*inv
        pad_x=(x2-x1)*.10+20; pad_y=(y2-y1)*.12+24
        x1=max(0,int(x1-pad_x)); y1=max(0,int(y1-pad_y))
        x2=min(w,int(x2+pad_x)); y2=min(h,int(y2+pad_y))
        if x2-x1<80 or y2-y1<60:continue
        crop=image[y1:y2,x1:x2].copy()
        k=min(1.,width/max(1,x2-x1))
        if k<1: crop=cv2.resize(crop,None,fx=k,fy=k)
        ch,cw=crop.shape[:2]
        table_h=int(ch*.80); top=int(ch*.08)
        quad=[[0.,0.],[float(cw-1),0.],[float(cw-1),float(ch-1)],[0.,float(ch-1)]]
        regions.append(Region(crop,quad,[x1,y1,x2-x1,y2-y1],table_h,table_top=top,extended_image=crop))
    return regions

class Pipeline:
    def __init__(self,settings: Settings|None=None,engine: OcrEngine|None=None):
        self.settings=settings or Settings()
        self.engine=engine or RapidEngine()
        self.field_reader=FieldReader(self.engine)

    def parse(self,path: Path,control=lambda:None,progress=lambda v:None,debug: Path|None=None) -> Household:
        """Never passes a filename to an OCR engine, image detector, or field parser."""
        start=time.perf_counter()
        failed=False
        h=Household(source_pdf=str(path.resolve()),source_hash=file_hash(path))
        try:
            with fitz.open(path) as doc:
                if doc.needs_pass: raise ValueError('PDF需要密码')
                if not len(doc): raise ValueError('PDF无页面')
                h.page_count=len(doc)
                for index,page in enumerate(doc):
                    control(); progress(dict(page=index+1,pages=len(doc),stage='页面方向与区域识别'))
                    # Avoid allocating arbitrarily large bitmaps from malformed PDF dimensions.
                    if page.rect.width*page.rect.height*(self.settings.dpi/72)**2>45_000_000:
                        h.issues.append(Issue('page','页面尺寸过大，未处理',page=index+1)); continue
                    pix=page.get_pixmap(dpi=self.settings.dpi,alpha=False,colorspace=fitz.csRGB)
                    image=cv2.cvtColor(np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,3),cv2.COLOR_RGB2BGR)
                    image,angle,scores=orient(image,self.engine,self.settings)
                    control()
                    image,skew=deskew(image)
                    ranked=sorted(scores,reverse=True)
                    if ranked[0]<5 or ranked[0]-ranked[1]<self.settings.orientation_margin:
                        h.issues.append(Issue('orientation','页面方向证据不足',page=index+1))
                    regions=detect_regions(image,self.settings.card_width)
                    if len(regions)==1:
                        regions[0].ocr_tokens=self.engine.read(regions[0].image)
                        single=classify(regions[0].ocr_tokens)
                        if single=='member_card':
                            regions.extend(recover_cover_regions(image,regions,self.engine,self.settings.card_width,('household_cover',)))
                        elif single=='household_cover':
                            regions.extend(recover_cover_regions(image,regions,self.engine,self.settings.card_width,('member_card',)))
                    page_info=dict(page=index+1,rotation=angle,scores=scores,deskew=round(skew,3),regions=[])
                    page_tokens=[]
                    context_kind=None
                    for ri,r in enumerate(regions):
                        control()
                        tokens=getattr(r,'ocr_tokens',None)
                        tokens=region_tokens(r.image,self.engine,tokens)
                        r.kind=classify(tokens)
                        region_rotation=0
                        if r.kind=='unknown':
                            reverse=rotate(r.image,180)
                            reverse_tokens=self.engine.read(reverse)
                            reverse_kind=classify(reverse_tokens)
                            if reverse_kind!='unknown':
                                r.table_top=r.image.shape[0]-r.table_height-r.table_top
                                r.image=reverse; tokens=reverse_tokens; r.kind=reverse_kind
                                # For reversed regions the extra right-side pixels
                                # would be on the left; do not misalign field seeds.
                                r.extended_image=None
                                region_rotation=180
                        if r.kind=='unknown':
                            if context_kind is None:context_kind=classify(self.engine.read(resize(image,1900)))
                            if context_kind in ('land_record','change_record'):
                                r.kind=context_kind
                            elif context_kind=='id_card':
                                bottom_id=any(re.fullmatch(r'[0-9]{17}[0-9Xx]',t.text.replace(' ','')) and
                                              t.y>r.image.shape[0]*.62 and t.confidence>=.85 for t in tokens)
                                if bottom_id:r.kind='id_card'
                        page_info['regions'].append(dict(index=ri,kind=r.kind,bbox=r.bbox,quad=r.quad))
                        page_info['regions'][-1]['rotation']=region_rotation
                        page_info['regions'][-1]['table_height']=r.table_height
                        page_info['regions'][-1]['table_top']=r.table_top
                        if context_kind:page_info['page_context_kind']=context_kind
                        page_tokens.append(tokens)
                        if r.kind=='member_card':
                            member,issues=parse_member(r,tokens,self.engine,index+1,ri,self.field_reader,debug/f'page_{index+1:03d}' if debug else None)
                            h.members.append(member)
                            h.issues.extend(Issue(key,reason,val,index+1,member.name) for key,reason,val in issues)
                        elif r.kind=='household_cover':
                            cover=parse_cover(r,tokens,self.engine,self.field_reader,index+1,ri,debug/f'page_{index+1:03d}' if debug else None)
                            cover['page']=index+1; h.covers.append(cover)
                            for key in ('household_no','householder','address'):
                                if not getattr(h,key) and cover.get(key): setattr(h,key,cover[key])
                        elif r.kind=='unknown':
                            h.issues.append(Issue('page','存在未能分类的文档区域，可能漏识别成员',page=index+1))
                        elif r.kind=='id_card':
                            h.identity_cards.extend(extract_identities(tokens,index+1,ri))
                    if not regions:
                        # ID-only pages can have no printed table; explicitly classify before ignoring.
                        non_table_tokens=self.engine.read(resize(image,1900));kind=classify(non_table_tokens)
                        if kind=='id_card':h.identity_cards.extend(extract_identities(non_table_tokens,index+1))
                        page_info['non_table_kind']=kind
                        if kind in ('member_card','household_cover'):
                            # Printed borders can be too faint for contours while
                            # the card text is still clearly readable. Split the
                            # page by text clusters so stacked cards stay separate.
                            fallbacks=_fallback_card_regions(image,non_table_tokens,self.settings.card_width)
                            if not fallbacks:
                                fallbacks=[_page_fallback_region(image,self.settings.card_width)]
                            for ri,fb in enumerate(fallbacks):
                                tokens=region_tokens(fb.image,self.engine)
                                fb.kind=classify(tokens);fb.field_tokens=tokens;fb.ocr_tokens=tokens
                                page_info['regions'].append(dict(index=ri,kind=fb.kind,bbox=fb.bbox,
                                                                 quad=fb.quad,rotation=0,
                                                                 table_height=fb.table_height,
                                                                 table_top=fb.table_top,fallback=True))
                                page_tokens.append(tokens)
                                regions.append(fb)
                                if fb.kind=='member_card':
                                    member,issues=parse_member(fb,tokens,self.engine,index+1,ri,self.field_reader,debug/f'page_{index+1:03d}' if debug else None)
                                    h.members.append(member)
                                    h.issues.extend(Issue(key,reason,val,index+1,member.name) for key,reason,val in issues)
                                elif fb.kind=='household_cover':
                                    cover=parse_cover(fb,tokens,self.engine,self.field_reader,index+1,ri,debug/f'page_{index+1:03d}' if debug else None)
                                    cover['page']=index+1;h.covers.append(cover)
                                    for key in ('household_no','householder','address'):
                                        if not getattr(h,key) and cover.get(key): setattr(h,key,cover[key])
                        elif kind not in NON_MEMBER_KINDS: h.issues.append(Issue('page','页面未找到可确认的文档区域',page=index+1))
                    only_supporting=(bool(regions) and all(r.kind in NON_MEMBER_KINDS for r in regions)) or (not regions and page_info.get('non_table_kind') in NON_MEMBER_KINDS)
                    if only_supporting:
                        # An independently classified ID/land document cannot add
                        # a household registration member. Keep its page evidence.
                        h.issues=[i for i in h.issues if not (i.field=='orientation' and i.page==index+1)]
                    h.pages.append(page_info)
                    if debug:
                        folder=debug/f'page_{index+1:03d}'
                        folder.mkdir(parents=True,exist_ok=True)
                        for ri,r in enumerate(regions):
                            cv2.imencode('.png',r.image)[1].tofile(folder/f'region_{ri}_{r.kind}.png')
                            (folder/f'tokens_{ri}.json').write_text(json.dumps([asdict(t) for t in page_tokens[ri]],ensure_ascii=False),encoding='utf-8')
                        (folder/'geometry.json').write_text(json.dumps(page_info,ensure_ascii=False),encoding='utf-8')
                    LOG.info('page=%d stage=complete regions=%d',index+1,len(regions))
        except Cancelled: raise
        except Exception as exc:
            failed=True
            # Exception messages may embed input paths or OCR text; keep public logs structural.
            LOG.error('PDF processing failed type=%s',type(exc).__name__)
            if debug:
                import traceback
                debug.mkdir(parents=True,exist_ok=True)
                (debug/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
            h.issues.append(Issue('page','PDF处理失败：'+type(exc).__name__))
        h.elapsed_seconds=round(time.perf_counter()-start,3)
        corroborate_damaged_members(h)
        cluster_households(h)
        resolve_household(h)
        validate(h,self.settings.confidence)
        if failed: h.status='error'
        return h
