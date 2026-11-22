"""Four-direction OCR, geometric table detection, and conservative enhancement."""
from dataclasses import dataclass
import math
import re
import cv2
import numpy as np
from .ocr import OcrEngine, Token
from .config import Settings

ANCHORS = {'常住人口登记卡':12,'家庭户口':9,'户主姓名':8,'户号':6,'居民身份证':7,
           '公民身份':5,'登记日期':5,'姓名':3,'住址':3,'户主关系':6,'出生日期':3,'中华人民共和国':6}
AUX_ANCHORS={'集体土地使用证':10,'土地使用者':8,'土地所有者':8,'使用权类型':6,'使用权面积':6,'建房用地通知书':10,'宗地图':6}
NON_MEMBER_KINDS={'id_card','land_record','change_record','booklet_instructions'}

def rotate(image: np.ndarray, degrees: int) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(image, (degrees//90)%4))

def resize(image: np.ndarray, side: int) -> np.ndarray:
    scale = min(1.,side/max(image.shape[:2]))
    return cv2.resize(image,None,fx=scale,fy=scale) if scale<1 else image

def orient(image: np.ndarray, engine: OcrEngine, settings: Settings) -> tuple[np.ndarray, int, list[float]]:
    """Choose rotation using text anchors before any field extraction."""
    small = resize(image,settings.orientation_side)
    scores=[];auxiliary=[]
    for angle in (0,90,180,270):
        tokens=engine.read(rotate(small,angle))
        score=0.;aux=0.
        for token in tokens:
            bw=max(p[0] for p in token.box)-min(p[0] for p in token.box)
            bh=max(p[1] for p in token.box)-min(p[1] for p in token.box)
            if bw<bh*1.3: continue
            clean=''.join(token.text.split())
            score+=sum(weight*token.confidence for anchor,weight in ANCHORS.items() if anchor in clean)
            aux+=sum(weight*token.confidence for anchor,weight in AUX_ANCHORS.items() if anchor in clean)
        scores.append(round(score,3))
        auxiliary.append(round(aux,3))
    if max(scores)<5 and max(auxiliary)>=6:scores=auxiliary
    best=int(np.argmax(scores))
    return rotate(image,best*90),best*90,scores

def enhance(image: np.ndarray, mode: str='original') -> np.ndarray:
    if mode=='original': return image
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    if mode=='clahe': gray=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(gray)
    elif mode=='sharpen': gray=cv2.addWeighted(gray,1.5,cv2.GaussianBlur(gray,(0,0),1),-.5,0)
    elif mode=='denoise': gray=cv2.fastNlMeansDenoising(gray,h=5)
    elif mode=='binary': gray=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,11)
    return cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)


def flatten_background(image: np.ndarray) -> np.ndarray:
    """Remove slow illumination changes without inventing or thresholding glyphs."""
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    smooth=cv2.GaussianBlur(gray,(0,0),1.2)
    background=cv2.GaussianBlur(smooth,(0,0),35)
    flat=cv2.divide(smooth,background,scale=200)
    low,high=np.percentile(flat,(1,95))
    contrast=np.uint8(np.clip((flat-low)*255/max(high-low,1),0,255))
    return cv2.cvtColor(contrast,cv2.COLOR_GRAY2BGR)


def region_tokens(image: np.ndarray,engine: OcrEngine,tokens: list[Token]|None=None) -> list[Token]:
    """Retry weak labels on locally normalized pixels; preserve original geometry."""
    tokens=engine.read(image) if tokens is None else tokens
    def anchors(ts):
        text='|'.join(t.text.replace(' ','') for t in ts)
        return sum(a in text for a in ANCHORS)
    count=anchors(tokens);kind=classify(tokens)
    if kind=='unknown' or (kind=='member_card' and count<5):
        alternate=engine.read(flatten_background(image))
        if anchors(alternate)>count:tokens=alternate
    return tokens

def deskew(image: np.ndarray) -> tuple[np.ndarray,float]:
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(gray,70,180)
    lines=cv2.HoughLinesP(edges,1,np.pi/1800,100,minLineLength=image.shape[1]*.18,maxLineGap=25)
    angles=[]
    for line in (lines.reshape(-1,4) if lines is not None else []):
        x1,y1,x2,y2=line
        a=math.degrees(math.atan2(y2-y1,x2-x1))
        if abs(a)<12: angles.append(a)
    angle=float(np.median(angles)) if angles else 0.
    if abs(angle)<.15: return image,0.
    h,w=image.shape[:2]
    matrix=cv2.getRotationMatrix2D((w/2,h/2),angle,1.)
    return cv2.warpAffine(image,matrix,(w,h),borderValue=(255,255,255)),angle

@dataclass
class Region:
    image: np.ndarray
    quad: list
    bbox: list
    table_height: int
    kind: str='unknown'
    table_top: int=0
    extended_image: np.ndarray|None=None

def order_quad(points: np.ndarray) -> np.ndarray:
    p=points.reshape(4,2).astype(np.float32)
    s=p.sum(axis=1); d=np.diff(p,axis=1).ravel()
    return np.array([p[s.argmin()],p[d.argmin()],p[s.argmax()],p[d.argmax()]],np.float32)

def detect_regions(image: np.ndarray, width: int=1500) -> list[Region]:
    """Detect separate table outlines; include title/footer around each rectified table."""
    h,w=image.shape[:2]
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    binary=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,41,13)
    # Close broken printed borders without joining vertically separated cards.
    mask=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    horizontal=cv2.morphologyEx(cv2.dilate(binary,np.ones((3,1),np.uint8)),cv2.MORPH_OPEN,np.ones((1,max(25,w//25)),np.uint8))
    vertical=cv2.morphologyEx(cv2.dilate(binary,np.ones((1,3),np.uint8)),cv2.MORPH_OPEN,np.ones((max(25,h//35),1),np.uint8))
    grid=cv2.morphologyEx(cv2.bitwise_or(horizontal,vertical),cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    grid_contours,_=cv2.findContours(grid,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contours=list(contours)+list(grid_contours)
    candidates=[]
    borderline=[]
    for c in contours:
        x,y,bw,bh=cv2.boundingRect(c)
        if bw<w*.23 or bh<h*.10 or bw*bh>w*h*.8: continue
        ar=bw/bh if bh else 0.
        area=cv2.contourArea(c)
        fill=area/(bw*bh) if bw*bh else 0.
        approx=cv2.approxPolyDP(c,.018*cv2.arcLength(c,True),True)
        quad=order_quad(approx if len(approx)==4 else cv2.boxPoints(cv2.minAreaRect(c)))
        if .9<ar<2.8 and fill>=.55:
            candidates.append((area,quad,(x,y,bw,bh)))
        elif w*.35<=bw and h*.07<=bh and 2.8<=ar<6.5 and fill>=.10:
            # Wide two-row header tables and border-stroke contours fail the
            # generic aspect/fill gate. Hold them for printed-grid evidence.
            borderline.append((area,quad,(x,y,bw,bh)))
    def grid_lines(quad):
        dst=np.array([[0,0],[799,0],[799,499],[0,499]],np.float32)
        patch=cv2.warpPerspective(gray,cv2.getPerspectiveTransform(quad,dst),(800,500),borderValue=255)
        ink=cv2.adaptiveThreshold(patch,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,12)
        horizontal=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,220),np.uint8))
        rows=np.where((horizontal>0).sum(axis=1)>250)[0]
        return 0 if not len(rows) else 1+int((np.diff(rows)>7).sum())
    # Promote only when the crop actually contains printed table rules.
    for area,quad,b in borderline:
        if grid_lines(quad)>=2:
            candidates.append((area,quad,b))
    candidates.sort(key=lambda v:v[0],reverse=True)
    grids=[grid_lines(c[1]) for c in candidates]
    # A photographed plastic frame is not the printed table: prefer its enclosed grid.
    filtered=[]
    for i,(area,quad,b) in enumerate(candidates):
        x,y,bw,bh=b
        inner=False
        for j,(other,q,(xx,yy,ww,hh)) in enumerate(candidates):
            if j==i or not .4<other/area<.94 or grids[j]<4: continue
            overlap=max(0,min(x+bw,xx+ww)-max(x,xx))*max(0,min(y+bh,yy+hh)-max(y,yy))/(ww*hh)
            if overlap>.96 and ww>bw*.7: inner=True; break
        if not inner: filtered.append((area,quad,b))
    candidates=filtered
    accepted=[]
    for area,quad,b in candidates:
        x,y,bw,bh=b
        if any(max(0,min(x+bw,xx+ww)-max(x,xx))*max(0,min(y+bh,yy+hh)-max(y,yy)) / (bw*bh)>.7 for _,_,(xx,yy,ww,hh) in accepted): continue
        accepted.append((area,quad,b))
    regions=[]
    for _,quad,bbox in sorted(accepted,key=lambda v:(v[2][1],v[2][0])):
        tw=(np.linalg.norm(quad[1]-quad[0])+np.linalg.norm(quad[2]-quad[3]))/2
        th=(np.linalg.norm(quad[3]-quad[0])+np.linalg.norm(quad[2]-quad[1]))/2
        table_h=int(width*th/tw)
        top=int(.11*table_h); bottom=int(.17*table_h)
        dst=np.array([[0,top],[width-1,top],[width-1,top+table_h],[0,top+table_h]],np.float32)
        matrix=cv2.getPerspectiveTransform(quad,dst)
        crop=cv2.warpPerspective(image,matrix,(width,top+table_h+bottom),borderValue=(255,255,255))
        # Text may overflow the printed right rule. Keep extra source pixels for
        # field OCR without changing direction/classification/detection geometry.
        extended=cv2.warpPerspective(image,matrix,(int(width*1.08),top+table_h+bottom),borderValue=(255,255,255))
        regions.append(Region(crop,quad.tolist(),list(bbox),table_h,table_top=top,extended_image=extended))
    return regions

def classify(tokens: list[Token]) -> str:
    # Separators prevent unrelated OCR boxes from synthesizing a false label.
    text='|'.join(t.text.replace(' ','') for t in tokens)
    if tokens:
        low=min(t.y for t in tokens); high=max(t.y for t in tokens)
        if any('常住人口' in t.text.replace(' ','') and t.y<low+.3*(high-low) for t in tokens): return 'member_card'
    if '出生' in text and ('文化程度' in text or '何时由何地' in text): return 'member_card'
    if '家庭户口' in text or ('户主姓名' in text and '户号' in text): return 'household_cover'
    if '户主姓名' in text and '住址' in text and ('家庭户' in text or '户口专用章' in text):return 'household_cover'
    if '户主姓名' in text and '户口专用章' in text and any(t.text.strip()=='址' for t in tokens):return 'household_cover'
    if ('集体土地使用证' in text or '建房用地通知书' in text or '宗地图' in text
        or ('土地使用者' in text and any(s in text for s in ('土地所有者','使用权','地号')))
        or ('比例' in text and '范围红线' in text and any(s in text for s in ('注明边长','共墙')))):
        return 'land_record'
    if '住址变动登记' in text or ('登记事项' in text and ('更正' in text or '变更' in text)):
        return 'change_record'
    if '居民身份证' in text or '有效期限' in text or '签发机关' in text: return 'id_card'
    if '公民身份' in text and '姓名' in text and '民族' in text: return 'id_card'
    if '公民身份' in text and '常住人口' not in text:
        bottom=max((t.y for t in tokens),default=0.)
        label=any('公民身份' in t.text and t.y>bottom*.65 for t in tokens)
        # A multiplication-sign-shaped final glyph is classification evidence
        # only. It is never converted into X in an extracted identity number.
        number=any(re.search(r'(?<!\d)\d{17}[0-9Xx×](?!\d)',''.join(t.text.split())) and t.y>bottom*.65 for t in tokens)
        if label and number:return 'id_card'
    return 'unknown'

def recover_cover_regions(image: np.ndarray,existing: list[Region],engine: OcrEngine,width: int=1500,
                          want: tuple[str,...]=('household_cover',)) -> list[Region]:
    """Local contrast can close faint borders missed beside another card.

    Only additional OCR-confirmed cards of the requested kinds are accepted.
    All OCR field pixels are re-warped from the original scan, not from the
    contrast-enhanced detector view.
    """
    recovered=[]
    for candidate in detect_regions(enhance(image,'clahe'),width):
        x,y,w,h=candidate.bbox
        overlap=False
        for old in existing+recovered:
            xx,yy,ww,hh=old.bbox
            intersection=max(0,min(x+w,xx+ww)-max(x,xx))*max(0,min(y+h,yy+hh)-max(y,yy))
            if intersection/min(w*h,ww*hh)>.5:overlap=True;break
        if overlap:continue
        tokens=engine.read(candidate.image)
        kind=classify(tokens)
        if kind not in want:continue
        top=candidate.table_top;th=candidate.table_height
        dst=np.array([[0,top],[width-1,top],[width-1,top+th],[0,top+th]],np.float32)
        matrix=cv2.getPerspectiveTransform(np.array(candidate.quad,np.float32),dst)
        extended=cv2.warpPerspective(image,matrix,(int(width*1.08),candidate.image.shape[0]),borderValue=(255,255,255))
        candidate.image=extended[:,:width].copy();candidate.extended_image=extended;candidate.kind=kind
        candidate.ocr_tokens=tokens;recovered.append(candidate)
    return recovered
