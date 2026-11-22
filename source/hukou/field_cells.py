"""Locate printed rows and cells inside normalized cards; retain scan overflow."""
from dataclasses import dataclass
import re
import cv2
import numpy as np
from .imaging import Region

@dataclass
class Cell:
    image: np.ndarray
    box: tuple[int,int,int,int]
    boundary_confidence: float
    line_splits: tuple[int,...]=()
    expanded_image: np.ndarray|None=None

def horizontal_rows(image: np.ndarray) -> list[tuple[float,float,float]]:
    """Cluster long, almost horizontal printed rules, returning y/x extents."""
    h,w=image.shape[:2];gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(0,0),1.)
    ink=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,41,12)
    lines=cv2.HoughLinesP(ink,1,np.pi/1800,threshold=max(80,w//5),minLineLength=w*.56,maxLineGap=w*.025)
    groups=[]
    for x1,y1,x2,y2 in ([] if lines is None else lines.reshape(-1,4)):
        if abs(y2-y1)>abs(x2-x1)*.055:continue
        xs=np.arange(min(x1,x2),max(x1,x2),2)
        yy=np.rint(y1+(xs-x1)*(y2-y1)/(x2-x1 if x2!=x1 else 1)).astype(int)
        coverage=np.zeros(len(xs),dtype=bool)
        for dy in (-2,-1,0,1,2):coverage|=ink[np.clip(yy+dy,0,h-1),xs]>0
        if coverage.mean()<.80:continue
        yc=(y1+y2)/2
        found=next((g for g in groups if abs(np.median([v[0] for v in g])-yc)<h*.009),None)
        row=(yc,min(x1,x2),max(x1,x2))
        if found is None:groups.append([row])
        else:found.append(row)
    return sorted([(float(np.median([v[0] for v in g])),float(np.median([v[1] for v in g])),float(np.median([v[2] for v in g]))) for g in groups])

def table_layout(region: Region) -> tuple[float,float,list[float],float]:
    cached=getattr(region,'_field_layout',None)
    if cached is None:
        cached=_table_layout(region)
        region._field_layout=cached
    return cached

def _table_layout(region: Region) -> tuple[float,float,list[float],float]:
    """Find a regular run of printed table rows, excluding a plastic outer frame."""
    rows=horizontal_rows(region.image);h,w=region.image.shape[:2]
    if region.kind=='member_card':
        # V1 member quadrilaterals are already the printed table. Refine locally
        # instead of letting decorative/text lines shift the entire row sequence.
        anchors=[]
        for token in getattr(region,'field_tokens',[]):
            text=''.join(token.text.split())
            if token.x<w*.30:
                for label,row in [('姓名',0),('曾用名',1),('出生地',2),('籍贯',3),('其他住址',4),('其他地址',4),('公民身份',5),('文化程度',6),('服务处所',7)]:
                    if (text==label if label=='姓名' else label in text):anchors.append((row,token.y));break
            elif '出生日期' in text:anchors.append((3,token.y))
        start=region.table_top;step=region.table_height/10
        slopes=[(b[1]-a[1])/(b[0]-a[0]) for a in anchors for b in anchors if b[0]>a[0]]
        slopes=[s for s in slopes if h*.04<s<h*.13]
        if len(slopes)>=2:
            step=float(np.median(slopes));start=float(np.median([y-(row+.5)*step for row,y in anchors]))
        else:
            titles=[max(p[1] for p in t.box) for t in getattr(region,'field_tokens',[]) if '常住人口' in t.text]
            # Some scans enclose the title in an outer rounded frame. Infer the
            # printed grid only when that title proves the old top is too high.
            if titles and max(titles)>start+step*.1:
                below=[r[0] for r in rows if r[0]>max(titles)-step*.1]
                gaps=[b-a for a,b in zip(below,below[1:]) if h*.035<b-a<h*.13]
                if len(gaps)>=6:
                    fitted=float(np.median(gaps));top=below[0]
                    support=sum(any(abs(y-(top+k*fitted))<fitted*.25 for y in below) for k in range(11))
                    if support>=9:
                        start,step=top,fitted
                        region._inferred_grid=True
        ys=[]
        for k in range(11):
            target=start+k*step
            near=min(rows,key=lambda r:abs(r[0]-target),default=(target,0,w-1))
            ys.append(near[0] if abs(near[0]-target)<step*.30 else target)
        return 0.,float(w-1),ys,.98
    if region.kind=='household_cover':
        tokens=getattr(region,'field_tokens',[])
        first=[t.y for t in tokens if '户主姓名' in ''.join(t.text.split())]
        second=[t.y for t in tokens if ('住址' in ''.join(t.text.split()) or ''.join(t.text.split()) in ('住','址','户号'))
                and (not first or t.y>min(first))]
        if first and second:
            step=float(np.median(second)-np.median(first))
            if h*.035<step<h*.18:
                top=float(np.median(first))-.5*step
                ys=[]
                for k in range(3):
                    expected=top+k*step
                    near=min(rows,key=lambda r:abs(r[0]-expected),default=(expected,0,w-1))
                    ys.append(near[0] if abs(near[0]-expected)<step*.32 else expected)
                return 0.,float(w-1),ys,.98
    needed=3 if region.kind=='household_cover' else 11
    best=None
    for i,a in enumerate(rows):
        if a[0]>h*.42:continue
        for b in rows[i+1:]:
            step=b[0]-a[0]
            if not h*.045<step<h*.13:continue
            hits=[];distance=0.
            for k in range(needed):
                target=a[0]+k*step
                nearest=min(rows,key=lambda r:abs(r[0]-target))
                if abs(nearest[0]-target)<step*.23:
                    hits.append(nearest);distance+=abs(nearest[0]-target)/step
            if len(hits)<min(needed,8):continue
            score=len(hits)*10-distance
            # For covers, address labels identify the second of the two small rows.
            if needed==3:score-=a[0]/h*.1
            else:score-=a[0]/step*3
            if best is None or score>best[0]:best=(score,hits,a[0],step)
    if best:
        _,hits,top,step=best
        ys=[]
        for k in range(needed):
            target=top+k*step;near=min(rows,key=lambda r:abs(r[0]-target))
            ys.append(near[0] if abs(near[0]-target)<step*.23 else target)
        return float(min(r[1] for r in hits)),float(max(r[2] for r in hits)),ys,.98
    top=region.table_top;th=region.table_height
    return 0.,float(w-1),[top+k*th*.1 for k in range(needed)],.6

def vertical_boundary(image: np.ndarray,y1: int,y2: int,expected: float,tolerance: float) -> int:
    """Snap a normalized seed to a nearby vertical rule, without changing row."""
    gray=cv2.cvtColor(image[max(0,y1):min(image.shape[0],y2)],cv2.COLOR_BGR2GRAY)
    if gray.shape[0]<12:return round(expected)
    ink=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,12)
    lines=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((max(12,int(gray.shape[0]*.65)),1),np.uint8))
    counts=(lines>0).sum(axis=0);lo=max(0,round(expected-tolerance));hi=min(len(counts),round(expected+tolerance)+1)
    inds=np.where(counts[lo:hi]>gray.shape[0]*.6)[0]+lo
    return int(min(inds,key=lambda x:abs(x-expected))) if len(inds) else round(expected)

def locate(region: Region,key: str) -> Cell:
    left,right,ys,confidence=table_layout(region);width=right-left;line_cuts=[]
    if region.kind=='household_cover':
        row=0 if key=='householder' else 1
        bounds={'householder':(.48,1.015),'household_no':(.125,.315),'address':(.425,1.035)}
    else:
        row={'name':0,'relation':0,'id_number':5}.get(key,0)
        bounds={'name':(.205,.555),'relation':(.735,1.035),'id_number':(.205,.555),'household_no':(0,.40)}
    if key=='registration_date':
        # Footer only: deliberately excludes every birth/migration date in the table.
        step=float(np.median(np.diff(ys)));y1=ys[-1]-step*.40;y2=ys[-1]+step*1.45
        anchors=[t for t in getattr(region,'field_tokens',[]) if '登记' in t.text and t.y>ys[0]+step*8]
        if anchors:
            anchor=max(anchors,key=lambda t:t.y)
            y1=min(y1,min(p[1] for p in anchor.box)-step*.40)
            y2=max(y2,max(p[1] for p in anchor.box)+step*.30)
        x1=left+width*.595;x2=right+width*.015
    elif region.kind=='member_card' and key=='household_no':
        step=float(np.median(np.diff(ys)));margin=1.3 if getattr(region,'_inferred_grid',False) else .85
        y1=max(0,ys[0]-step*margin);y2=ys[0]-3;x1=left;x2=left+width*.40
    else:
        y1,y2=ys[row],ys[row+1];a,b=bounds[key]
        x1=vertical_boundary(region.image,int(y1),int(y2),left+width*a,width*.055)
        x2=vertical_boundary(region.image,int(y1),int(y2),left+width*b,width*.035) if b<1 else left+width*b
        # Dot-matrix printing often crosses the lower rule. Keep a small margin,
        # while staying far above the next row's text baseline.
        row_height=y2-y1
        x1+=4;y1+=3;y2+=row_height*.22
        if region.kind=='household_cover' and key=='householder':
            # The cover's name cell sits above the address cell, but the long
            # address value can print high and enter this row band. Prefer the
            # name-row value token; never fall through to the address strip.
            step=float(np.median(np.diff(ys))) if len(ys)>1 else row_height
            labels=[t for t in getattr(region,'field_tokens',[]) if '户主姓名' in ''.join(t.text.split())]
            label_right=max((max(p[0] for p in t.box) for t in labels),default=left+width*.50)
            name_cands=[]
            for token in getattr(region,'field_tokens',[]):
                text=''.join(token.text.split())
                if not text:continue
                if any(s in text for s in ('户主姓名','户主','住址','户号','家庭户','户别','常住人口','登记','机关','专用')):continue
                if not re.search(r'[一-鿿]',text):continue
                if token.x<=label_right-4:continue
                if not (ys[0]-step*.25<token.y<ys[1]-step*.25):continue
                name_cands.append(token)
            if name_cands:
                token=min(name_cands,key=lambda t:(t.y,t.x))
                upper=min(p[1] for p in token.box);lower=max(p[1] for p in token.box)
                y1=max(0,upper-6);y2=lower+6
                x1=min(x1,min(p[0] for p in token.box)-6)
                x2=max(x2,max(p[0] for p in token.box)+6)
            else:
                y2=min(y2,ys[row+1]+min(6.,row_height*.06))
                addr_labels=[t for t in getattr(region,'field_tokens',[])
                             if '住址' in ''.join(t.text.split()) or ''.join(t.text.split()) in ('住','址')]
                if addr_labels:
                    addr_top=min(min(p[1] for p in t.box) for t in addr_labels)
                    y2=min(y2,addr_top-4)
        if b<1:x2+=width*.015 if key=='household_no' else -4
    # Full-card text detection supplies geometry only. Re-read original field
    # pixels; never treat the first OCR text as the final value. This handles
    # print offsets and residual perspective without moving the stable page detector.
    tokens=getattr(region,'field_tokens',[])
    step=float(np.median(np.diff(ys)))
    if region.kind=='member_card' and key in ('name','relation','id_number'):
        lo,hi=(.20,.56) if key=='name' else (.72,1.08) if key=='relation' else (.20,.63)
        base=ys[0] if key!='id_number' else ys[5]
        possible=[]
        for token in tokens:
            text=''.join(token.text.split())
            position=min(p[1] for p in token.box) if key in ('name','relation') else token.y
            if not (left+width*lo<token.x<left+width*hi and base-step*.55<position<base+step*.95):continue
            if any(label in text for label in ('常住人口','登记卡','家庭户','户号','姓名','曾用名','性别','户主或','公民身份')):continue
            if key=='name' and (re.search(r'\d',text) or text in ('姓','名')):continue
            if key=='relation' and text in ('男','性','别','与系','主','关','户主关系','民族','族','汉族'):continue
            if key=='relation' and not re.search(r'[\u3400-\u9fff]',text):continue
            if key=='id_number' and len(re.findall(r'[0-9]',text))<12:continue
            possible.append(token)
        if possible:
            token=min(possible,key=lambda t:t.y)
            upper=min(p[1] for p in token.box);lower=max(p[1] for p in token.box)
            if lower-upper<step*1.35:
                y1=upper-7;y2=lower+7
                # A detector may join the final 名 label glyph across the cell
                # divider with the value. Keep the actual cell boundary in that
                # case; never delete 名 from a person's name by string matching.
                label_overlap=key=='name' and token.text.strip().startswith('名') and min(p[0] for p in token.box)<left+width*.20
                if not label_overlap:x1=min(x1,min(p[0] for p in token.box)-6)
                x2=max(x2,max(p[0] for p in token.box)+6)
            elif key in ('name','relation'):
                # A tall text-detector box can merge the target with the next
                # row (e.g. relation and sex); retain the first printed row.
                y1=upper-7;y2=upper+step*.90
                x1=min(x1,min(p[0] for p in token.box)-6)
    if region.kind=='household_cover' and key=='householder':
        unrelated=[t for t in tokens if t.text.strip() in ('户主','男','女') and t.x>left+width*.77 and ys[0]-step*.2<t.y<ys[1]]
        if unrelated:x2=min(x2,min(min(p[0] for p in t.box) for t in unrelated)-8)
    if region.kind=='household_cover' and key=='address':
        main_lines=[t for t in tokens if t.x>left+width*.43 and ys[1]<t.y<ys[2] and len(t.text.replace(' ',''))>=8]
        baseline=min((t.y for t in main_lines),default=(ys[1]+ys[2])*.5)
        continuation=[t for t in tokens if t.x>left+width*.43 and max(ys[2]-step*.25,baseline+step*.35)<t.y<ys[2]+step*.80
                      and re.search(r'[楼栋幢单元组号室层]',t.text) and not any(s in t.text for s in ('机关','专用','登记'))]
        if continuation:
            y2=max(y2,max(max(p[1] for p in t.box) for t in continuation)+8)
            lower_top=min(min(p[1] for p in t.box) for t in continuation)
            first=[t for t in tokens if t.x>left+width*.43 and ys[1]<t.y<ys[2] and t not in continuation]
            upper_bottom=max((max(p[1] for p in t.box) for t in first),default=ys[2]-step*.2)
            line_cuts=[int((upper_bottom+lower_top)/2)]
    canvas=getattr(region,'extended_image',None)
    if canvas is None:canvas=region.image
    x1=max(0,int(x1));y1=max(0,int(y1));x2=min(canvas.shape[1],int(x2));y2=min(canvas.shape[0],int(y2))
    crop=canvas[y1:y2,x1:x2].copy()
    expanded=None
    if key=='id_number':
        expanded=canvas[max(0,int(y1-step*.18)):min(canvas.shape[0],int(y2+step*.40)),x1:min(canvas.shape[1],int(right-width*.25))].copy()
    elif region.kind=='member_card' and key in ('name','relation'):
        expanded=canvas[y1:min(canvas.shape[0],int(y2+step*.45)),x1:x2].copy()
    elif region.kind=='household_cover' and key=='household_no':
        # Misregistered printing may cross the divider into the address label.
        expanded=canvas[y1:min(canvas.shape[0],int(y2+step*.20)),x1:min(canvas.shape[1],int(left+width*.43))].copy()
    return Cell(crop,(x1,y1,x2,y2),confidence,tuple(v-y1 for v in line_cuts if y1<v<y2),expanded)

def remove_lines(image: np.ndarray) -> np.ndarray:
    """Only long table rules, not stamps; optional candidate, never the only view."""
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ink=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    horizontal=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,max(50,int(image.shape[1]*.55))),np.uint8))
    vertical=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((max(35,int(image.shape[0]*.85)),1),np.uint8))
    mask=cv2.dilate(cv2.bitwise_or(horizontal,vertical),np.ones((3,3),np.uint8))
    return cv2.inpaint(image,mask,2,cv2.INPAINT_TELEA)
