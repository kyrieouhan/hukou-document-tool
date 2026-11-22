"""Recover the three printed date slots using ink and complete-line evidence."""
from collections import defaultdict
from datetime import date
import re
import cv2,numpy as np
from .validators import parse_date
from .field_cells import remove_lines

def numeral_band(image: np.ndarray) -> tuple[int,int]|None:
    """Use the aligned dark month/day glyphs to locate the numeral baseline.

    Preprinted units and stamps often sit above these glyphs. Components must
    agree on their lower edge; no date value participates in this geometry.
    """
    h,w=image.shape[:2]
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    patch=gray[:,int(w*.62):int(w*.98)]
    cutoff=max(50,min(110,float(np.median(patch))-50))
    _,_,stats,_=cv2.connectedComponentsWithStats(np.uint8(patch<cutoff),8)
    glyphs=[]
    for x,y,cw,ch,area in stats[1:]:
        if .10*h<ch<.50*h and .008*w<cw<.10*w and area>=15 and y+ch>h*.45:
            glyphs.append((int(y),int(y+ch),int(area)))
    if len(glyphs)<2:return None
    clusters=[[g for g in glyphs if abs(g[1]-ref[1])<max(5,h*.055)] for ref in glyphs]
    group=max(clusters,key=lambda gs:sum(g[2] for g in gs))
    if len(group)<2:return None
    top=min(g[0] for g in group);bottom=max(g[1] for g in group)
    pad=max(3,int((bottom-top)*.12))
    return max(0,top-pad),min(h,bottom+pad)

def numeric_band(image: np.ndarray) -> tuple[int,int]|None:
    """Find the printed numeral baseline from the right-hand date slot.

    This keeps full glyph height while dropping empty space, table rules and
    the unrelated upper part of a stamp. It does not erase any glyph pixels.
    """
    h,w=image.shape[:2];gray=cv2.cvtColor(remove_lines(image),cv2.COLOR_BGR2GRAY)
    patch=gray[:,int(w*.60):int(w*.97)]
    threshold=max(45,min(130,float(np.median(patch))-35))
    ink=patch<threshold;projection=ink.sum(axis=1)
    # The bottom table rule is a spatial guard against migration text in the
    # expanded upper margin. Keep the lowest endpoint for a slanted footer.
    edges=cv2.Canny(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),60,160)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,70,minLineLength=w*.55,maxLineGap=20)
    borders=[]
    for x1,y1,x2,y2 in ([] if lines is None else lines.reshape(-1,4)):
        if abs(y2-y1)<abs(x2-x1)*.08 and max(y1,y2)<h*.50:borders.append(min(y1,y2))
    if borders:projection[:max(0,int(max(borders))+2)]=0
    active=np.uint8(projection>=max(2,patch.shape[1]*.012))*255
    active=cv2.morphologyEx(active[:,None],cv2.MORPH_CLOSE,np.ones((7,1),np.uint8)).ravel()
    indices=np.where(active>0)[0]
    runs=np.split(indices,np.where(np.diff(indices)>1)[0]+1) if len(indices) else []
    runs=[r for r in runs if 10<=len(r)<h*.80]
    if not runs:return None
    selected=max(runs,key=lambda r:float(projection[r].sum()))
    pad=max(4,int(len(selected)*.15))
    return max(0,int(selected[0])-pad),min(h,int(selected[-1])+pad+1)

def slots(image: np.ndarray) -> list[tuple[float,float]]:
    h,w=image.shape[:2]
    gray=cv2.cvtColor(remove_lines(image),cv2.COLOR_BGR2GRAY)
    ink=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    projection=(ink[:int(h*.92)]>0).sum(axis=0)
    def gap(a,b,default):
        lo,hi=int(a*w),int(b*w);blank=np.where(projection[lo:hi]<max(3,h*.045))[0]+lo
        runs=np.split(blank,np.where(np.diff(blank)>1)[0]+1) if len(blank) else []
        runs=[r for r in runs if len(r)>=3]
        return float(np.mean(max(runs,key=len)))/w if runs else default
    # Constrain the gap search to the inter-slot space: a stamp can otherwise
    # move a "widest gap" past a leading digit (12 must not become 2).
    boundaries=[gap(.36,.45,.40),gap(.59,.65,.63),gap(.75,.81,.79),1.]
    return list(zip(boundaries,boundaries[1:]))

def aligned_slots(image: np.ndarray,engines: dict) -> list[tuple[float,float]]:
    """Locate year/month/day by OCR alignment, not fixed printed offsets.

    A misread year can still locate its pixels. Only the subsequent cropped
    recognition contributes numeric values to the final date.
    """
    h,w=image.shape[:2];band=numeric_band(image);views=[]
    digits=numeral_band(image)
    digit_view=image[digits[0]:digits[1]] if digits else None
    if band:
        tight=image[band[0]:band[1]];views.append(tight)
        gray=cv2.cvtColor(tight,cv2.COLOR_BGR2GRAY)
        for threshold in (90,130):views.append(cv2.cvtColor(cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY)[1],cv2.COLOR_GRAY2BGR))
    views.append(image)
    if digit_view is not None:views.append(digit_view)
    candidates=[];fallback_candidates=[]
    for en in ('v5','v6'):
        engine=engines.get(en)
        if engine is None or not hasattr(engine,'read_aligned'):continue
        for view in views:
            destination=fallback_candidates if view is digit_view else candidates
            runs=[];groups=engine.read_aligned(view)
            units=[(ch,x) for group in groups for ch,x in zip(group['text'],group['positions']) if ch in ('月','日')]
            for group in groups:
                current=[]
                for ch,x in zip(group['text'],group['positions']):
                    if ch.isascii() and ch.isdigit():
                        if current and x-current[-1][1]>view.shape[0]*.75:
                            runs.append(current);current=[]
                        current.append((ch,x))
                    elif current:runs.append(current);current=[]
                if current:runs.append(current)
            for i,run in enumerate(runs):
                s=''.join(c for c,x in run)
                if not 3<=len(run)<=4 or not s.startswith(('19','20')) or run[0][1]<w*.20:continue
                spacing=float(np.median(np.diff([x for c,x in run])))
                b0=max(0.,(run[0][1]-spacing*1.2)/w)
                month_units=[x for ch,x in units if ch=='月' and x>run[-1][1]+w*.04]
                if month_units:
                    month_unit=min(month_units)
                    day_units=[x for ch,x in units if ch=='日' and x>month_unit+w*.05]
                    if day_units:
                        day_unit=min(day_units)
                        b1=(run[-1][1]+.35*(month_unit-run[-1][1]))/w
                        b2=(month_unit+.35*(day_unit-month_unit))/w
                        if .12<b1-b0<.50 and .08<b2-b1<.35:
                            destination.append((b0,b1,b2,1.));continue
                following=[r for r in runs[i+1:] if 1<=len(r)<=2 and r[0][1]>run[-1][1]+w*.055]
                if len(following)<2:continue
                month,day=following[:2]
                if day[0][1]-month[-1][1]<w*.05:continue
                b1=(run[-1][1]+month[0][1])/2/w
                b2=(month[-1][1]+day[0][1])/2/w
                if .12<b1-b0<.50 and .08<b2-b1<.35:
                    destination.append((b0,b1,b2,1.))
    # Preserve established unit anchors. The extra baseline only recovers
    # missing anchors; mixing it into an already valid fit can shift a cell.
    candidates=candidates or fallback_candidates
    if candidates:
        b=np.median(np.array(candidates),axis=0).tolist()
        gray=cv2.cvtColor(remove_lines(image),cv2.COLOR_BGR2GRAY)
        y1,y2=band or (0,h);gray=gray[y1:y2]
        threshold=max(45,min(130,float(np.median(gray))-35))
        projection=(gray<threshold).sum(axis=0)
        def left_gap(x):
            lo=max(0,int((x-.06)*w));hi=min(w,int(x*w)+1)
            indices=np.where(projection[lo:hi]<max(2,gray.shape[0]*.04))[0]+lo
            runs=np.split(indices,np.where(np.diff(indices)>1)[0]+1) if len(indices) else []
            runs=[r for r in runs if len(r)>=4]
            return float(np.mean(max(runs,key=len)))/w if runs else max(0.,x-.015)
        b[1]=left_gap(b[1]);b[2]=left_gap(b[2])
        return list(zip(b,b[1:]))
    return slots(image)

def collect(image: np.ndarray,engines: dict,whole: list[dict]) -> list[dict]:
    """Retain every raw observation for auditable, field-local decisions."""
    h,w=image.shape[:2];crops=[];keys=[]
    bands=[(0,0,h),(1,0,int(h*.75))]
    baseline=numeric_band(image)
    if baseline:bands.append(('baseline',baseline[0],baseline[1]))
    digits=numeral_band(image)
    if digits:bands.append(('numeral',digits[0],digits[1]))
    for part,(a,b) in enumerate(aligned_slots(image,engines)):
        for band,y1,y2 in bands:
            x1=int(w*a);x2=min(w,int(w*b))
            if y2<=y1 or x2<=x1:continue
            im=image[y1:y2,x1:x2]
            if im is None or im.size==0 or im.shape[0]<4 or im.shape[1]<4:continue
            gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            variants={'original':im}
            for threshold in (90,130):
                ink=cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY)[1]
                variants[f'ink{threshold}']=cv2.cvtColor(ink,cv2.COLOR_GRAY2BGR)
                opened=cv2.morphologyEx(255-ink,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
                variants[f'ink{threshold}_open']=cv2.cvtColor(255-opened,cv2.COLOR_GRAY2BGR)
            for mode,crop in variants.items():
                if mode.startswith('ink'):
                    fraction=float((crop[:,:,0]<128).mean())
                    # Reject over-thresholded backgrounds and almost blank views.
                    if not .018<=fraction<=.40:continue
                crops.append(crop);keys.append((part,band,mode))
    observations=[]
    for name,engine in engines.items():
        values=engine.read_lines(crops) if hasattr(engine,'read_lines') else [engine.read_line(c) for c in crops]
        for (part,band,mode),(value,confidence) in zip(keys,values):
            raw=value;value=re.sub(r'[\s年月日:：|_.-]','',value)
            if not value.isascii() or not value.isdigit():continue
            n=int(value)
            valid=(part==0 and len(value)==4 and 1900<=n<=date.today().year) or (part==1 and len(value)<=2 and 1<=n<=12) or (part==2 and len(value)<=2 and 1<=n<=31)
            if valid and confidence>=.60:observations.append(dict(part=part,value=str(n),raw=raw,confidence=float(confidence),engine=name,mode=mode,band=band))
    for candidate in whole:
        if candidate.get('preprocess_type','').startswith('date_') or candidate.get('engine') not in ('v4','v5','v6','en'):continue
        raw=candidate.get('raw_value') or candidate.get('value','');value=parse_date(raw)
        if value:
            components=list(enumerate(value.split('-')));mode='whole'
        else:
            # A missing year digit must not discard clearly read month/day
            # tokens in the same anchored footer (e.g. "202年8月26日").
            components=[];mode='partial_whole'
            for i,pattern in enumerate((r'(?<!\d)(\d{4})\s*年',r'(?<!\d)(\d{1,2})\s*月',r'月\s*(\d{1,2})\s*(?:日|$)')):
                match=re.search(pattern,raw)
                if match:
                    n=int(match.group(1))
                    if (i==0 and 1900<=n<=date.today().year) or (i==1 and 1<=n<=12) or (i==2 and 1<=n<=31):components.append((i,str(n)))
        for i,v in components:
            observations.append(dict(part=i,value=str(int(v)),raw=raw,confidence=candidate['confidence'],engine=candidate['engine'],mode=mode,band=candidate['preprocess_type']))
    return observations

def decide(observations: list[dict]) -> dict:
    components=[]
    for part in range(3):
        groups=defaultdict(list)
        # Thresholding can close the open stroke of 6 into 0. When at least
        # three recognizers agree on the original numeral pixels, retain that
        # observed year instead of outvoting it with correlated binary views.
        original=defaultdict(set)
        if part==0:
            for r in observations:
                if r['part']==part and r['band']=='numeral' and r['mode']=='original' and r['confidence']>=.90:
                    original[r['value']].add(r['engine'])
        preferred=next((v for v,models in original.items() if len(models)>=3),None)
        high_models={r['engine'] for r in observations if r['part']==part and r['confidence']>=.90}
        cutoff=.90 if len(high_models)>=2 else .65
        for r in observations:
            if r['part']==part and r['confidence']>=cutoff:groups[r['value']].append(r)
        if not groups:
            components.append(dict(part=part,value='',confidence=0.,agreement=0.,models=0,evidence=[],competing=False))
            continue
        def score(rs):
            sources={}
            for r in rs:
                ink=r['mode'].startswith('ink');whole=r['mode'] in ('whole','partial_whole')
                weights=((1.,1.,1.),(1.,1.,4.),(2.,1.,4.))
                weight=weights[part][0 if ink else 2 if whole else 1]
                if r['mode']=='partial_whole':weight*=.25
                # Development numeric A/B found V4 more prone to unit/digit
                # confusion. Retain it as corroboration, not an equal voter.
                if r['engine']=='v4':weight*=.25
                # Preprocessing modes on the same crop band are correlated
                # views of one image, not independent ballots.
                key=(r['engine'],'line' if whole else str(r['band']))
                sources[key]=max(sources.get(key,0.),weight*r['confidence']**2)
            return sum(sources.values())
        ranking=sorted(groups.items(),key=lambda kv:(kv[0]==preferred,score(kv[1])),reverse=True)
        value,rs=ranking[0];best=score(rs);second=score(ranking[1][1]) if len(ranking)>1 else 0.
        # A second place made only of weak noise must not look like a real dispute.
        if len(ranking)>1:
            rival_value,rival=ranking[1]
            rival_strong=[r for r in rival if r['confidence']>=.85]
            if len(rival_strong)<2:
                second*=.35
            competing=bool(second>0 and best/(best+second)<.78 and len(rival_strong)>=2)
        else:
            competing=False
        model_best={}
        for r in rs:model_best[r['engine']]=max(model_best.get(r['engine'],0.),r['confidence'])
        components.append(dict(part=part,value=value,confidence=float(np.median(sorted(model_best.values(),reverse=True)[:3])),
                               agreement=1. if value==preferred else best/(best+second) if (best+second) else 1.,
                               models=len({r['engine'] for r in rs}),evidence=rs,competing=competing))
    value=parse_date('-'.join(c['value'] for c in components if c.get('value'))) if all(c.get('value') for c in components) and len(components)==3 else ''
    if not value:
        # Partial slots (e.g. year+month only) must not invent a date.
        return dict(value='',confidence=0.,tier='LOW',competing=False,components=components,observations=observations)
    confidence=min(c['confidence'] for c in components)
    high=bool(value) and all(c['models']>=2 and c['confidence']>=.88 and c['agreement']>=.7 for c in components)
    # A settled date with multi-engine support is MEDIUM, not a dispute, even
    # when a few correlated noise views disagree.
    medium=(not high) and bool(value) and all(c['models']>=2 and c['confidence']>=.80 and c['agreement']>=.55 for c in components)
    competing=any(c.get('competing') for c in components)
    tier='HIGH' if high else 'MEDIUM' if medium else 'LOW'
    return dict(value=value,confidence=confidence,tier=tier,competing=competing,components=components,observations=observations)
