"""Character evidence may select an observed year; it never invents a year."""
from collections import defaultdict
from datetime import date
import re
import cv2
import numpy as np
from .validators import parse_date


def choose_year(years: set[str], digits: list[dict]) -> tuple[str,list[dict]]:
    """Require two recognizers at every position where observed years differ."""
    if len(years)<2:return '',[]
    disputed=[i for i in range(4) if len({y[i] for y in years})>1]
    accepted=[]
    for year in years:
        evidence=[]
        for index in disputed:
            best={}
            for item in digits:
                if item['index']==index and item['value']==year[index] and item['confidence']>=.65:
                    best[item['engine']]=max(best.get(item['engine'],0.),item['confidence'])
            if len(best)<2:break
            evidence.append(dict(index=index,value=year[index],models=len(best),confidence=float(np.median(list(best.values())))))
        else:accepted.append((year,evidence))
    return accepted[0] if len(accepted)==1 else ('',[])


def refine_year(image: np.ndarray,engines: dict,observations: list[dict],result: dict) -> dict:
    """Verify a weak year with small cell expansions and isolated digit reads.

    The selected four-digit string must already be a complete OCR candidate.
    Conflicting character evidence stays LOW, even when it resolves the value.
    """
    from .date_evidence import aligned_slots,numeric_band,numeral_band
    components=result.get('components',[])
    if len(components)!=3:return result
    year=components[0]
    if year['confidence']>=.90 and year['agreement']>=.70 and year['models']>=2:return result
    band=numeral_band(image) or numeric_band(image)
    if not band:return result
    width=image.shape[1];a,b=aligned_slots(image,engines)[0]
    crops=[];keys=[]
    for left in (0.,-.03,-.06):
        for right in (0.,-.03,.03):
            crops.append(image[band[0]:band[1],max(0,int((a+left)*width)):min(width,int((b+right)*width))])
            keys.append((left,right))
    extra=[]
    for name,engine in engines.items():
        values=engine.read_lines(crops) if hasattr(engine,'read_lines') else [engine.read_line(c) for c in crops]
        for key,(raw,confidence) in zip(keys,values):
            value=re.sub(r'[\s年月日:：|_.-]','',raw)
            if re.fullmatch('[0-9]{4}',value) and 1900<=int(value)<=date.today().year and confidence>=.65:
                extra.append(dict(part=0,value=value,raw=raw,confidence=float(confidence),engine=name,mode='year_geometry',band=str(key)))
    years={v['value'] for v in observations+extra if v['part']==0 and len(v['value'])==4 and v['confidence']>=.65}
    if len(years)<2:return result
    base=crops[0];aligned=engines['v6'].read_aligned(base)
    centers=[x for group in aligned for ch,x in zip(group['text'],group['positions']) if ch.isascii() and ch.isdigit()]
    if len(centers)!=4:return result
    pitch=float(np.median(np.diff(centers)))
    bounds=[max(0.,centers[0]-.65*pitch)]+[(x+y)/2 for x,y in zip(centers,centers[1:])]+[min(base.shape[1],centers[-1]+.65*pitch)]
    digits=[]
    for index in range(4):
        if len({y[index] for y in years})==1:continue
        crop=base[:,int(bounds[index]):int(bounds[index+1])]
        variants=[crop,cv2.resize(crop,None,fx=1.5,fy=1),cv2.GaussianBlur(crop,(0,0),.7)]
        for name,engine in engines.items():
            values=engine.read_lines(variants) if hasattr(engine,'read_lines') else [engine.read_line(c) for c in variants]
            for raw,confidence in values:
                value=raw.strip()
                if re.fullmatch('[0-9]',value) and confidence>=.65:
                    digits.append(dict(index=index,value=value,confidence=float(confidence),engine=name))
    selected,support=choose_year(years,digits)
    if not selected:return result
    value=parse_date('-'.join([selected]+[c['value'] for c in components[1:]]))
    if not value:return result
    verified=dict(year,value=selected,confidence=min(d['confidence'] for d in support),models=min(d['models'] for d in support),
                  evidence=extra,character_evidence=digits)
    return dict(result,value=value,confidence=min(result['confidence'],verified['confidence']),tier='LOW',
                components=[verified]+components[1:],observations=observations+extra)


def refine_day(image: np.ndarray,engines: dict,result: dict) -> dict:
    """Recover a clipped leading day digit only from explicit 月...日 OCR evidence."""
    from .date_evidence import aligned_slots,numeric_band,numeral_band
    if result.get('tier')!='LOW' or len(result.get('components',[]))!=3:return result
    a,b=aligned_slots(image,engines)[2];h,w=image.shape[:2]
    bands=[(0,h),numeric_band(image),numeral_band(image)]
    crops=[image[band[0]:band[1],max(int(w*.62),int(w*(a+offset))):min(w,int(w*b))]
           for band in bands if band for offset in (-.05,-.08,-.11)]
    evidence=[];groups=defaultdict(set)
    for name in ('v5','v6','v4'):
        engine=engines[name]
        values=engine.read_lines(crops) if hasattr(engine,'read_lines') else [engine.read_line(c) for c in crops]
        for raw,confidence in values:
            match=re.search(r'月\s*([0-9]{1,2})\s*日',raw)
            if match and confidence>=.90 and 1<=int(match[1])<=31:
                value=str(int(match[1]));groups[value].add(name)
                evidence.append(dict(part=2,value=value,raw=raw,confidence=float(confidence),engine=name,mode='day_unit_bounds',band='expanded'))
    choices=[v for v,models in groups.items() if len(models)>=2]
    if len(choices)!=1 or choices[0]==result['components'][2]['value']:return result
    previous=result['components'][2]['value']
    if len(previous)!=1 or len(choices[0])!=2 or not choices[0].endswith(previous):return result
    components=list(result['components']);old=components[2]
    components[2]=dict(old,value=choices[0],evidence=evidence)
    value=parse_date('-'.join(c['value'] for c in components))
    return dict(result,value=value,components=components,observations=result.get('observations',[])+evidence,tier='LOW') if value else result
