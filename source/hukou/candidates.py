"""Field-specific OCR evidence. Resolvers choose observed values, never invent IDs."""
from dataclasses import dataclass,asdict,field
from collections import defaultdict
import re
import cv2,numpy as np
from .ocr import OcrEngine,PrecisionRecognizer
from .field_cells import Cell,remove_lines
from .imaging import enhance,flatten_background
from .validators import parse_date,valid_id,clean_id
from .text_equivalence import comparison_key

@dataclass
class FieldCandidate:
    value: str
    confidence: float
    preprocess_type: str
    scale: float=1.
    source_page: int=0
    source_region: int=0
    engine: str='v5'
    validity_score: float=0.
    raw_value: str=''
    components: list=field(default_factory=list)

@dataclass
class Resolution:
    value: str
    confidence: float
    tier: str
    reason: str
    candidates: list[FieldCandidate]
    observations: list=field(default_factory=list)

def normalize(key: str,value: str) -> str:
    value=value.strip()
    if key=='registration_date':return parse_date(value)
    value=re.sub(r'\s+','',value)
    if key=='id_number':
        value=clean_id(value)
        matches=re.findall(r'(?<![0-9A-Z])\d{17}[0-9X](?![0-9A-Z])',value)
        return matches[0] if len(matches)==1 else value
    if key=='household_no':
        matches=re.findall(r'\d{5,15}',value)
        return matches[0] if len(matches)==1 else ''
    if key in ('name','householder','relation'):
        value=re.sub(r'^(?:户主姓名|户主或与户主关系|姓名)[:：]?','',value)
        if key=='relation':return value.strip(' |_:：[].。')
        return re.sub(r'^[^\u3400-\u9fff·]+|[^\u3400-\u9fff·]+$','',value)
    if key=='address':return re.sub(r'^住址[:：]?','',value).strip('|_:：')
    return value

def validity(key: str,value: str) -> float:
    if not value:return 0.
    if key=='id_number':return float(valid_id(value)[0])
    if key=='registration_date':return float(bool(parse_date(value)))
    if key=='household_no':return float(bool(re.fullmatch(r'\d{5,15}',value)))
    if key=='relation':
        return float(value!='男' and bool(re.search(r'[\u3400-\u9fff]',value)) and not any(v in value for v in ('户主姓名','登记卡','性别','姓名')))
    if key in ('name','householder'):
        return float(bool(re.fullmatch(r'[\u3400-\u9fff·]{1,24}',value)) and not any(v in value for v in ('常住人口','曾用名','户主姓名','登记卡','性别','姓名')))
    if key=='address':
        chinese=len(re.findall(r'[\u3400-\u9fff]',value));return 1. if chinese>=5 and len(value)>=8 and '户主姓名' not in value else 0.
    return 1.


def relation_medium(value: str,candidates: list[dict]) -> bool:
    """Unanimous models plus high-confidence tight text geometry, without a lexicon."""
    if not validity('relation',value):return False
    meaningful=[c for c in candidates if c['confidence']>=.70 and validity('relation',normalize('relation',c['value']))]
    if {normalize('relation',c['value']) for c in meaningful}!={value}:return False
    models={c['engine'] for c in meaningful}
    geometry={c['preprocess_type'] for c in meaningful if c['confidence']>=.95 and c['preprocess_type'].startswith('detected_')}
    return len(models)>=2 and len(geometry)>=2

def _near_variant(a: str,b: str) -> bool:
    """True when b looks like an OCR slip of a, not a different answer."""
    if not a or not b or a==b:return a==b
    if abs(len(a)-len(b))>2:return False
    if min(len(a),len(b))<6:return False
    mismatches=sum(1 for x,y in zip(a,b) if x!=y)+abs(len(a)-len(b))
    return mismatches<=2


def resolve(key: str,candidates: list[FieldCandidate]) -> Resolution:
    groups=defaultdict(list)
    for c in candidates:
        if not c.raw_value:c.raw_value=c.value
        c.value=normalize(key,c.value);c.validity_score=validity(key,c.value)
        if c.validity_score and c.confidence>=.55:groups[comparison_key(c.value) if key=='address' else c.value].append(c)
    if not groups:
        observed=max(candidates,key=lambda c:c.confidence,default=None)
        return Resolution(observed.value if observed and key in ('id_number','name','householder','relation','address') else '',0.,'LOW','没有合法且清晰的识别候选',candidates)
    glyph_constraints={}
    if key=='address':
        for c in candidates:
            if c.preprocess_type!='glyph_verified':continue
            for p in c.components:
                glyph_constraints.setdefault((p['length'],p['index']),set()).add(p['value'])
        if glyph_constraints and all(len(v)==1 for v in glyph_constraints.values()):
            constrained={v:cs for v,cs in groups.items() if all(len(v)==length and v[index] in values for (length,index),values in glyph_constraints.items())}
            if constrained:groups=constrained
    def score(cs):
        # A rescale of the same image has less weight than another treatment.
        evidence={}
        for c in cs:
            # One engine on one field cell is one source. Preprocessing modes
            # are correlated views, not independent ballots.
            family=(c.engine,c.source_page,c.source_region)
            weight=1.
            if key=='address':
                # Development A/B: mild descreening wins across addresses; raw
                # and contrast views remain weaker corroboration, not extra votes.
                weight=1. if c.preprocess_type.startswith('descreen') else .20
                if c.engine=='v4':weight*=.25
            evidence[family]=max(evidence.get(family,0.),c.confidence*weight**.5)
        return sum(v*v for v in evidence.values())
    ranked=sorted(groups.items(),key=lambda kv:score(kv[1]),reverse=True)
    value,cs=ranked[0];best=score(cs);second=score(ranked[1][1]) if len(ranked)>1 else 0.
    if second and _near_variant(value,ranked[1][0]):
        # One-character OCR slips of the same printed string are noise, not a
        # competing answer. They must not crush the margin into a false review.
        second*=.35
    if not any(c.value==value for c in cs):
        # Output must be an actually observed complete string, even if the
        # comparison key is a different script. No generated replacement value.
        value=max(cs,key=lambda c:c.confidence).value
    # Confidence from one representative per independent engine/cell family,
    # not the median of every correlated preprocessing view.
    family_best={}
    for c in cs:
        family=(c.engine,c.source_page,c.source_region)
        family_best[family]=max(family_best.get(family,0.),c.confidence)
    votes=len(family_best)
    confidence=float(np.median(list(family_best.values()))) if family_best else 0.
    margin=best/(best+second) if best else 0.
    high=votes>=2 and confidence>=.9 and margin>=.70
    medium=(votes>=2 and confidence>=.85 and margin>=.60) or (confidence>=.96 and margin>=.65)
    tier='HIGH' if high else 'MEDIUM' if medium else 'LOW'
    if glyph_constraints:tier='LOW'
    if key in ('name','householder'):
        from .name_evidence import includes_interior
        supporting={c.engine for c in cs if c.confidence>=.70}
        conflicting={c.engine for c in candidates if c.validity_score and c.confidence>=.70 and len(c.value)!=len(value)}
        if len(supporting)<2 and conflicting-supporting:
            tier='LOW'
        if any(c.validity_score and c.confidence>=.70 and includes_interior(value,c.value) for c in candidates):tier='LOW'
    if key=='relation' and tier=='LOW' and relation_medium(value,[asdict(c) for c in candidates]):tier='MEDIUM'
    return Resolution(value,confidence,tier,'' if tier!='LOW' else '候选分歧或证据置信度不足',candidates)

class FieldReader:
    def __init__(self,base: OcrEngine,precision: PrecisionRecognizer|None=None):
        self.base=base;self.precision=precision or PrecisionRecognizer()
        self.alternate=PrecisionRecognizer('PP-OCRv6_rec_medium.onnx')
        self.numeric=PrecisionRecognizer('en_PP-OCRv5_rec_mobile.onnx')

    def read(self,cell: Cell,key: str,page: int,index: int) -> Resolution:
        im=cell.image
        if not im.size:return resolve(key,[])
        # Fast path: one strong engine on lightly treated views. The full
        # multi-engine battery runs only when this cannot settle the field.
        if key not in ('registration_date','address'):
            fast_views={'original':im,'no_lines':remove_lines(im)}
            main_rec=self.precision; main_name='v5'
            fast=[]
            for mode,(value,confidence) in zip(fast_views,main_rec.read_lines(list(fast_views.values()))):
                fast.append(FieldCandidate(value,float(confidence),mode,1.,page,index,main_name))
            value,conf=self.base.read_line(im)
            fast.append(FieldCandidate(value,float(conf), 'original',1.,page,index,'v4'))
            quick=resolve(key,fast)
            if quick.tier in ('HIGH','MEDIUM') and validity(key,quick.value):
                return quick
            candidates=fast
        else:
            candidates=[]
        variants={'original':im,'no_lines':remove_lines(im),'clahe':enhance(im,'clahe'),'sharpen':enhance(im,'sharpen'),
                  'scale2':cv2.resize(im,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC),
                  'scale3':cv2.resize(im,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)}
        if key in ('name','householder'):
            variants['descreen1']=cv2.GaussianBlur(im,(0,0),1.)
            variants['descreen1.5']=cv2.GaussianBlur(im,(0,0),1.5)
        if key=='address':
            smooth=cv2.GaussianBlur(im,(0,0),2.)
            variants.update(descreen=smooth,descreen_no_lines=remove_lines(smooth),descreen_scale2=cv2.resize(smooth,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC))
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        if np.percentile(gray,95)-np.percentile(gray,5)<95:
            variants['flattened']=flatten_background(im)
            variants['flattened_no_lines']=remove_lines(variants['flattened'])
        if key=='registration_date':
            from .date_evidence import numeric_band
            band=numeric_band(im)
            if band:variants['baseline']=im[band[0]:band[1]]
        if key=='address' and cell.line_splits:
            bounds=(0,*cell.line_splits,im.shape[0])
            for name,rec in (('v5',self.precision),('v6',self.alternate)):
                for mode in ('original','no_lines','descreen','descreen_no_lines'):
                    crops=[variants[mode][a:b] for a,b in zip(bounds,bounds[1:])]
                    lines=rec.read_lines(crops)
                    text=''.join(normalize(key,v) for v,_ in lines)
                    confidence=min(c for _,c in lines)
                    candidates.append(FieldCandidate(text,confidence,mode+'/multiline',1.,page,index,name,
                        components=[dict(value=v,confidence=c,line=i) for i,(v,c) in enumerate(lines)]))
            return resolve(key,candidates)
        main_rec=self.alternate if key=='address' else self.precision
        main_name='v6' if key=='address' else 'v5'
        def add(value,confidence,mode,engine=None):
            engine=engine or main_name
            candidates.append(FieldCandidate(value,float(confidence),mode,2. if mode=='scale2' else 3. if mode=='scale3' else 1.,page,index,engine))
        for mode,(value,confidence) in zip(variants,main_rec.read_lines(list(variants.values()))):add(value,confidence,mode)
        # Tight detected text strips provide different geometry from entire cells.
        for mode in ('original','no_lines','clahe',*(['flattened','flattened_no_lines'] if 'flattened' in variants else [])):
            crop=variants[mode]
            impl=getattr(self.base,'impl',None)
            if impl is None:continue
            boxes,_=impl.auto_text_det(crop)
            if boxes is None:continue
            strips=impl.get_crop_img_list(crop,boxes)
            recs=main_rec.read_lines(strips)
            ordered=sorted(zip(boxes,recs),key=lambda v:float(np.mean(v[0][:,0])))
            if key=='id_number':
                digits=[]
                for box,(v,c) in ordered:
                    compact=re.sub(r'\s+','',v).upper()
                    if re.fullmatch(r'[0-9X]{1,18}',compact):
                        digits.append((box,compact,c))
                        if len(compact)==18:add(compact,c,'strip_'+mode)
                for anchor,value,c in digits:
                    if len(value)<9:continue
                    cy=float(np.mean(anchor[:,1]));height=float(np.max(anchor[:,1])-np.min(anchor[:,1]))
                    row=[(b,v,s) for b,v,s in digits if abs(float(np.mean(b[:,1]))-cy)<height*.6]
                    joined=' '.join(v for b,v,s in row)
                    add(joined,min(s for b,v,s in row),'numeric_row_'+mode)
            text=' '.join(v[1][0] for v in ordered);conf=min((v[1][1] for v in ordered),default=0.)
            add(text,conf,'detected_'+mode)
        for mode in ('original','no_lines'):
            value,conf=self.base.read_line(variants[mode]);add(value,conf,mode,'v4')
        secondary=self.precision if key=='address' else self.alternate
        for mode in ('original','no_lines'):
            value,conf=secondary.read_line(variants[mode]);add(value,conf,mode,'v5' if key=='address' else 'v6')
        if key=='address':
            for mode in ('descreen','descreen_no_lines'):
                value,conf=secondary.read_line(variants[mode]);add(value,conf,mode,'v5')
        if 'flattened' in variants:
            for mode in ('flattened','flattened_no_lines'):
                value,conf=secondary.read_line(variants[mode]);add(value,conf,mode,'v5' if key=='address' else 'v6')
        if key=='registration_date':
            from .date_evidence import collect,decide
            from .validators import parse_date
            try:
                observations=collect(im,{'v5':self.precision,'v6':self.alternate,'v4':self.base,'en':self.numeric},[asdict(c) for c in candidates])
                result=decide(observations)
                from .year_verification import refine_year,refine_day
                result=refine_year(im,{'v5':self.precision,'v6':self.alternate,'v4':self.base,'en':self.numeric},observations,result)
                result=refine_day(im,{'v5':self.precision,'v6':self.alternate,'v4':self.base,'en':self.numeric},result)
            except Exception:
                # Empty ROI / geometry failure must not abort the household.
                observations=[]
                result=dict(value='',confidence=0.,tier='LOW',competing=False,components=[],observations=[])
            observations=result.get('observations',observations)
            full=resolve(key,candidates)
            full.observations=observations
            # Conservative release: one legal ISO date and no second legal
            # date means the evidence is settled. Noise views must not
            # demote a unique correct date into a whole-household review.
            value=result.get('value') or ''
            if value and parse_date(value) and not result.get('competing') and result.get('tier')=='LOW':
                result['tier']='MEDIUM'
                result['confidence']=max(float(result.get('confidence') or 0.), .88)
            if result['value']:
                assembled=FieldCandidate(result['value'],result['confidence'],'date_ink_consensus',1.,page,index,'ensemble',1.,result['value'],result['components'])
                if result['tier'] in ('HIGH','MEDIUM'):
                    reason=''
                elif result.get('competing'):
                    reason='日期数字候选仍有分歧'
                else:
                    reason='日期证据不足'
                return Resolution(result['value'],result['confidence'],result['tier'],reason,candidates+[assembled],observations)
            return full
        result=resolve(key,candidates)
        retry=(key=='id_number' and not valid_id(result.value)[0]) or (key=='household_no' and result.tier=='LOW') or (key in ('name','relation') and (not validity(key,result.value) or result.confidence<.70))
        if retry and cell.expanded_image is not None and cell.expanded_image.size:
            extra=self.read(Cell(cell.expanded_image,cell.box,cell.boundary_confidence),key,page,index)
            for c in extra.candidates:c.preprocess_type='expanded/'+c.preprocess_type
            result=resolve(key,candidates+extra.candidates)
            if key=='household_no':
                # Read the lower printed line after widening a weak cover cell;
                # upper parts may contain the neighboring address label.
                expanded=cell.expanded_image
                for fraction in (.20,.35):
                    crop=expanded[int(expanded.shape[0]*fraction):]
                    views=[crop,remove_lines(crop),cv2.GaussianBlur(crop,(0,0),1.5)]
                    for name,engine in (('v5',self.precision),('v6',self.alternate)):
                        for mode,(value,confidence) in zip(('original','no_lines','descreen'),engine.read_lines(views)):
                            candidates.append(FieldCandidate(value,confidence,'overflow_'+mode,1.,page,index,name))
                result=resolve(key,candidates+extra.candidates)
        if key=='address':
            from .glyph_evidence import verify_address
            proof=verify_address(im,result.value,result.candidates,{'v5':self.precision,'v6':self.alternate})
            if proof:
                compatible=[c for c in result.candidates if len(c.value)==len(result.value) and all(c.value[p['index']]==p['value'] for p in proof)]
                if compatible:
                    chosen=resolve(key,compatible)
                    if chosen.value!=result.value:
                        candidate=FieldCandidate(chosen.value,chosen.confidence,'glyph_verified',1.,page,index,'local_glyph',components=proof)
                        result=resolve(key,result.candidates+[candidate])
        if key in ('name','householder'):
            if key=='householder' and result.tier=='LOW':
                from .padded_text import padded_name_candidates
                extra=padded_name_candidates(im,result,{'v5':self.precision,'v6':self.alternate,'v4':self.base},page,index)
                if extra:
                    previous=result.value
                    result=resolve(key,result.candidates+extra)
                    if result.value!=previous:result.tier='LOW';result.reason='扩大文字框后读数改变，保留全部候选待复核'
            from .name_evidence import verify_omission
            value,proof=verify_omission(im,result.value,[asdict(c) for c in result.candidates],self.base,self.alternate)
            if value!=result.value:
                extra=[FieldCandidate(p['value'],p['confidence'],'name_omission/'+p['mode'],1.,page,index,'v6',components=[p]) for p in proof]
                confidence=float(np.median([p['confidence'] for p in proof if p['value']==value]))
                result=Resolution(value,confidence,'LOW','姓名存在漏字分歧；采用文本框多视图一致结果，仍需复核',result.candidates+extra,proof)
        return result

    def date_parts(self,image: np.ndarray,page: int,index: int) -> list[FieldCandidate]:
        """Read three independent printed date slots within the footer only."""
        h,w=image.shape[:2];candidates=[];parts=defaultdict(list)
        crops=[];keys=[]
        gray=cv2.cvtColor(remove_lines(image),cv2.COLOR_BGR2GRAY)
        ink=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
        projection=(ink[:int(h*.92)]>0).sum(axis=0)
        # Split at actual whitespace between printed values, not inside a digit.
        def gap(a,b,default):
            lo,hi=int(a*w),int(b*w)
            blank=np.where(projection[lo:hi]<max(3,h*.045))[0]+lo
            runs=np.split(blank,np.where(np.diff(blank)>1)[0]+1) if len(blank) else []
            runs=[r for r in runs if len(r)>=3]
            return float(np.mean(max(runs,key=len)))/w if runs else default
        boundaries=[gap(.32,.48,.40),gap(.58,.73,.65),gap(.76,.89,.82),1.]
        for band,(y1,y2) in enumerate(((0.,1.),(0.,.70),(.30,.85),(.45,1.))):
            for i,(x1,x2) in enumerate(zip(boundaries,boundaries[1:])):
                for expansion in (0.,.015):
                    crop=image[int(h*y1):int(h*y2),max(0,int(w*(x1-expansion))):min(w,int(w*(x2+expansion)))]
                    crops.append(crop);keys.append((band,i,expansion))
        from datetime import date
        for engine,results in [('v5',self.precision.read_lines(crops)),('v6',self.alternate.read_lines(crops)),('v4',[self.base.read_line(c) for c in crops])]:
            for (band,i,offset),(value,confidence) in zip(keys,results):
                # Remove unit labels only; no similar-character digit replacement.
                value=re.sub(r'[\s年月日:：|_.-]','',value)
                if not value.isascii() or not value.isdigit():continue
                n=int(value)
                if (i==0 and len(value)==4 and 1900<=n<=date.today().year) or (i==1 and len(value)<=2 and 1<=n<=12) or (i==2 and len(value)<=2 and 1<=n<=31):
                    if confidence>=.60:parts[i].append((value,float(confidence),engine,(band,offset)))
        selected=[]
        for i in range(3):
            groups=defaultdict(list)
            for v,c,e,b in parts[i]:groups[str(int(v))].append((c,e,b))
            if not groups:return []
            value,cs=max(groups.items(),key=lambda kv:sum(c*c for c,e,b in kv[1]))
            selected.append((value,cs))
        value=f'{selected[0][0]}-{int(selected[1][0]):02d}-{int(selected[2][0]):02d}'
        if not parse_date(value):return []
        conf=min(float(np.median([c for c,e,b in cs])) for v,cs in selected)
        engines=set.intersection(*[{e for c,e,b in cs} for v,cs in selected])
        for engine in engines or {'mixed'}:
            c=FieldCandidate(value,conf,'date_slots',1.,page,index,engine,1.)
            c.components=[dict(part=i,value=v,evidence=[dict(confidence=s,engine=e,band=b) for s,e,b in cs]) for i,(v,cs) in enumerate(selected)]
            candidates.append(c)
        return candidates
