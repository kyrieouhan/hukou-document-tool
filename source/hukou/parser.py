"""Spatial field parsers, isolated from OCR implementation and PDF filenames."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import cv2
from .ocr import Token,OcrEngine
from .imaging import Region
from .models import Member,Household,Issue
from .field_cells import locate
from .candidates import FieldReader,FieldCandidate,resolve

def parse_member(region: Region,tokens: list[Token],engine: OcrEngine,page: int,index: int,
                 reader: FieldReader|None=None,debug: Path|None=None) -> tuple[Member,list]:
    """One observed card becomes one candidate member, with its own footer date."""
    reader=reader or FieldReader(engine)
    region.field_tokens=tokens
    member=Member(source_page=page,region_index=index)
    small=cv2.resize(cv2.cvtColor(region.image,cv2.COLOR_BGR2GRAY),(256,160))
    member.fingerprint=hashlib.sha256(small.tobytes()).hexdigest()
    for key in ('name','relation','id_number','registration_date','household_no'):
        cell=locate(region,key);result=reader.read(cell,key,page,index)
        if key!='household_no':
            setattr(member,key,result.value);member.confidence[key]=result.confidence;member.raw[key]=result.value
        member.evidence[key]=dict(tier=result.tier,reason=result.reason,box=cell.box,
                                   confidence=result.confidence,candidates=[asdict(c) for c in result.candidates],value=result.value,observations=result.observations)
        if debug and cell.image.size:
            debug.mkdir(parents=True,exist_ok=True)
            cv2.imencode('.png',cell.image)[1].tofile(debug/f'{index}_{key}_roi.png')
    return member,[]

def parse_cover(region: Region,tokens: list[Token],engine: OcrEngine,reader: FieldReader|None=None,
                page: int=0,index: int=0,debug: Path|None=None) -> dict:
    """Collect evidence from each repeated cover; household resolver runs later."""
    reader=reader or FieldReader(engine)
    region.field_tokens=tokens
    result={'confidence':{},'evidence':{},'fields':{},'page':page,'region_index':index}
    for key in ('household_no','householder','address'):
        cell=locate(region,key);resolved=reader.read(cell,key,page,index)
        result[key]=resolved.value;result['confidence'][key]=resolved.confidence
        result['evidence'][key]=resolved.tier!='LOW';result['fields'][key]=asdict(resolved)
        if debug and cell.image.size:
            debug.mkdir(parents=True,exist_ok=True)
            cv2.imencode('.png',cell.image)[1].tofile(debug/f'{index}_{key}_roi.png')
    return result

def resolve_household(h: Household) -> None:
    """Cross-page evidence may settle a weak cover, but never hide head conflict."""
    for key in ('household_no','householder','address'):
        candidates=[]
        for c in h.covers:
            candidates.extend(FieldCandidate(**v) for v in c.get('fields',{}).get(key,{}).get('candidates',[]))
        if key=='household_no':
            for m in h.members:candidates.extend(FieldCandidate(**v) for v in m.evidence.get(key,{}).get('candidates',[]))
        resolved=resolve(key,candidates)
        setattr(h,key,resolved.value);h.field_evidence[key]=asdict(resolved)
    for member in h.members:
        if member.relation!='户主' or member.evidence.get('relation',{}).get('tier')=='LOW':continue
        if member.evidence.get('name',{}).get('tier')=='HIGH' and member.name!=h.householder:
            # Two independently settled fields are a real conflict. A minority
            # OCR variant must never silently overwrite the member's name.
            continue
        observed=[FieldCandidate(**v) for v in member.evidence.get('name',{}).get('candidates',[])]
        if not any(c.value==h.householder and c.confidence>=.70 for c in observed):continue
        corroboration=[]
        for cover in h.covers:
            for v in cover.get('fields',{}).get('householder',{}).get('candidates',[]):
                if v['value']==h.householder:corroboration.append(FieldCandidate(**v))
        settled=resolve('name',observed+corroboration)
        if settled.value==h.householder and settled.tier!='LOW':
            member.name=settled.value;member.confidence['name']=settled.confidence
            member.evidence['name'].update(tier=settled.tier,confidence=settled.confidence,reason='',value=settled.value,
                                          candidates=[asdict(c) for c in settled.candidates],cross_source='cover + observed member name')


def _cover_key(cover: dict) -> tuple:
    """Stable identity for clustering: settled household number and head name."""
    no=(cover.get('household_no') or '').strip()
    head=(cover.get('householder') or '').strip()
    addr=(cover.get('address') or '').strip()
    no_tier=(cover.get('fields',{}).get('household_no',{}) or {}).get('tier','')
    head_tier=(cover.get('fields',{}).get('householder',{}) or {}).get('tier','')
    return no,head,addr,no_tier,head_tier


def _digits(value: str) -> str:
    return ''.join(ch for ch in (value or '') if ch.isdigit())


def _same_household_no(a: str,b: str) -> bool:
    da,db=_digits(a),_digits(b)
    if not da or not db:return False
    if da==db:return True
    # Truncated OCR of the same printed number (prefix/substring, len>=5).
    if len(da)>=5 and len(db)>=5 and (da in db or db in da):return True
    return False


def _member_cover_no(member: Member) -> str:
    ev=member.evidence.get('household_no',{}) or {}
    return (ev.get('value') or '').strip()


def cluster_households(h: Household) -> None:
    """Group covers/members into households; keep only the primary group visible.

    Strong links: matching household number or matching head name with similar
    address. Page order decides the primary when several groups remain. If the
    primary cannot be established from real structure, everything is kept but
    the PDF is marked for review instead of guessing.
    """
    covers=list(h.covers or [])
    members=list(h.members or [])
    if not covers:
        # Member-only PDFs have no cover seeds; keep the single household.
        h.household_groups=[{'covers':[],'members':members,'label':'members_only'}]
        h.primary_group_index=0
        return
    if len(covers)<=1 and len({_member_cover_no(m) for m in members if _member_cover_no(m)})<=1:
        # Single cover and no conflicting member header numbers: keep as-is.
        h.household_groups=[{'covers':covers,'members':members,'label':'single'}]
        h.primary_group_index=0
        return
    # Union-find over covers first.
    parent=list(range(len(covers)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        ri,rj=find(i),find(j)
        if ri!=rj:parent[max(ri,rj)]=min(ri,rj)
    keys=[_cover_key(c) for c in covers]
    for i in range(len(covers)):
        for j in range(i+1,len(covers)):
            ni,hi,ai,nti,hti=keys[i]
            nj,hj,aj,ntj,htj=keys[j]
            same_no=_same_household_no(ni,nj)
            same_head=bool(hi and hi==hj)
            same_addr=bool(ai and aj and (ai in aj or aj in ai or _digits(ai)==_digits(aj)))
            if same_no or same_head or (same_addr and same_head):
                union(i,j)
    groups={}
    for i,c in enumerate(covers):
        groups.setdefault(find(i),{'covers':[],'members':[],'label':''})
        groups[find(i)]['covers'].append((i,c))
    # Assign each member to the best group.
    ordered=sorted(groups.items(),key=lambda kv:min(i for i,_ in kv[1]['covers']))
    for m in members:
        m_no=_member_cover_no(m)
        best=None;best_score=0
        for gi,(root,g) in enumerate(ordered):
            score=0
            for _i,c in g['covers']:
                c_no,c_head,c_addr,_nt,_ht=_cover_key(c)
                if m_no and _same_household_no(m_no,c_no):score+=5
                if m.name and m.name==c_head:score+=4
                if m.relation=='户主' and m.name and m.name==c_head:score+=3
                # Page adjacency as auxiliary evidence only.
                if m.source_page and c.get('page') and abs(m.source_page-c['page'])<=1:score+=1
            if score>best_score:
                best_score=score;best=gi
        if best is None or best_score<2:
            # Unassigned: attach to the primary later as review material.
            g=ordered[0][1];g['members'].append(m);g.setdefault('weak_members',0)
            g['weak_members']=g.get('weak_members',0)+1
        else:
            ordered[best][1]['members'].append(m)
    # Choose primary from real page structure, not raw occurrence counts.
    # Earlier first appearance is the strongest structural signal; settled
    # cover fields and a matching head member corroborate. Group size is only
    # a last-resort tiebreaker and never outweighs page order.
    primary=0;primary_score=-1e18
    for gi,(_root,g) in enumerate(ordered):
        pages=[c.get('page') or 10**6 for _i,c in g['covers']]
        first_page=min(pages) if pages else 10**6
        cover_heads=[(_cover_key(c)[1]) for _i,c in g['covers']]
        cover_heads=[x for x in cover_heads if x]
        has_head_member=any(m.relation=='户主' for m in g['members'])
        head_match=any(m.name and m.name in cover_heads for m in g['members'])
        no_tiers=[(_cover_key(c)[3] or '') for _i,c in g['covers']]
        head_tiers=[(_cover_key(c)[4] or '') for _i,c in g['covers']]
        settled=sum(1 for t in no_tiers+head_tiers if t in ('HIGH','MEDIUM'))
        score=-first_page*100.
        score+=25 if settled>=2 else (12 if settled==1 else 0)
        score+=8 if has_head_member else 0
        score+=6 if head_match else 0
        score+=0.01*len(g['members'])
        if score>primary_score:
            primary_score=score;primary=gi
    # Persist groups for debug/export; only primary becomes the visible household.
    if not ordered:
        h.household_groups=[{'covers':covers,'members':members,'label':'fallback'}]
        h.primary_group_index=0
        return
    h.household_groups=[]
    for gi,(_root,g) in enumerate(ordered):
        h.household_groups.append({
            'index':gi,
            'covers':[c for _i,c in g['covers']],
            'members':list(g['members']),
            'cover_pages':[c.get('page') for _i,c in g['covers']],
            'member_pages':[m.source_page for m in g['members']],
            'head_names':[m.name for m in g['members'] if m.relation=='户主'],
            'weak_member_count':g.get('weak_members',0),
            'is_primary':gi==primary,
        })
    h.primary_group_index=primary
    primary_g=ordered[primary][1]
    # Quarantine other groups' evidence so they cannot vote on primary fields.
    h.covers=[c for _i,c in primary_g['covers']]
    h.members=list(primary_g['members'])
    secondary=[gi for gi in range(len(ordered)) if gi!=primary]
    h.field_evidence['household_groups']=[
        {
            'index':item['index'],
            'is_primary':item['is_primary'],
            'cover_pages':item['cover_pages'],
            'member_pages':item['member_pages'],
            'head_names':item['head_names'],
            'member_names':[m.name for m in item['members']],
            'cover_household_nos':[(_cover_key(c)[0]) for c in item['covers']],
            'cover_householders':[(_cover_key(c)[1]) for c in item['covers']],
            'cover_addresses':[(_cover_key(c)[2]) for c in item['covers']],
        }
        for item in h.household_groups
    ]
    if secondary:
        h.field_evidence['secondary_household_count']=len(secondary)
        # Structural finding: never silently keep a multi-household PDF as OK.
        h.issues.append(Issue('household_group',
            '检测到多个不同家庭资料，已按页面结构选择主家庭；其余成员未并入本户',
            f'primary_group={primary} groups={len(ordered)}'))
    if primary_g.get('weak_members'):
        h.issues.append(Issue('members',
            '存在无法可靠归入任一家庭的成员记录，已保留待复核',
            str(primary_g.get('weak_members'))))
