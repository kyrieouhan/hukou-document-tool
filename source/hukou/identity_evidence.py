"""Conservative links between a damaged registration card and an actual ID scan."""
import re
from difflib import SequenceMatcher
from .ocr import Token
from .models import Household
from .validators import valid_id,clean_id
from .candidates import normalize


def extract_identities(tokens: list[Token],page: int,region: int=-1) -> list[dict]:
    """Read only observed ID numbers and names adjacent to the 姓名 label."""
    identities=[]
    for number_token in tokens:
        value=clean_id(number_token.text)
        found=re.findall(r'(?<![0-9])[0-9]{17}[0-9X](?![0-9])',value)
        for number in found:
            if number_token.confidence<.90 or not valid_id(number)[0]:continue
            anchors=[t for t in tokens if '姓名' in t.text.replace(' ','') and t.y<number_token.y]
            if not anchors:continue
            anchor=max(anchors,key=lambda t:t.y);text=anchor.text.replace(' ','')
            name=normalize('name',text.split('姓名',1)[1]);confidence=anchor.confidence
            if not name:
                height=max(p[1] for p in anchor.box)-min(p[1] for p in anchor.box)
                candidates=[t for t in tokens if t.x>anchor.x and abs(t.y-anchor.y)<max(15,height*.85)
                            and re.fullmatch(r'[\u3400-\u9fff·]{2,24}',normalize('name',t.text))
                            and not any(s in t.text for s in ('民族','性别','住址','姓名'))]
                if candidates:
                    token=min(candidates,key=lambda t:t.x);name=normalize('name',token.text);confidence=min(confidence,token.confidence)
            if name and confidence>=.85:
                identities.append(dict(name=name,id_number=number,name_confidence=confidence,id_confidence=number_token.confidence,
                                       source_page=page,source_region=region))
    return identities


def corroborate_damaged_members(h: Household) -> None:
    """Never repair an 18-digit disagreement or choose a number by checksum.

    A complete, valid number must actually occur on an ID scan. Linking also
    requires a contiguous 14-character match covering the full birth date in a
    truncated (15-17 character) OCR candidate, plus an independently read name
    fragment of at least two characters. The full value comes from the actual
    ID scan, never checksum enumeration. The link always remains LOW/review.
    """
    for member in h.members:
        if valid_id(member.id_number)[0]:
            different_name=[i for i in h.identity_cards if i['id_number']==member.id_number and i['name']!=member.name]
            if different_name:
                member.evidence.setdefault('name',{}).update(tier='LOW',reason='相同身份证号码在登记卡与身份证扫描件上的姓名不一致，保留原识别值待核对',identity_conflicts=different_name)
            same_name=[i for i in h.identity_cards if i['name']==member.name and i['id_number']!=member.id_number]
            if same_name and member.evidence.get('name',{}).get('tier') in ('HIGH','MEDIUM'):
                member.evidence.setdefault('id_number',{}).update(tier='LOW',reason='登记卡与同名身份证扫描件号码不一致，保留两处来源待核对',identity_conflicts=same_name)
            continue
        if re.fullmatch(r'[0-9X]{18}',clean_id(member.id_number)):continue
        ids=[clean_id(c.get('value','')) for c in member.evidence.get('id_number',{}).get('candidates',[]) if c.get('confidence',0)>=.80]
        partials=[v for v in ids if re.fullmatch(r'[0-9X]{15,17}',v)]
        names=[normalize('name',c.get('value','')) for c in member.evidence.get('name',{}).get('candidates',[]) if c.get('confidence',0)>=.55]
        matching={}
        for identity in h.identity_cards:
            name=identity['name'];number=identity['id_number']
            support=[b for v in partials for b in SequenceMatcher(None,v,number,autojunk=False).get_matching_blocks()
                     if b.size>=14 and b.b<=6 and b.b+b.size>=14]
            if not support:continue
            if not any(len(v)>=2 and len(name)-len(v)<=1 and v in name for v in names):continue
            matching[(name,number)]=identity
        if len(matching)!=1:continue
        identity=next(iter(matching.values()))
        reason='身份证扫描件与登记卡连续编号片段及姓名片段相符；已保留两处来源，仍需核对完整号码'
        for key,confidence_key in (('id_number','id_confidence'),('name','name_confidence')):
            if key=='name' and member.evidence.get(key,{}).get('tier')!='LOW':continue
            value=identity['id_number' if key=='id_number' else 'name'];confidence=identity[confidence_key]
            setattr(member,key,value);member.confidence[key]=confidence
            evidence=member.evidence.setdefault(key,{})
            evidence.update(value=value,confidence=confidence,tier='LOW',reason=reason,identity_link_confirmed=True,identity_scan_source=identity)
            evidence.setdefault('candidates',[]).append(dict(value=value,raw_value=value,confidence=confidence,preprocess_type='identity_scan_link',
                scale=1.,source_page=identity['source_page'],source_region=identity['source_region'],engine='v4_id_scan',validity_score=1.,components=[]))
