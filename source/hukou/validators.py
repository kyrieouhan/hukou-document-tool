"""Validation never repairs a digit to satisfy an ID checksum."""
import re
from datetime import date
from .models import Household,Issue

WEIGHTS=(7,9,10,5,8,4,2,1,6,3,7,9,10,5,8,4,2)
CHECK='10X98765432'

def clean_id(value: str) -> str:
    # Only separators are removed: O/I/l must remain invalid, never become digits.
    return re.sub(r'[\s\-·:：]','',value).upper()

def valid_id(value: str) -> tuple[bool,str]:
    value=clean_id(value)
    if not re.fullmatch(r'[0-9]{17}[0-9X]',value): return False,'身份证须为18位，前17位数字，末位数字或X'
    try:
        birthday=date(int(value[6:10]),int(value[10:12]),int(value[12:14]))
        if birthday>date.today() or birthday.year<1800: return False,'身份证出生日期超出合理范围'
    except ValueError: return False,'身份证出生日期无效'
    if value[:6]=='000000' or value[14:17]=='000': return False,'身份证区划或顺序码无效'
    if CHECK[sum(int(v)*w for v,w in zip(value[:17],WEIGHTS))%11]!=value[-1]: return False,'身份证校验位失败'
    return True,''

def parse_date(value: str) -> str:
    match=re.search(r'(?<!\d)(\d{4})\s*(?:[年./-]|\s)\s*(\d{1,2})\s*(?:[月./-]|\s)\s*(\d{1,2})(?:\s*日)?(?!\d)',value)
    if not match: return ''
    try:
        d=date(*map(int,match.groups()))
        return d.isoformat() if date(1900,1,1)<=d<=date.today() else ''
    except ValueError: return ''


def address_review_reason(value: str) -> str:
    """Flag number-slot contamination, without repairing or limiting place names."""
    text=''.join(value.split())
    if (re.search(r'[0-9零〇一二三四五六七八九十百]+[男女](?:号|栋|幢|单元|室|楼)',text)
        or re.search(r'[楼栋幢组室][:：](?:号|室|楼|栋|幢|单元)',text)
        or re.search(r'(?:姓名|性别|出生日期|登记日期|户主或与户主关系)[:：]',text)):
        return '住址疑似混入其他字段或门牌数字未读出，请核对原件；未自动补改字符'
    return ''

def field_state(value: str,tier: str,has_conflict: bool=False) -> str:
    """Map evidence to an explicit field state for the quality gate."""
    if has_conflict:return 'CONFLICT'
    if not value:return 'MISSING'
    if tier in ('HIGH','MEDIUM'):return 'VALID'
    return 'UNCERTAIN'


def explain(field: str,member: str,value: str,problem: str) -> str:
    """One-line, reviewable description instead of a bare threshold note."""
    who=f'成员：{member}；' if member else ''
    shown=value if value else '（空）'
    return f'{who}字段：{field}；当前值：{shown}；问题：{problem}'


def validate(h: Household, threshold: float=.94) -> Household:
    def issue(field,reason,value='',page=0,member=''):
        h.issues.append(Issue(field,reason,value,page,member))
    for field in ('household_no','householder','address'):
        evidence=h.field_evidence.get(field,{}) or {}
        tier=evidence.get('tier','LOW')
        value=getattr(h,field) or ''
        has_conflict=False
        if field=='address' and (warning:=address_review_reason(h.address)):
            evidence=h.field_evidence.setdefault('address',{})
            pages=[c.get('source_page',0) for c in evidence.get('candidates',[]) if c.get('value')==h.address and c.get('source_page',0)>0]
            evidence.update(tier='LOW',reason=warning)
            issue(field,warning,h.address,min(pages,default=0))
            h.field_status[field]=field_state(value,'LOW',True)
            continue
        values={c.get(field,'') for c in h.covers if c.get(field)}
        settled={c.get(field) for c in h.covers if c.get(field) and c.get('fields',{}).get(field,{}).get('tier') in ('HIGH','MEDIUM')}
        if field=='household_no' and len(settled)>1:
            # Truncated OCR of one printed number is not a second household.
            def _digits(v): return ''.join(ch for ch in (v or '') if ch.isdigit())
            nums=sorted(_digits(v) for v in settled if _digits(v))
            distinct=[]
            for n in nums:
                if not any(n in d or d in n for d in distinct):
                    distinct.append(n)
            if len(distinct)>1:
                has_conflict=True
                issue(field,explain(field,'', ' / '.join(sorted(distinct)),'多张首页存在独立确认的不同户号，可能混入其他家庭资料'))
        if len(values)>1 and not h.field_evidence:
            has_conflict=True
            issue(field,explain(field,'',' / '.join(sorted(values)),'重复户口首页字段不一致'))
        if not getattr(h,field):
            issue(field,explain(field,'','','未能可靠识别字段'))
            h.field_status[field]='MISSING'
        elif h.field_evidence:
            evidence=h.field_evidence.get(field,{})
            if evidence.get('tier','LOW')=='LOW':
                reason=evidence.get('reason') or '证据置信度不足'
                issue(field,explain(field,'',getattr(h,field),reason))
                h.field_status[field]=field_state(value,'LOW',has_conflict)
            else:
                h.field_status[field]=field_state(value,evidence.get('tier','LOW'),has_conflict)
        elif not any(c.get('confidence',{}).get(field,0)>=threshold for c in h.covers):
            issue(field,explain(field,'',getattr(h,field),'OCR置信度过低'))
            h.field_status[field]=field_state(value,'LOW',has_conflict)
        else:
            h.field_status[field]=field_state(value,'MEDIUM',has_conflict)
    if not h.members: issue('members','整个PDF未找到有效成员登记卡')
    # Pass 1: identity-based merge. Field completeness is judged on the
    # surviving records only, so a blank duplicate card cannot create a
    # missing-field issue before it is merged away.
    unique=[]; ids={}; weak={}; pending=list(h.members)
    for m in pending:
        good,_reason=valid_id(m.id_number)
        if good and m.id_number in ids:
            other=ids[m.id_number]
            if (m.name,m.relation,m.registration_date)==(other.name,other.relation,other.registration_date):
                continue
            same_person=bool(m.name==other.name and m.relation==other.relation)
            both_dates=bool(m.registration_date and other.registration_date)
            if same_person and not both_dates:
                if m.registration_date and not other.registration_date:
                    ids[m.id_number]=m
                    unique=[u for u in unique if u is not other]
                    unique.append(m)
                continue
            unique.append(m)
        elif good:
            ids[m.id_number]=m
            unique.append(m)
        else:
            weak_key=(m.name,m.relation)
            if m.name and weak_key in weak and (not good or not valid_id(weak[weak_key].id_number)[0]):
                issue('members','姓名关系相同且身份证未确认，疑似重复；未强制合并',m.name,m.source_page,m.name)
            weak[weak_key]=m
            unique.append(m)
    # Pass 2: field-level checks and conflict reporting on unique members.
    valid_count=0
    def reliable(m,key):
        return m.evidence[key].get('tier') in ('HIGH','MEDIUM') if key in m.evidence else m.confidence.get(key,0)>=threshold
    seen_ids=set()
    for m in unique:
        for key in ('name','relation','registration_date','id_number'):
            tier=(m.evidence.get(key,{}) or {}).get('tier','')
            conflict=False
            if not getattr(m,key):
                issue(key,explain(key,m.name,m.raw.get(key,''),'字段缺失或无法确定'),m.raw.get(key,''),m.source_page,m.name)
                m.field_status[key]='MISSING'
            elif not reliable(m,key):
                problem=m.evidence.get(key,{}).get('reason') or 'OCR置信度过低'
                issue(key,explain(key,m.name,getattr(m,key),problem),getattr(m,key),m.source_page,m.name)
                m.field_status[key]=field_state(getattr(m,key),tier or 'LOW',conflict)
            else:
                m.field_status[key]=field_state(getattr(m,key),tier or 'MEDIUM',conflict)
        good,reason=valid_id(m.id_number)
        if not good:
            issue('id_number',explain('id_number',m.name,m.id_number,reason),m.id_number,m.source_page,m.name)
            m.field_status['id_number']='CONFLICT' if m.id_number else 'MISSING'
        if m.registration_date and not parse_date(m.registration_date):
            issue('registration_date',explain('registration_date',m.name,m.registration_date,'登记日期无效'),m.registration_date,m.source_page,m.name)
            m.field_status['registration_date']='CONFLICT'
        if good:
            if m.id_number in seen_ids:
                issue('members',explain('members',m.name,m.id_number,'相同身份证的姓名、关系或登记日期有冲突；未强制合并'),m.id_number,m.source_page,m.name)
            seen_ids.add(m.id_number)
        if good and (reliable(m,'id_number') or m.evidence.get('id_number',{}).get('identity_link_confirmed')): valid_count+=1
    h.members=unique
    heads=[m for m in unique if m.relation=='户主']
    cover_head=h.householder
    if len(heads)==0:
        # The printed cover is the primary head source. A missing head card
        # alone is not a defect when the cover already names the householder.
        if not cover_head:
            issue('householder','未能确认户主')
    elif len(heads)==1:
        if cover_head and heads[0].name!=cover_head:
            issue('householder','户主首页与户主成员卡不一致',f'{cover_head} / {heads[0].name}')
    else:
        names=','.join(m.name for m in heads)
        issue('householder','成员卡中出现多个户主，可能混入其他家庭资料或关系识别错误',names)
    h.rename_reliable=bool(h.householder and len(heads)==1 and heads[0].name==h.householder and
                           reliable(heads[0],'name') and reliable(heads[0],'relation') and
                           not any(i.field=='householder' for i in h.issues))
    h.valid_member_count=valid_count
    # A relation/date transcription dispute does not create an extra person.
    # Count unique validated identities independently of unrelated field quality.
    uncertain=any(i.field in ('members','page','orientation','household_group') for i in h.issues) or valid_count!=len(unique)
    h.family_size=None if uncertain else len(unique)
    if h.family_size is None: issue('family_size','存在未确认或可能遗漏成员，家庭人口待复核',str(valid_count)+'名已验证成员')
    # Quality gate: only high-risk uncertainty blocks OK. Auxiliary warnings
    # (unknown page regions, family-size bookkeeping, orientation hints) stay
    # on the record but do not force a whole-household review when household
    # number, head, identities and dates are already VALID.
    high_risk_fields=('household_no','householder','address','name','relation','id_number','registration_date')
    soft_only={'family_size','page','orientation'}
    hard=[i for i in h.issues if i.field not in soft_only]
    high_risk_bad=any(h.field_status.get(f) in ('MISSING','CONFLICT','UNCERTAIN') for f in ('household_no','householder','id_number'))
    if h.members:
        for m in h.members:
            for f in ('name','id_number'):
                if m.field_status.get(f) in ('MISSING','CONFLICT'):
                    high_risk_bad=True
    else:
        high_risk_bad=True
    h.status='review' if (hard or high_risk_bad) else 'ok'
    return h
