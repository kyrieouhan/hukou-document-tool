# The numeric examples use a synthetic region prefix outside the mainland code ranges.
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from hukou.validators import valid_id,clean_id,parse_date,validate
from hukou.models import Household,Member

def member(**kw):
    args=dict(name='测试甲',relation='户主',id_number='00990419491231002X',registration_date='2024-11-01',
              confidence={k:1. for k in ['name','relation','id_number','registration_date']})
    args.update(kw)
    return Member(**args)

def household(members):
    return Household(household_no='123456',householder='测试甲',address='虚构测试地址',members=members,
                     covers=[dict(household_no='123456',householder='测试甲',address='虚构测试地址',confidence={k:1. for k in ['household_no','householder','address']})])

def test_id_checksum_and_no_repair():
    assert valid_id('00990419491231002X')[0]
    assert valid_id('00990419491231002x')[0]
    assert not valid_id('009904194912310021')[0]
    assert not valid_id('00990419490231002X')[0]
    assert not valid_id('00990419491231OO2X')[0]
    assert clean_id('009904 19491231 002x')=='00990419491231002X'
    assert clean_id('00990419491231OO2X')=='00990419491231OO2X'

def test_date_and_member_scope():
    assert parse_date('2024年11月13日')=='2024-11-13'
    assert parse_date('2024年11月')==''
    assert parse_date('2024-02-30')==''
    h=household([member(),member(name='测试乙',relation='曾孙女或外曾孙女',id_number='009904194912310011',registration_date='2024-11-13')])
    validate(h)
    assert h.members[0].registration_date!=h.members[1].registration_date
    assert h.members[1].relation=='曾孙女或外曾孙女'

def test_dedupe_and_conflicts():
    h=validate(household([member(),member()]))
    assert len(h.members)==1 and h.family_size==1
    h=validate(household([member(),member(name='测试乙')]))
    assert len(h.members)==2 and h.family_size is None and h.status=='review'
    assert any(i.field=='members' for i in h.issues)
    h=validate(household([member(id_number=''),member(id_number='')]))
    assert len(h.members)==2 and h.family_size is None

def test_head_mismatch_and_low_confidence():
    h=household([member(name='测试乙')])
    assert validate(h).rename_reliable is False
    h=household([member(confidence={'name':.5,'relation':1,'id_number':1,'registration_date':1})])
    assert validate(h).status=='review' and not h.rename_reliable

def test_region_classification_uses_spatial_title():
    from hukou.imaging import classify
    from hukou.ocr import Token
    def t(s,y): return Token([[0,y],[200,y],[200,y+20],[0,y+20]],s,1.)
    assert classify([t('家庭户口',0),t('户主姓名',30),t('户号',50),t('常住人口登记卡',900)])=='household_cover'
    assert classify([t('常住人口登记卡',0),t('户号',25),t('户主',60),t('姓名',90),t('出生',140),t('文化程度',200)])=='member_card'
