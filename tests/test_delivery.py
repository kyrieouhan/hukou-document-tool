# The numeric examples use a synthetic region prefix outside the mainland code ranges.
from pathlib import Path
import sys,json,logging
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from hukou.models import Household,Member
from hukou.pipeline import file_hash
from hukou.config import Settings
from hukou.storage import copy_pdf,safe_stem,atomic_json,PrivacyFilter
from hukou.exporter import export_excel,HEADERS
from hukou.batch import batch,Control,scan_pdfs
from openpyxl import load_workbook

class FakePipeline:
    settings=Settings()
    calls=0
    def parse(self,path,**kwargs):
        self.calls+=1
        return Household(source_pdf=str(path),source_hash=file_hash(path),householder='测试户主',household_no='000123',
                         members=[Member(name='测试成员',id_number='00990419491231002X',registration_date='2024-11-13')],
                         family_size=1,page_count=2,status='ok',rename_reliable=True)

def test_excel_exact_text_and_formula_safety(tmp_path):
    path=tmp_path/'中文汇总.xlsx'
    h=Household(household_no='0000123',householder='=HYPERLINK("x")',source_pdf='001.pdf',family_size=1,
                members=[Member(name='虚构测试',relation='曾孙女或外曾孙女',id_number='00990419491231002X',registration_date='2024-11-13')])
    export_excel([h],path)
    w=load_workbook(path)
    assert w.sheetnames==['家庭成员汇总','待复核','处理日志']
    s=w.worksheets[0]
    assert [c.value for c in s[1]]==HEADERS
    assert s['F2'].value=='00990419491231002X' and s['F2'].number_format=='@' and s['F2'].data_type=='s'
    assert s['A2'].value=='0000123' and s['B2'].data_type=='s'
    assert s['G2'].value=='2024-11-13' and s.freeze_panes=='A2'

def test_copy_collisions_and_preservation(tmp_path):
    a=tmp_path/'1.pdf'; b=tmp_path/'2.pdf'; a.write_bytes(b'alpha'); b.write_bytes(b'beta')
    pipe=FakePipeline(); h1=pipe.parse(a); h2=pipe.parse(b)
    first=copy_pdf(h1,tmp_path/'out'); second=copy_pdf(h2,tmp_path/'out')
    assert first!=second and first.read_bytes()==b'alpha' and second.read_bytes()==b'beta'
    assert a.read_bytes()==b'alpha' and b.read_bytes()==b'beta'
    assert copy_pdf(h1,tmp_path/'out')==first
    assert safe_stem('CON')=='_CON' and '/' not in safe_stem('a/b')

def test_resume_changed_input_and_force(tmp_path):
    folder=tmp_path/'测试输入'; folder.mkdir(); (folder/'1.pdf').write_bytes(b'test')
    p=FakePipeline()
    assert batch(folder,pipeline=p)['processed']==1 and p.calls==1
    assert batch(folder,pipeline=p)['resumed']==1 and p.calls==1
    (folder/'1.pdf').write_bytes(b'changed')
    assert batch(folder,pipeline=p)['resumed']==0 and p.calls==2
    batch(folder,pipeline=p,force=True); assert p.calls==3
    assert len(scan_pdfs(folder,recursive=True))==1

def test_stop_and_checkpoint(tmp_path):
    folder=tmp_path/'输入'; folder.mkdir()
    for i in range(3): (folder/f'{i}.pdf').write_bytes(str(i).encode())
    control=Control(); pipe=FakePipeline()
    def progress(v):
        if v.get('index')==1 and v.get('stage')=='已保存': control.stopped.set()
    result=batch(folder,pipeline=pipe,control=control,progress=progress)
    assert result['cancelled'] and result['processed']==1
    assert json.loads((folder/'输出/state.json').read_text(encoding='utf-8'))['processed']
    assert batch(folder,pipeline=pipe)['resumed']==1

def test_log_mask():
    r=logging.LogRecord('x',20,'',1,'ID=%s',('00990419491231002X',),None)
    PrivacyFilter().filter(r)
    assert '00990419491231002X' not in r.getMessage() and '002X' in r.getMessage()
