"""Exercise the real process scheduler with cheap synthetic recognition."""
from pathlib import Path
import json,sys,time,threading
import os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import hukou.batch as batch_module
from hukou.batch import batch,Control
from hukou.models import Household,Member
from hukou.pipeline import file_hash
from openpyxl import load_workbook

STOP=None


def initialize_stub(settings,threads,cancel,events,parent_pid,parent_created):
    global STOP
    STOP=cancel


def recognize_stub(path):
    for _ in range(10):
        if STOP.is_set():return None
        time.sleep(.03)
    return Household(source_pdf=path,source_hash=file_hash(Path(path)),householder='测试户主',
        household_no='000001',members=[Member(name='测试成员',relation='户主')],family_size=1,
        page_count=1,status='review',elapsed_seconds=.3).to_dict()


def recognize_crashing_stub(path):
    if Path(path).stem=='002':os._exit(7)
    return recognize_stub(path)


def prepare(tmp_path,monkeypatch,count=8):
    monkeypatch.setattr(batch_module,'initialize_worker',initialize_stub)
    monkeypatch.setattr(batch_module,'recognize_pdf',recognize_stub)
    folder=tmp_path/'中文输入';folder.mkdir()
    for i in range(count):(folder/f'{i:03}.pdf').write_bytes(f'synthetic {i}'.encode())
    return folder


def test_parallel_pause_stops_dispatch_and_resume_preserves_all(tmp_path,monkeypatch):
    folder=prepare(tmp_path,monkeypatch);control=Control();dispatch=[];paused=[]
    def progress(v):
        if v['stage']=='识别':dispatch.append(v['file'])
        if v['stage']=='已保存' and not paused:
            paused.append(True);control.paused.set()
            before=len(dispatch)
            def resume():
                paused.append(len(dispatch)==before)
                paused.append(control.pause_acknowledged.is_set())
                control.paused.clear()
            threading.Timer(.6,resume).start()
    result=batch(folder,workers=2,control=control,progress=progress)
    assert result['processed']==8 and paused==[True,True,True]
    assert len(dispatch)==8 and len(set(dispatch))==8
    records=json.loads((folder/'输出/state.json').read_text(encoding='utf-8'))['processed']
    assert len(records)==8
    workbook=load_workbook(result['excel']);assert workbook.worksheets[0].max_row==9;workbook.close()
    assert batch(folder,workers=2)['resumed']==8


def test_parallel_stop_commits_complete_results_and_restarts(tmp_path,monkeypatch):
    folder=prepare(tmp_path,monkeypatch);control=Control()
    def progress(v):
        if v['stage']=='已保存':control.stopped.set()
    partial=batch(folder,workers=2,control=control,progress=progress)
    assert partial['cancelled'] and 1<=partial['processed']<=2
    final=batch(folder,workers=2)
    assert final['processed']==8 and final['resumed']==partial['processed']
    rows=json.loads((folder/'输出/results.json').read_text(encoding='utf-8'))
    assert len(rows)==8 and all(file_hash(Path(h['new_pdf']))==h['source_hash'] for h in rows)


def test_worker_process_crash_is_reported_and_completed_households_resume(tmp_path,monkeypatch):
    folder=prepare(tmp_path,monkeypatch)
    monkeypatch.setattr(batch_module,'recognize_pdf',recognize_crashing_stub)
    interrupted=batch(folder,workers=2)
    assert interrupted['error'] and interrupted['cancelled'] and 1<=interrupted['processed']<8
    monkeypatch.setattr(batch_module,'recognize_pdf',recognize_stub)
    restored=batch(folder,workers=2)
    assert restored['processed']==8 and restored['resumed']==interrupted['processed']
