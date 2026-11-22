from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from hukou.runtime import choose_plan,GIB
from hukou.batch import batch,Control
from tests.test_delivery import FakePipeline
from openpyxl import load_workbook
from types import SimpleNamespace


def hardware(cores=4,ram=8,free=5,load=0):
    return dict(physical_cpus=cores,logical_cpus=cores*2,total_memory=ram*GIB,
                available_memory=free*GIB,cpu_percent=load)


def test_low_resource_and_automatic_load_limits():
    assert choose_plan(hardware=hardware()).workers==1
    assert choose_plan(hardware=hardware(16,32,3)).workers==1
    assert choose_plan(hardware=hardware(16,32,25,95)).workers==1
    low=choose_plan('low',workers=4,hardware=hardware(16,32,25))
    assert low.workers==1 and low.threads.onnx_intra==1 and low.threads.opencv==1
    custom=choose_plan('custom',workers=4,hardware=hardware())
    assert custom.workers==4 and custom.warning


def test_resume_rebuild_never_drops_later_completed_households(tmp_path):
    folder=tmp_path/'input';folder.mkdir();pipe=FakePipeline()
    for i in range(3):(folder/f'{i}.pdf').write_bytes(str(i).encode())
    first=batch(folder,pipeline=pipe)
    (folder/'3.pdf').write_bytes(b'new')
    control=Control();control.stopped.set()
    second=batch(folder,pipeline=pipe,control=control)
    assert second['resumed']==3 and second['processed']==3 and pipe.calls==3
    rows=json.loads((folder/'输出/results.json').read_text(encoding='utf-8'))
    assert len(rows)==3
    book=load_workbook(first['excel']);assert book.worksheets[0].max_row==4;book.close()


def test_insufficient_memory_is_reported_before_model_processes_start(tmp_path,monkeypatch):
    folder=tmp_path/'input';folder.mkdir();(folder/'1.pdf').write_bytes(b'synthetic')
    monkeypatch.setattr('psutil.virtual_memory',lambda:SimpleNamespace(total=8*GIB,available=64*1024**2))
    result=batch(folder)
    assert result['cancelled'] and result['processed']==0 and result['error']
    assert result['worker_runtime']=={}
