"""Crash-sensitive behavior without touching customer data."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from hukou.directory_lock import directory_locks
from hukou.storage import copy_pdf
from hukou.pipeline import file_hash
from hukou.models import Household


def test_same_input_or_output_cannot_be_reserved_twice(tmp_path):
    a,b,c,d=[tmp_path/x for x in 'abcd']
    for p in (a,b,c,d):p.mkdir()
    with directory_locks(a,b):
        with pytest.raises(RuntimeError), directory_locks(a,c):pass
        with pytest.raises(RuntimeError), directory_locks(c,b):pass
        with directory_locks(c,d):pass
    with directory_locks(a,b):pass


def test_interrupted_copy_never_exposes_a_pdf(tmp_path,monkeypatch):
    source=tmp_path/'source.pdf';source.write_bytes(b'original synthetic document')
    h=Household(source_pdf=str(source),source_hash=file_hash(source))
    def fail(src,dst):
        dst.write(b'partial');raise OSError('synthetic disk interruption')
    monkeypatch.setattr('hukou.storage.shutil.copyfileobj',fail)
    with pytest.raises(OSError):copy_pdf(h,tmp_path/'output')
    assert not list((tmp_path/'output').rglob('*.pdf'))
    assert not list((tmp_path/'output').rglob('*.tmp'))
    assert file_hash(source)==h.source_hash


def test_copy_rejects_changed_source_even_if_previous_copy_exists(tmp_path):
    source=tmp_path/'source.pdf';source.write_bytes(b'original')
    h=Household(source_pdf=str(source),source_hash=file_hash(source))
    target=copy_pdf(h,tmp_path/'output')
    source.write_bytes(b'new document')
    with pytest.raises(OSError):copy_pdf(h,tmp_path/'output')
    assert target.read_bytes()==b'original'
