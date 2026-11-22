"""Cross-process Windows locks released by the OS, including after a crash."""
from contextlib import ExitStack, contextmanager
from pathlib import Path
import hashlib, os, tempfile


@contextmanager
def file_lock(path: Path):
    """Lock a byte without relying on stale PID files or deleting another lock."""
    import msvcrt
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open('a+b')
    acquired = False
    try:
        if path.stat().st_size == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            acquired = True
        except OSError as exc:
            raise RuntimeError('此任务目录正在由另一个程序处理。') from exc
        yield
    finally:
        if acquired:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        stream.close()


@contextmanager
def directory_locks(folder: Path, output: Path):
    """Reserve both input and output; also respect the previous release's lock."""
    root = Path(os.environ.get('LOCALAPPDATA', tempfile.gettempdir()))/'HukouTool/locks'
    keys = sorted({str(p.resolve()).casefold() for p in (folder, output)})
    with ExitStack() as stack:
        for key in keys:
            digest = hashlib.sha256(key.encode('utf-8')).hexdigest()
            stack.enter_context(file_lock(root/(digest+'.lock')))
        stack.enter_context(file_lock(output/'logs/run.lock'))
        yield
