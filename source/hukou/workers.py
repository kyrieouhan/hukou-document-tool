"""One process owns one OCR pipeline; only complete households cross the boundary."""
from pathlib import Path
import os, queue, threading, time
from .config import Settings
from .runtime import ThreadSettings, configure_runtime

_pipeline = None
_cancel = None
_events = None


def initialize_worker(settings: Settings, threads: ThreadSettings, cancel, events,
                      parent_pid: int, parent_created: float) -> None:
    """Initialize local models once and exit if the owning GUI is forcibly closed."""
    global _pipeline, _cancel, _events
    import psutil
    from .offline import install_guard
    from .pipeline import Pipeline
    install_guard(); configure_runtime(threads)
    _cancel, _events = cancel, events
    _events.cancel_join_thread()  # Progress may be dropped on exit; results never use this queue.

    def watch_parent():
        while True:
            try:
                parent = psutil.Process(parent_pid)
                if parent.create_time() != parent_created: os._exit(1)
            except psutil.Error:
                os._exit(1)
            time.sleep(2)
    threading.Thread(target=watch_parent, daemon=True).start()
    _pipeline = Pipeline(settings)
    import cv2
    engine=_pipeline.engine.impl;reader=_pipeline.field_reader
    sessions=[engine.text_det.infer.session,engine.text_cls.infer.session,engine.text_rec.session.session,
              reader.precision.impl.session.session,reader.alternate.impl.session.session,reader.numeric.impl.session.session]
    runtime=dict(pid=os.getpid(),opencv_threads=cv2.getNumThreads(),
        sessions=[dict(providers=s.get_providers(),intra=s.get_session_options().intra_op_num_threads,
                       inter=s.get_session_options().inter_op_num_threads) for s in sessions],
        native_threads=psutil.Process().num_threads(),affinity=psutil.Process().cpu_affinity(),
        blas_threads=os.environ.get('OPENBLAS_NUM_THREADS'))
    try:_events.put_nowait(dict(path='',stage='本地模型已加载',runtime=runtime))
    except queue.Full:pass


def recognize_pdf(path: str) -> dict | None:
    from .pipeline import Cancelled
    def check():
        if _cancel.is_set(): raise Cancelled()
    def progress(value):
        try: _events.put_nowait(dict(path=path, **value))
        except queue.Full: pass  # Optional UI progress; completed data uses the future.
    try:
        check()
        return _pipeline.parse(Path(path), control=check, progress=progress).to_dict()
    except Cancelled:
        return None
    except Exception as exc:
        # One bad PDF must never abort the batch or kill the worker pool.
        from .models import Household,Issue
        h=Household(source_pdf=str(path))
        h.status='error'
        h.issues.append(Issue('page','PDF处理失败：'+type(exc).__name__))
        return h.to_dict()
