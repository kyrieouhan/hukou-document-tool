"""Bounded per-household scheduling with one owner of outputs and checkpoints."""
from pathlib import Path
from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from concurrent.futures.process import BrokenProcessPool
import json, logging, multiprocessing, os, queue, threading, time
import psutil
from .pipeline import Pipeline, file_hash, Cancelled
from .config import Settings
from .models import Household, Issue
from .storage import atomic_json, copy_pdf, PrivacyFilter, safe_stem
from .exporter import export_excel
from .directory_lock import directory_locks
from .runtime import (choose_plan, configure_environment, keep_awake, GIB,
                      MIN_START_AVAILABLE_BYTES,MIN_RUNNING_AVAILABLE_BYTES)
from .workers import initialize_worker, recognize_pdf


WORKER_FAILURE = '识别进程异常退出，已完成结果已保存；请再次开始恢复。'


class Control:
    def __init__(self):
        self.paused=threading.Event(); self.stopped=threading.Event(); self.pause_acknowledged=threading.Event()

    def check(self):
        try:
            while self.paused.is_set():
                self.pause_acknowledged.set()
                if self.stopped.wait(.1): raise Cancelled()
        finally: self.pause_acknowledged.clear()
        if self.stopped.is_set(): raise Cancelled()


def review_field_count(results: list[Household]) -> int:
    """Count a field once even when several validation rules flag it."""
    return sum(len({(i.field,i.page,i.member) for i in h.issues}) for h in results)


def scan_pdfs(folder: Path,output: Path|None=None,recursive: bool=False) -> list[Path]:
    folder=folder.resolve()
    if not folder.is_dir(): raise ValueError('输入文件夹不存在')
    output=output.resolve() if output else folder/'输出'
    candidates=folder.rglob('*') if recursive else folder.iterdir()
    return sorted((p for p in candidates if p.is_file() and p.suffix.lower()=='.pdf' and
                   not p.is_symlink() and not p.resolve().is_relative_to(output) and
                   not any(x in ('已识别PDF','待复核PDF','输出') for x in p.relative_to(folder).parts[:-1])),
                  key=lambda p:str(p).casefold())


def batch(folder: Path,output: Path|None=None,naming: str='name',force: bool=False,
          control: Control|None=None,progress=lambda v:None,pipeline: Pipeline|None=None,
          recursive: bool=False,mode: str='auto',workers: int|None=None,
          settings: Settings|None=None) -> dict:
    """Workers recognize; the caller commits complete household results."""
    folder=folder.resolve(); output=(output or folder/'输出').resolve()
    if output==folder or folder.is_relative_to(output):
        raise ValueError('输出目录必须独立于输入目录，或使用输入目录下的“输出”子目录')
    files=scan_pdfs(folder,output,recursive)
    if not files: raise ValueError('输入文件夹中没有PDF')
    output.mkdir(parents=True,exist_ok=True)
    with directory_locks(folder,output),keep_awake():
        return _run(files,folder,output,naming,force,control or Control(),progress,pipeline,
                    recursive,mode,workers,settings or (pipeline.settings if pipeline else Settings()))


def _run(files,folder,output,naming,force,control,progress,pipeline,recursive,mode,workers,settings):
    started=time.perf_counter();plan=choose_plan(mode,workers)
    if pipeline is not None and workers not in (None,1):
        raise ValueError('注入的识别器只支持单进程测试')
    logger=logging.getLogger('hukou');logger.setLevel(logging.INFO)
    handler=logging.FileHandler(output/'logs/process.log',encoding='utf-8')
    handler.addFilter(PrivacyFilter());handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(handler)
    state_path=output/'state.json';excel=output/(safe_stem(folder.name)+'汇总.xlsx')
    signature=dict(input=str(folder),settings=settings.signature(),naming=naming,recursive=recursive)
    results={};resumed=0;cancelled=False;error='';excel_error='';durations=deque(maxlen=5)
    state={};pending=[];worker_runtimes={}

    def ordered():return [results[str(p)] for p in files if str(p) in results]

    def notify(stage,file='',**extra):
        rows=ordered();average=sum(durations)/len(durations) if durations else None
        active_workers=1 if pipeline else plan.workers
        remaining=len(files)-len(rows)
        progress(dict(index=len(rows),total=len(files),file=file,stage=stage,
            ok=sum(h.status=='ok' for h in rows),review=sum(h.status!='ok' for h in rows),
            review_fields=review_field_count(rows),workers=active_workers,
            average_seconds=average,eta_seconds=average*remaining/active_workers if len(durations)>=3 else None,
            plan=plan.to_dict(),**extra))

    def save():
        nonlocal excel_error
        rows=ordered()
        # Checkpoint is authoritative; JSON and Excel are reconstructible projections.
        atomic_json(state_path,dict(signature=signature,processed={h.source_pdf:h.to_dict() for h in rows}))
        atomic_json(output/'results.json',[h.to_dict() for h in rows])
        try:export_excel(rows,excel);excel_error=''
        except PermissionError:
            excel_error='Excel正在被占用，请关闭后再次开始；识别结果已保存，可断点续跑。'
            logger.warning('Excel file locked; checkpoint saved')

    def commit(h,path,record_duration=True):
        try:h.new_pdf=str(copy_pdf(h,output,naming))
        except Exception as exc:
            h.status='error';h.issues.append(Issue('page','PDF副本保存失败：'+type(exc).__name__))
            logger.error('copy failed type=%s',type(exc).__name__)
        results[str(path)]=h
        if record_duration:durations.append(h.elapsed_seconds)
        save()
        logger.info('completed=%d status=%s members=%d elapsed=%.2f',len(results),h.status,len(h.members),h.elapsed_seconds)
        notify('已保存',path.name)

    try:
        if state_path.exists() and not force:
            try:state=json.loads(state_path.read_text(encoding='utf-8'))
            except (ValueError,OSError):logger.warning('checkpoint unreadable; reprocessing')
            if state.get('signature')!=signature:state={}
        processed=state.get('processed',{})
        # Load all completed households before replacing any existing workbook.
        for path in files:
            digest=file_hash(path);previous=processed.get(str(path))
            if previous and previous.get('source_hash')==digest and previous.get('status') in ('ok','review'):
                try:
                    h=Household.from_dict(previous);h.new_pdf=str(copy_pdf(h,output,naming))
                    results[str(path)]=h;resumed+=1
                except (ValueError,TypeError,KeyError,OSError):
                    logger.warning('completed record or copy invalid; reprocessing');pending.append(path)
            else:pending.append(path)
        if results:save()
        notify('已恢复' if resumed else '准备识别')
        if pending and pipeline is None and psutil.virtual_memory().available<MIN_START_AVAILABLE_BYTES:
            error='可用内存不足，请关闭其他较大的程序后再次开始；已完成结果会保留。'
            cancelled=True
        elif pipeline is not None:
            for path in pending:
                try:
                    control.check();notify('识别',path.name)
                    h=pipeline.parse(path,control=control.check,
                        progress=lambda v,p=path:notify(v.get('stage','识别'),p.name,**{k:x for k,x in v.items() if k!='stage'}))
                except Cancelled:cancelled=True;break
                commit(h,path)
        elif pending and not control.stopped.is_set():
            configure_environment(plan.threads)
            ctx=multiprocessing.get_context('spawn');abort=ctx.Event();events=ctx.Queue(maxsize=plan.workers*16)
            todo=deque(pending);running={};parent=psutil.Process(os.getpid())
            pool=ProcessPoolExecutor(max_workers=plan.workers,mp_context=ctx,
                initializer=initialize_worker,initargs=(settings,plan.threads,abort,events,parent.pid,parent.create_time()))
            try:
                while todo or running:
                    if psutil.virtual_memory().available<MIN_RUNNING_AVAILABLE_BYTES:
                        error='系统可用内存过低，任务已安全停止；关闭其他较大的程序后可从断点继续。'
                        cancelled=True;abort.set();todo.clear()
                    if control.stopped.is_set():
                        cancelled=True;abort.set();todo.clear()
                    paused=control.paused.is_set() and not control.stopped.is_set()
                    if paused:control.pause_acknowledged.set()
                    else:control.pause_acknowledged.clear()
                    memory_ok=psutil.virtual_memory().available>=max(GIB,int(plan.hardware['total_memory']*.10))
                    capacity=plan.workers if memory_ok else max(1,len(running))
                    while todo and not paused and not control.stopped.is_set() and len(running)<capacity:
                        path=todo.popleft()
                        try:running[pool.submit(recognize_pdf,str(path))]=path
                        except Exception as exc:
                            error=WORKER_FAILURE
                            logger.error('worker submit failed type=%s',type(exc).__name__)
                            abort.set();todo.clear();cancelled=True;break
                        notify('识别',path.name)
                    while True:
                        try:value=events.get_nowait()
                        except queue.Empty:break
                        if 'runtime' in value:
                            runtime=value.pop('runtime');worker_runtimes[str(runtime['pid'])]=runtime
                            atomic_json(output/'logs/runtime.json',worker_runtimes)
                        path=Path(value.pop('path'));stage=value.pop('stage','识别')
                        notify(stage,path.name,**value)
                    if not running:
                        if paused:control.stopped.wait(.1);continue
                        break
                    done,_=wait(running,timeout=.1,return_when=FIRST_COMPLETED)
                    for future in done:
                        path=running.pop(future)
                        try:row=future.result()
                        except BrokenProcessPool:
                            error=WORKER_FAILURE
                            cancelled=True
                            abort.set()
                            todo.clear()
                            logger.error('recognition worker pool stopped unexpectedly')
                            continue
                        except Exception as exc:
                            # Per-file failure: record and continue the batch.
                            logger.error('worker failed type=%s',type(exc).__name__)
                            failed=Household(source_pdf=str(path),status='error')
                            failed.issues.append(Issue('page','PDF处理失败：'+type(exc).__name__))
                            commit(failed,path)
                            continue
                        if row is not None:commit(Household.from_dict(row),path)
            finally:
                abort.set()
                pool.shutdown(wait=True,cancel_futures=True)
                events.close();events.join_thread();control.pause_acknowledged.clear()
        elif pending:cancelled=True
        rows=ordered()
        summary=dict(total=len(files),processed=len(rows),resumed=resumed,cancelled=cancelled,
            ok=sum(h.status=='ok' for h in rows),review=sum(h.status!='ok' for h in rows),
            excel=str(excel),output=str(output),error=error or excel_error,
            review_fields=review_field_count(rows),seconds=time.perf_counter()-started,
            plan=plan.to_dict(),workers=1 if pipeline else plan.workers,worker_runtime=worker_runtimes)
        atomic_json(output/'logs/run_summary.json',summary)
        return summary
    finally:
        control.pause_acknowledged.clear();logger.removeHandler(handler);handler.close()
