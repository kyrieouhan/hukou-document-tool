"""Central CPU-only execution policy; never changes recognition parameters."""
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import os
import psutil

GIB = 1024**3


@dataclass(frozen=True)
class ThreadSettings:
    onnx_intra: int = 2
    onnx_inter: int = 1
    opencv: int = 1
    blas: int = 1


_threads = ThreadSettings()


def thread_settings() -> ThreadSettings:
    return _threads


def configure_environment(settings: ThreadSettings) -> None:
    """Set before spawning workers so native libraries inherit limits on import."""
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(settings.blas)
    os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'


def configure_runtime(settings: ThreadSettings) -> None:
    global _threads
    _threads = settings
    configure_environment(settings)
    import cv2
    cv2.setNumThreads(settings.opencv)


def hardware_snapshot(sample_seconds: float = 0.) -> dict:
    memory = psutil.virtual_memory()
    return dict(physical_cpus=psutil.cpu_count(logical=False) or 1,
                logical_cpus=psutil.cpu_count() or 1,
                total_memory=memory.total, available_memory=memory.available,
                cpu_percent=psutil.cpu_percent(interval=sample_seconds or None))


@dataclass(frozen=True)
class ExecutionPlan:
    workers: int
    threads: ThreadSettings
    mode: str
    hardware: dict
    recommended: int
    warning: str = ''

    def to_dict(self) -> dict:
        return asdict(self)


# Measured workers peaked near 1 GiB on the 22-page benchmark; reserve 2 GiB each.
# Three processes beat four on that mixed workload. Unmeasured higher counts are not defaults.
WORKER_MEMORY_BYTES = 2 * GIB
BENCHMARK_WORKER_CAP = 3
MIN_START_AVAILABLE_BYTES = 5 * GIB // 2
MIN_RUNNING_AVAILABLE_BYTES = GIB // 2


@contextmanager
def keep_awake():
    """Prevent automatic system sleep during a batch without changing power plans."""
    import ctypes
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:yield
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def choose_plan(mode: str = 'auto', workers: int | None = None,
                hardware: dict | None = None) -> ExecutionPlan:
    """Respect free RAM and current load rather than merely dividing CPU threads."""
    if mode not in ('auto', 'low', 'standard', 'high', 'custom'):
        raise ValueError('未知性能模式')
    if workers is not None and not 1 <= workers <= 6:
        raise ValueError('同时处理户数须为1至6')
    hw = hardware or hardware_snapshot(.15)
    reserve = max(2 * GIB, int(hw['total_memory'] * .20))
    memory_limit = max(1, int((hw['available_memory'] - reserve) // WORKER_MEMORY_BYTES))
    cpu_limit = max(1, hw['physical_cpus'] // 3)
    recommended = max(1, min(BENCHMARK_WORKER_CAP, memory_limit, cpu_limit))
    if hw['cpu_percent'] >= 80: recommended = 1
    elif hw['cpu_percent'] >= 55: recommended = max(1, recommended // 2)
    selected = recommended
    if mode == 'low': selected = 1
    elif mode == 'standard': selected = min(2, recommended)
    elif mode == 'high': selected = min(BENCHMARK_WORKER_CAP, memory_limit, cpu_limit)
    if workers is not None and mode != 'low': selected = workers
    warning = ''
    if selected > recommended:
        warning = f'当前资源建议同时处理{recommended}户，已按您的设置使用{selected}户。'
    if hw['available_memory'] < reserve + WORKER_MEMORY_BYTES:
        warning = '当前可用内存较少，建议关闭其他较大的程序，使用低资源模式。'
    return ExecutionPlan(selected, ThreadSettings(onnx_intra=1 if mode == 'low' else 2),
                         mode, hw, recommended, warning)
