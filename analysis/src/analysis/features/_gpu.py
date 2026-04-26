"""Shared GPU lock. CUDA modules acquire; CPU paths skip."""
import threading
from contextlib import contextmanager


class GpuLock:
    def __init__(self) -> None:
        self._lock = threading.Lock()

    @contextmanager
    def acquire(self):
        self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()


gpu_lock = GpuLock()


def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False
