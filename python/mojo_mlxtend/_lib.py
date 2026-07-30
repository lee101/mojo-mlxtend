"""ctypes bridge to the compiled Mojo kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_MLXTEND_LIB", os.path.join(ROOT, "dist", "libmojo-mlxtend.so"))

I = ctypes.c_int64

_SIGNATURES = {
    "mmlx_count_candidates": ([I] * 8, I),
    "mmlx_intersect_count": ([I] * 9, I),
    "mmlx_rule_metrics": ([I] * 4, I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_MLXTEND_LIB"):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(f"MOJO_MLXTEND_LIB does not exist: {LIB}")
    sources = [
        os.path.join(path, name)
        for path, _, names in os.walk(os.path.join(ROOT, "src"))
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB) and sources:
        if os.path.getmtime(LIB) >= max(os.path.getmtime(source) for source in sources):
            return LIB
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_LIBRARY: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_LIBRARY, name)
            function.argtypes = argtypes
            function.restype = restype
    return _LIBRARY


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray) or not array.flags.c_contiguous:
        raise TypeError("FFI buffers must be C-contiguous NumPy arrays")
    if array.size == 0:
        raise ValueError("empty arrays do not have an FFI buffer")
    return int(array.ctypes.data)


def check_status(operation: str, status: int) -> None:
    if status:
        raise RuntimeError(f"Mojo kernel {operation} failed with status {status}")
