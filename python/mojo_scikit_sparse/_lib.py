"""ctypes bindings for the Mojo kernels in ``src/mojo_sparse.mojo``.

Every kernel takes buffer addresses as plain integers and reads/writes memory
owned by numpy arrays on this side; nothing crosses the ABI by pointer type.
"""

from __future__ import annotations

import ctypes
import threading
from pathlib import Path

import numpy as np

_LIB_PATH = Path(__file__).resolve().parent.parent / "_mojo_sparse.so"

_lock = threading.Lock()
_lib: ctypes.CDLL | None = None

_i64 = ctypes.c_int64
_p = ctypes.c_void_p


def library_path() -> Path:
    return _LIB_PATH


def lib() -> ctypes.CDLL:
    """Load and declare the shared library on first use."""
    global _lib
    if _lib is None:
        with _lock:
            if _lib is None:
                if not _LIB_PATH.exists():
                    raise ImportError(
                        f"{_LIB_PATH} is missing; run `pixi run build` first"
                    )
                cdll = ctypes.CDLL(str(_LIB_PATH))
                _declare(cdll)
                _lib = cdll
    return _lib


def _declare(cdll: ctypes.CDLL) -> None:
    cdll.msp_csc_matvec.argtypes = [_i64] * 7
    cdll.msp_csr_matvec.argtypes = [_i64] * 7
    cdll.msp_csc_transpose.argtypes = [_i64] * 9
    cdll.msp_csc_lower_solve.argtypes = [_i64] * 6
    cdll.msp_csc_lower_transpose_solve.argtypes = [_i64] * 6
    cdll.msp_vec_scale_inv.argtypes = [_i64] * 4
    cdll.msp_ldl_analyze.argtypes = [_i64] * 6
    cdll.msp_ldl_factor.argtypes = [_i64] * 14
    for name in (
        "msp_csc_matvec",
        "msp_csr_matvec",
        "msp_csc_transpose",
        "msp_csc_lower_solve",
        "msp_csc_lower_transpose_solve",
        "msp_vec_scale_inv",
        "msp_ldl_analyze",
        "msp_ldl_factor",
    ):
        getattr(cdll, name).restype = None


def addr(a: np.ndarray) -> int:
    """Address of a C-contiguous array's first element.

    Kernels take non-nullable pointers, so a zero-length array has no address
    that can legally be handed over; callers must short circuit before here.
    """
    if a.size == 0:
        raise ValueError("cannot pass an empty buffer to a Mojo kernel")
    return a.ctypes.data
