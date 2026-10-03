"""ctypes bindings for the Mojo kernels in ``src/ported.mojo``.

Every kernel takes buffer addresses as plain integers and reads/writes memory
owned by numpy arrays on this side; nothing crosses the ABI by pointer type.

The invariants that keep that boundary safe live here rather than in the
kernels:

* **Index arrays are int64.**  The kernels are ports of CHOLMOD's
  ``cholmod_l_*`` (long) entry points and read every index as an ``Int``, which
  is 8 bytes.  scipy hands back ``int32`` index arrays for small matrices, so
  :func:`i64` widens them.  Passing an ``int32`` buffer straight through reads
  half of every entry as garbage and segfaults; this is the single most likely
  way to misuse the library.
* **Values are float64.**  The kernels read every value as a ``Float64``, so
  :func:`f64` widens lower-precision input and refuses complex input instead of
  letting numpy reinterpret the bits and silently produce a wrong factor.
* **Pointers are non-nullable.**  A zero-length numpy array still has a valid
  address, but the kernels take addresses as ``Int`` and rebuild the pointer
  inside, so an empty buffer is simply never handed over: callers short-circuit
  before calling in.
* **Every buffer must be a named object.**  ctypes does not keep a reference to
  a numpy array passed as an argument, so an array built inline (a temporary) can
  be freed before the kernel reads it, which segfaults.  Allocate scratch into a
  local or an object attribute first, then pass its address.
* **A library that does not export every kernel is rejected at load time**, so a
  stale build fails with a clear message instead of an ``AttributeError``
  halfway through a factorization.
"""

from __future__ import annotations

import ctypes
import os
import threading
from pathlib import Path

import numpy as np

_NAME = "libmojo-scikit-sparse.so"


def _candidate_paths():
    """Where the built kernel library may live, in priority order.

    ``MOJO_SCIKIT_SPARSE_LIB`` overrides everything (useful when the library is
    staged somewhere else).  Otherwise: next to the package (which is where a
    wheel or an editable install puts it), then the repository's ``dist/``
    directory, which is where ``pixi run build`` writes it from a source
    checkout.
    """
    override = os.environ.get("MOJO_SCIKIT_SPARSE_LIB")
    if override:
        yield Path(override)
    here = Path(__file__).resolve().parent
    yield here / _NAME                     # installed package directory
    # <repo>/python/mojo_scikit_sparse -> <repo>/dist
    yield here.parent.parent / "dist" / _NAME


_LIB_PATH = next((p for p in _candidate_paths() if p.exists()), _candidate_paths().__next__())

_lock = threading.Lock()
_lib: ctypes.CDLL | None = None

_i64 = ctypes.c_int64
_f64 = ctypes.c_double
_p = ctypes.c_void_p

# name -> (argtypes, restype)
#
# Every list is positional and must match the Mojo parameter list exactly; an
# off-by-one here silently shifts arguments rather than failing loudly.
_SIGNATURES: dict[str, tuple[list, type | None]] = {
    # cholmod_transpose
    "msp_transpose": ([_i64, _i64, _p, _p, _p, _p, _p, _p, _i64, _i64, _p], None),
    # cholmod_etree
    "msp_etree": ([_i64, _i64, _p, _p, _p, _p, _p, _i64], None),
    # cholmod_postorder
    "msp_postorder": ([_i64, _p, _p, _p, _p, _p], None),
    # cholmod_rowcolcounts
    "msp_rowcolcounts": (
        [_i64, _i64, _p, _p, _p, _p, _p, _p, _p, _p, _p, _p, _p, _p, _p, _p, _i64, _p],
        None,
    ),
    # cholmod_row_subtree unrolled over every k, writing the pattern of L
    "msp_row_pattern": (
        [_i64, _p, _p, _p, _p, _p, _p, _p, _p, _p, _p, _i64],
        None,
    ),
    # cholmod_rowfac
    "msp_rowfac": (
        [_i64, _f64, _p, _p, _p, _p, _p, _p, _p, _p, _p, _p, _p, _p, _p, _i64, _p, _p, _i64],
        None,
    ),
    # cholmod_lsolve / cholmod_ltsolve
    "msp_lsolve": ([_i64, _p, _p, _p, _p, _p, _i64, _i64], None),
    "msp_ltsolve": ([_i64, _p, _p, _p, _p, _p, _i64], None),
    # csc_matvec (sparse matrix-vector product, used by the unsymmetric path)
    "msp_csc_matvec": ([_i64, _i64, _p, _p, _p, _p, _p], None),
    # the arithmetic the solve driver needs, kept in the same C ABI
    "msp_vec_scale_inv": ([_i64, _p, _p, _p], None),
    "msp_vec_permute": ([_i64, _p, _p, _p], None),
    "msp_vec_unpermute": ([_i64, _p, _p, _p], None),
}


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
                missing = [n for n in _SIGNATURES if not hasattr(cdll, n)]
                if missing:
                    raise ImportError(
                        f"{_LIB_PATH} is stale: it does not export {sorted(missing)}; "
                        "rebuild it with `pixi run build`"
                    )
                for name, (argtypes, restype) in _SIGNATURES.items():
                    fn = getattr(cdll, name)
                    fn.argtypes = argtypes
                    fn.restype = restype
                _lib = cdll
    return _lib


def addr(a: np.ndarray) -> int:
    """Address of a C-contiguous array's first element."""
    return a.ctypes.data


def i64(a) -> np.ndarray:
    """Contiguous int64 view of an index array.

    The kernels read indices as 8-byte ``Int``; scipy's index arrays are int32
    for matrices below 2**31, so they must be widened before crossing the ABI.
    The result is a fresh array when a conversion is needed and the input itself
    when it is already int64 and contiguous, so the common case costs nothing.
    """
    return np.ascontiguousarray(a, dtype=np.int64)


def f64(a) -> np.ndarray:
    """Contiguous float64 view of a value array.

    Raises on any dtype it cannot widen without inventing values.  numpy's
    ``ascontiguousarray`` would happily hand back a float64 *reinterpretation* of
    complex64 data (with a ``ComplexWarning``) and discard the imaginary part of
    complex128, which is a silently wrong factor rather than an error, so the
    check is explicit here instead.
    """
    a = np.asarray(a)
    if a.dtype.kind == "c":
        raise ValueError(
            "This port factors real float64 data; got complex input. "
            "See README, 'Coverage'."
        )
    return np.ascontiguousarray(a, dtype=np.float64)


def zeros_i(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.int64)


def zeros_f(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.float64)


def scalar_i() -> np.ndarray:
    return np.zeros(1, dtype=np.int64)


def scalar_f() -> np.ndarray:
    return np.zeros(1, dtype=np.float64)