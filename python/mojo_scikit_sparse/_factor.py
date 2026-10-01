"""Sparse LDL^T factorization and the solves that go with it.

Mirrors the CHOLMOD workflow the Python package is modelled on: ``analyze`` does
the symbolic pass and sizes the factor, ``factorize`` does the numeric pass, and
``cholesky`` additionally requires the result to be positive definite.
"""

from __future__ import annotations

import numpy as np

from . import _lib
from ._matrix import _SparseMatrix, csc_matrix, csr_matrix


class NotPositiveDefiniteError(np.linalg.LinAlgError):
    """Raised when a factorization has a non-positive pivot."""


class NotSymmetricError(ValueError):
    """Raised when stored entries contradict each other across the diagonal."""


def _scratch(size, dtype=np.int64):
    """Kernels take non-null pointers, so never hand them a zero-length array."""
    return np.zeros(max(size, 1), dtype=dtype)


def _addr_keep(*arrays):
    """Addresses of scratch arrays, keeping the arrays alive.

    ``_lib.addr(_scratch(n))`` on its own would drop the last reference to the
    temporary before the kernel writes through the pointer.
    """
    return [a.ctypes.data for a in arrays]


def _empty_csr(n):
    return csr_matrix(
        (np.empty(0, np.float64), np.empty(0, np.int64), np.zeros(n + 1, np.int64)),
        shape=(n, n),
    )


def _empty_csc(n):
    return csc_matrix(
        (np.empty(0, np.float64), np.empty(0, np.int64), np.zeros(n + 1, np.int64)),
        shape=(n, n),
    )


def _as_csr(A):
    """Row-compressed view of A, oriented so the lower triangle is populated.

    The factorization reads the lower triangle and the diagonal, so a symmetric
    matrix handed over with only its upper triangle stored would otherwise
    factor as a diagonal matrix.  Transposing costs one pass and keeps that
    case honest.
    """
    if not isinstance(A, _SparseMatrix):
        raise TypeError(f"expected a sparse matrix, got {type(A).__name__}")
    Acsr = A.tocsr()
    n = Acsr.shape[0]
    if Acsr.nnz == 0:
        return Acsr
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(Acsr.indptr))
    above = int((rows < Acsr.indices).sum())
    below = int((rows > Acsr.indices).sum())
    if above and not below:
        return Acsr.T.tocsr()
    return Acsr


def _check_square_symmetric(A):
    n, m = A.shape
    if n != m:
        raise ValueError(f"expected a square matrix, got shape {A.shape}")
    if A.nnz == 0:
        return n
    indptr = A.indptr
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    cols = A.indices
    vals = A.data
    upper = rows < cols
    if not upper.any():
        return n
    keys = rows * n + cols
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    sorted_vals = vals[order]
    upper_vals = vals[upper]
    target = cols[upper] * n + rows[upper]
    pos = np.searchsorted(sorted_keys, target)
    idx = np.flatnonzero(pos < sorted_keys.size)
    hit = np.zeros(target.size, dtype=bool)
    match = np.zeros(target.size, dtype=bool)
    hit[idx] = sorted_keys[pos[idx]] == target[idx]
    match[idx] = sorted_vals[pos[idx]] == upper_vals[idx]
    if np.any(hit & ~match):
        raise NotSymmetricError(
            "matrix is not symmetric: an upper-triangle entry disagrees with "
            "its lower-triangle counterpart"
        )
    return n


class Analysis:
    """Symbolic result of :func:`analyze`: the sparsity of the factor."""

    def __init__(self, matrix, lnz, lp):
        self._matrix = matrix
        self.Lnz = lnz
        self.Lp = lp
        self.n = matrix.shape[0]

    @property
    def shape(self):
        return (self.n, self.n)

    @property
    def nnz_L(self):
        """Number of off-diagonal nonzeros in L."""
        return int(self.Lnz.sum())

    def factorize(self):
        return _factorize(self)


class Factor:
    """A numeric LDL^T factor ``A = L @ diag(D) @ L.T`` with unit-diagonal L."""

    def __init__(self, L, D):
        self._L = L
        self.D = D

    @property
    def L(self):
        """The strictly lower triangular part of L, column-compressed.

        The unit diagonal is implicit, so ``L.toarray() + eye(n)`` is the full
        factor.  Column-compressed is what both triangular solves want: the
        forward and backward substitutions are then the same scatter with the
        columns walked in opposite directions.
        """
        return self._L

    @property
    def shape(self):
        return self._L.shape

    @property
    def nnz(self):
        return self._L.nnz

    def _vector(self, b):
        arr = np.ascontiguousarray(np.asarray(b, dtype=np.float64).ravel())
        if arr.size != self.shape[0]:
            raise ValueError(
                f"expected a vector of length {self.shape[0]}, got {arr.size}"
            )
        return arr

    def solve_L(self, b):
        """Solve ``L y = b``."""
        b = self._vector(b)
        n = self.shape[0]
        y = np.zeros(n, dtype=np.float64)
        if n == 0 or self._L.nnz == 0:
            # An empty L is the identity, so there is nothing to substitute.
            return b.copy() if self._L.nnz == 0 else y
        _lib.lib().msp_csc_lower_solve(
            n,
            _lib.addr(self._L.indptr), _lib.addr(self._L.indices),
            _lib.addr(self._L.data), _lib.addr(b), _lib.addr(y),
        )
        return y

    def solve_Lt(self, b):
        """Solve ``L.T x = b``."""
        b = self._vector(b)
        n = self.shape[0]
        y = np.zeros(n, dtype=np.float64)
        if n == 0 or self._L.nnz == 0:
            return b.copy() if self._L.nnz == 0 else y
        _lib.lib().msp_csc_lower_transpose_solve(
            n,
            _lib.addr(self._L.indptr), _lib.addr(self._L.indices),
            _lib.addr(self._L.data), _lib.addr(b), _lib.addr(y),
        )
        return y

    def solve_A(self, b):
        """Solve ``A x = b`` using the stored factorization."""
        b = self._vector(b)
        n = self.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        y = self.solve_L(b)
        scaled = _scratch(n, np.float64)
        _lib.lib().msp_vec_scale_inv(
            n, _lib.addr(self.D), _lib.addr(y), _lib.addr(scaled)
        )
        return self.solve_Lt(scaled)

    def toarray(self):
        """Reconstruct ``L @ diag(D) @ L.T`` densely."""
        L = self._L.toarray() + np.eye(self.shape[0])
        return (L * self.D) @ L.T

    def __repr__(self):
        return f"<Factor {self.shape[0]}x{self.shape[1]} nnz={self.nnz}>"


def analyze(A):
    """Symbolic pass: sizes the factor without doing any arithmetic."""
    Acsr = _as_csr(A)
    n = _check_square_symmetric(Acsr)
    if n == 0:
        return Analysis(_empty_csr(0), np.zeros(0, np.int64), np.zeros(1, np.int64))
    lnz = np.zeros(n, dtype=np.int64)
    parent, mark = _scratch(n), _scratch(n)
    _lib.lib().msp_ldl_analyze(
        n,
        _lib.addr(Acsr.indptr), _lib.addr(Acsr.indices),
        *_addr_keep(parent, lnz, mark),
    )
    lp = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lnz, out=lp[1:])
    return Analysis(Acsr, lnz, lp)


def _factorize(analysis):
    Acsr = analysis._matrix
    n = analysis.n
    if n == 0:
        return Factor(_empty_csc(0), np.zeros(0, np.float64))
    nnz = analysis.nnz_L
    li = _scratch(nnz)
    lx = _scratch(nnz, np.float64)
    D = np.zeros(n, dtype=np.float64)
    nnz_out = np.zeros(1, dtype=np.int64)
    lp = analysis.Lp.copy()
    # Scratch must outlive the call: taking the address of a temporary would
    # let numpy free it while the kernel is still writing through the pointer.
    mark, val = _scratch(n), _scratch(n, np.float64)
    parent, pat, ptmp = _scratch(n), _scratch(n), _scratch(n)
    _lib.lib().msp_ldl_factor(
        n,
        _lib.addr(Acsr.indptr), _lib.addr(Acsr.indices), _lib.addr(Acsr.data),
        _lib.addr(lp), _lib.addr(li), _lib.addr(lx), _lib.addr(D),
        *_addr_keep(mark, val, parent, pat, ptmp, nnz_out),
    )
    actual = int(nnz_out[0])
    if actual != nnz:
        raise RuntimeError(
            f"symbolic pass predicted {nnz} factor nonzeros but the numeric "
            f"pass produced {actual}"
        )
    # The numeric pass writes L row by row; the triangular solves both want the
    # column-compressed form, so transpose once here.
    L = csr_matrix((lx[:actual], li[:actual], lp), shape=(n, n)).tocsc()
    return Factor(L, D)


def factorize(A):
    """Numeric LDL^T factorization of a symmetric sparse matrix.

    Only the lower triangle and the diagonal of ``A`` are read, so a matrix
    stored with just its lower triangle is fine.  Stored entries that straddle
    the diagonal must agree or :class:`NotSymmetricError` is raised.
    """
    return analyze(A).factorize()


def cholesky(A):
    """Like :func:`factorize`, but require ``A`` to be positive definite."""
    factor = factorize(A)
    if factor.D.size and not np.all(factor.D > 0.0):
        raise NotPositiveDefiniteError(
            "matrix is not positive definite: a pivot of L*D*L.T is non-positive"
        )
    return factor


def spsolve(A, b):
    """Solve ``A x = b`` for symmetric positive definite ``A``."""
    return cholesky(A).solve_A(b)


__all__ = [
    "Analysis",
    "Factor",
    "NotPositiveDefiniteError",
    "NotSymmetricError",
    "SparseFormatError",
    "analyze",
    "cholesky",
    "factorize",
    "spsolve",
]