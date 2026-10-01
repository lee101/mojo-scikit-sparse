"""Sparse matrix containers backed by the Mojo kernels.

Only float64 is supported: the kernels read and write Float64 buffers directly.
Compressed indices are canonicalised on construction (sorted, duplicates summed),
which is the form every kernel here assumes.
"""

from __future__ import annotations

import numpy as np

from . import _lib


class SparseFormatError(ValueError):
    """Raised when index arrays do not describe a valid compressed matrix."""


def _as_1d(a, dtype, name):
    out = np.asarray(a)
    if out.ndim != 1:
        raise SparseFormatError(f"{name} must be 1-D, got shape {out.shape}")
    return np.ascontiguousarray(out, dtype=dtype)


def _canonicalize(data, indices, indptr):
    """Sort each compressed segment and sum duplicate entries.

    Returns fresh (data, indices, indptr); callers own the result.
    """
    nnz = data.size
    if nnz == 0:
        return data, indices, indptr
    seg = np.repeat(np.arange(indptr.size - 1), np.diff(indptr))
    order = np.lexsort((indices, seg))
    data = data[order]
    indices = indices[order]
    seg = seg[order]
    starts = np.empty(nnz, dtype=bool)
    starts[0] = True
    np.logical_or(seg[1:] != seg[:-1], indices[1:] != indices[:-1], out=starts[1:])
    run_start = np.flatnonzero(starts)
    data = np.add.reduceat(data, run_start)
    indices = indices[starts]
    # Each run belongs to exactly one segment, and runs stay in segment order,
    # so a per-segment histogram rebuilds indptr.
    counts = np.bincount(seg[run_start], minlength=indptr.size - 1)
    new_indptr = np.zeros(indptr.size, dtype=np.int64)
    np.cumsum(counts, out=new_indptr[1:])
    indptr = new_indptr
    return data, indices, indptr


def _from_dense(dense, compressed_axis):
    """Build a compressed triple from a dense 2-D array.

    ``np.nonzero`` enumerates in row-major order, which is only the layout the
    CSR form wants; canonicalising regroups the entries by the compressed axis.
    """
    dense = np.ascontiguousarray(np.asarray(dense, dtype=np.float64))
    if dense.ndim != 2:
        raise ValueError(f"expected a 2-D dense array, got shape {dense.shape}")
    m, n = dense.shape
    rows, cols = np.nonzero(dense)
    major, minor = (rows, cols) if compressed_axis == 0 else (cols, rows)
    ncomp = m if compressed_axis == 0 else n
    data = np.ascontiguousarray(dense[rows, cols], dtype=np.float64)
    indices = np.ascontiguousarray(minor, dtype=np.int64)
    # np.nonzero walks in row-major order, which is the layout CSR wants but not
    # the one CSC wants, so regroup by the compressed axis before counting.
    order = np.lexsort((indices, major))
    data = np.ascontiguousarray(data[order])
    indices = np.ascontiguousarray(indices[order])
    indptr = np.zeros(ncomp + 1, dtype=np.int64)
    if major.size:
        np.cumsum(np.bincount(major, minlength=ncomp), out=indptr[1:])
    return (data, indices, indptr), (m, n)


class _SparseMatrix:
    """Shared behaviour of the compressed formats.

    ``_compressed_axis`` is 0 for CSR (rows are compressed) and 1 for CSC.
    """

    _compressed_axis = 0

    def __init__(self, arg1, shape=None):
        if not (isinstance(arg1, tuple) and len(arg1) == 3):
            raise TypeError(
                "expected a (data, indices, indptr) triple; "
                "build from a dense array with csr_matrix(dense) or csc_matrix(dense)"
            )
        if shape is None:
            raise ValueError("shape is required for compressed input")
        if len(shape) != 2:
            raise ValueError(f"shape must be 2-D, got {shape!r}")
        m, n = int(shape[0]), int(shape[1])
        if m < 0 or n < 0:
            raise ValueError(f"negative dimension in {shape!r}")
        data, indices, indptr = (
            _as_1d(arg1[0], np.float64, "data"),
            _as_1d(arg1[1], np.int64, "indices"),
            _as_1d(arg1[2], np.int64, "indptr"),
        )
        ncomp = m if self._compressed_axis == 0 else n
        nminor = n if self._compressed_axis == 0 else m
        if indptr.size != ncomp + 1:
            raise SparseFormatError(
                f"indptr must have {ncomp + 1} entries, got {indptr.size}"
            )
        if indptr[0] != 0:
            raise SparseFormatError("indptr[0] must be 0")
        if np.any(np.diff(indptr) < 0):
            raise SparseFormatError("indptr must be non-decreasing")
        nnz = int(indptr[-1])
        if indices.size < nnz:
            raise SparseFormatError(
                f"indices has {indices.size} entries but indptr points to {nnz}"
            )
        data, indices = data[:nnz], indices[:nnz]
        if indices.size and (indices.min() < 0 or indices.max() >= nminor):
            raise SparseFormatError(f"indices out of range [0, {nminor})")
        data, indices, indptr = _canonicalize(data, indices, indptr)
        self._shape = (m, n)
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def copy(self):
        return type(self)(
            (self.data.copy(), self.indices.copy(), self.indptr.copy()),
            shape=self._shape,
        )

    # -- basic properties ----------------------------------------------------

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return 2

    @property
    def dtype(self):
        return np.dtype(np.float64)

    @property
    def nnz(self):
        return int(self.indptr[-1])

    def __len__(self):
        return self._shape[0]

    # -- conversions ---------------------------------------------------------

    def toarray(self):
        m, n = self._shape
        out = np.zeros((m, n), dtype=np.float64)
        for k in range(self.indptr.size - 1):
            lo, hi = int(self.indptr[k]), int(self.indptr[k + 1])
            if self._compressed_axis == 0:
                out[k, self.indices[lo:hi]] = self.data[lo:hi]
            else:
                out[self.indices[lo:hi], k] = self.data[lo:hi]
        return out

    def _transposed_csc(self, rows, cols):
        """CSC of A^T for the compressed buffers of an A with `rows` x `cols`."""
        nnz = self.nnz
        bp = np.zeros(rows + 1, dtype=np.int64)
        if nnz == 0:
            return (np.empty(0, np.float64), np.empty(0, np.int64), bp)
        bi = np.empty(nnz, dtype=np.int64)
        bx = np.empty(nnz, dtype=np.float64)
        work = np.zeros(rows, dtype=np.int64)
        _lib.lib().msp_csc_transpose(
            rows, cols,
            _lib.addr(self.indptr), _lib.addr(self.indices), _lib.addr(self.data),
            _lib.addr(bp), _lib.addr(bi), _lib.addr(bx), _lib.addr(work),
        )
        return bx, bi, bp

    # -- arithmetic ----------------------------------------------------------

    def matvec(self, v):
        m, n = self._shape
        v = _as_1d(v, np.float64, "the right-hand side")
        if v.size != n:
            raise ValueError(f"expected a vector of length {n}, got {v.size}")
        y = np.empty(m, dtype=np.float64)
        if self.nnz == 0:
            y.fill(0.0)
            return y
        kernel = (
            _lib.lib().msp_csr_matvec
            if self._compressed_axis == 0
            else _lib.lib().msp_csc_matvec
        )
        kernel(
            m, n,
            _lib.addr(self.indptr), _lib.addr(self.indices), _lib.addr(self.data),
            _lib.addr(v), _lib.addr(y),
        )
        return y

    def __matmul__(self, other):
        if isinstance(other, _SparseMatrix):
            raise NotImplementedError(
                "sparse @ sparse is not implemented; densify the right operand"
            )
        arr = np.asarray(other, dtype=np.float64)
        if arr.ndim == 1:
            return self.matvec(arr)
        raise NotImplementedError(
            "only sparse @ vector is supported; use (a @ b.T).T for a matrix right operand"
        )

    def __repr__(self):
        kind = type(self).__name__
        return f"<{kind} {self._shape[0]}x{self._shape[1]} nnz={self.nnz} dtype=float64>"


class csr_matrix(_SparseMatrix):
    """Row-compressed sparse matrix."""

    _compressed_axis = 0

    def __init__(self, arg1, shape=None):
        if shape is None and not (isinstance(arg1, tuple) and len(arg1) == 3):
            arg1, shape = _from_dense(arg1, 0)
        super().__init__(arg1, shape)

    def tocsr(self):
        return self

    def tocsc(self):
        # The CSR buffers reinterpreted as CSC are A^T, so transposing them
        # gives A in CSC.
        bx, bi, bp = self._transposed_csc(self._shape[1], self._shape[0])
        return csc_matrix((bx, bi, bp), shape=self._shape)

    @property
    def T(self):
        m, n = self._shape
        return csc_matrix(
            (self.data.copy(), self.indices.copy(), self.indptr.copy()),
            shape=(n, m),
        )


class csc_matrix(_SparseMatrix):
    """Column-compressed sparse matrix."""

    _compressed_axis = 1

    def __init__(self, arg1, shape=None):
        if shape is None and not (isinstance(arg1, tuple) and len(arg1) == 3):
            arg1, shape = _from_dense(arg1, 1)
        super().__init__(arg1, shape)

    def tocsc(self):
        return self

    def tocsr(self):
        # CSC of A^T is row-compressed A, so the transpose is the CSR form.
        bx, bi, bp = self._transposed_csc(self._shape[0], self._shape[1])
        return csr_matrix((bx, bi, bp), shape=self._shape)

    @property
    def T(self):
        m, n = self._shape
        bx, bi, bp = self._transposed_csc(m, n)
        return csc_matrix((bx, bi, bp), shape=(n, m))