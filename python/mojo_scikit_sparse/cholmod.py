"""Cholesky / LDL factorization of a sparse matrix, ported from ``sksparse.cholmod``.

The public surface here mirrors upstream's ``sksparse.cholmod`` names and
signatures so that this package is a drop-in for the covered subset:

* the symbolic entry points :func:`symbfact` and :func:`etree`;
* the factorization objects :class:`CholeskyFactor`, :func:`cho_factor` and
  :func:`ldl_factor`;
* the convenience wrappers :func:`cholesky`, :func:`ldl` and :func:`cho_solve`.

The heavy arithmetic -- elimination tree, postorder, row/column counts, the
factor pattern walk, the numeric row factorization and the triangular solves --
lives in ``src/ported.mojo`` behind a C ABI.  This module owns the buffers and
drives the kernels in upstream's order.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
from scipy.sparse import csc_array, diags_array, eye_array, issparse

from . import _lib
from ._utils import validate_csc_input

__all__ = [
    "CholmodError",
    "CholmodNotPositiveDefiniteError",
    "CholmodWarning",
    "CholeskyFactor",
    "SymbolicCholFactor",
    "cho_factor",
    "cho_solve",
    "cholesky",
    "etree",
    "ldl",
    "ldl_factor",
    "symbfact",
]


# -----------------------------------------------------------------------------
#         Error Handling
# -----------------------------------------------------------------------------
class CholmodError(Exception):
    """Base class for CHOLMOD-related errors."""


class CholmodNotPositiveDefiniteError(CholmodError):
    """Raised when the input matrix is not positive definite."""


class CholmodWarning(Warning):
    """Base class for CHOLMOD-related warnings."""


# -----------------------------------------------------------------------------
#         Utilities
# -----------------------------------------------------------------------------
_SUPERNODAL_MODES = {"auto", "simplicial", "supernodal"}

_ORDERING_METHODS = {
    "default",
    "best",
    "natural",
    "given",
    "amd",
    "metis",
    "nesdis",
    "colamd",
    "postordered",
}

# This port factors in the natural ordering only.  Upstream's "default" tries AMD
# then METIS; since neither fill-reducing permutation is ported, "default" is
# treated as "natural" here so the common call keeps working (a documented
# divergence -- see README, "Coverage").  "best", "metis", "nesdis", "amd",
# "colamd" and "postordered" genuinely require an unimplemented permutation and
# are rejected rather than silently downgraded.
_SUPPORTED_ORDERINGS = {"default", "natural", None}

_SOLVE_SYSTEMS = {"A", "LDLt", "LD", "DLt", "L", "Lt", "D"}

# Upstream exposes the CHOLMOD ordering code as an int when it does not match a
# name; we surface the same fallback so ``factor.order`` has upstream's type.
_ORDERING_CODES = {
    "natural": 0,
    "given": 1,
    "amd": 2,
    "metis": 3,
    "nesdis": 4,
    "colamd": 5,
    "postordered": 6,
}


def _check_ordering(order):
    """Validate ``order`` and report whether this port can factor with it."""
    if order is not None and order not in _ORDERING_METHODS:
        raise ValueError(
            f"Unknown ordering method: {order}. Must be one of {set(_ORDERING_METHODS)}."
        )
    if order not in _SUPPORTED_ORDERINGS:
        raise ValueError(
            f"order={order!r} requires a fill-reducing ordering that this port does "
            "not implement; use order='natural' (or None). See README, 'Coverage'."
        )


def _as_int64(A):
    """Return (indptr, indices, data) as int64/int64/float64 contiguous arrays.

    The kernels are CHOLMOD's ``cholmod_l_*`` (long) ports and read every index
    as a native 64-bit int; scipy's index arrays are int32 for small matrices,
    so they must be widened before their address crosses the ABI.

    ``f64`` raises on anything it cannot convert without inventing values, which
    keeps complex and low-precision data from reaching a Float64 kernel (where
    numpy would reinterpret the bits instead).
    """
    return _lib.i64(A.indptr), _lib.i64(A.indices), _lib.f64(A.data)


# CHOLMOD's EMPTY sentinel, matching the kernels.
_EMPTY = -1


def _check_csc(A):
    """Reject anything whose layout the kernels would read out of bounds.

    This is the one place that guards the FFI boundary from the Python side: the
    kernels index indptr/indices/data with no bounds checks of their own, so a
    matrix that is not canonical CSC has to be refused here.  The upstream case
    this mirrors is CHOLMOD's ``A->sorted`` check ("requires input indices to be
    sorted").
    """
    if not issparse(A) or A.format != "csc":
        raise ValueError("Kernels require a CSC-format sparse array.")
    n_rows, n_cols = A.shape
    indptr, indices = A.indptr, A.indices
    if indptr.shape != (n_cols + 1,):
        raise ValueError(f"indptr has length {indptr.shape} for {n_cols} columns.")
    if indices.shape != (A.nnz,) or A.data.shape != (A.nnz,):
        raise ValueError("indices/data length does not match nnz.")
    if indptr[0] != 0 or indptr[-1] != A.nnz:
        raise ValueError("indptr must start at 0 and end at nnz.")
    if A.nnz > 1 and np.any(indptr[1:] < indptr[:-1]):
        raise ValueError("indptr must be non-decreasing.")
    # Strictly increasing inside each column.  The column boundaries matter: a
    # decrease that straddles two columns is legal there, so the comparison has
    # to be done per column rather than over the whole index array.
    if A.nnz:
        rows = np.asarray(indices).astype(np.int64, copy=False)
        col_of = np.zeros(A.nnz, dtype=np.int64)
        starts = np.asarray(indptr[1:-1], dtype=np.int64)
        keep = (starts > 0) & (starts < A.nnz)
        col_of[starts[keep]] = (np.arange(1, n_cols))[keep]
        if A.nnz > 1:
            same = col_of[1:] == col_of[:-1]
            bad = np.flatnonzero(same & (rows[1:] <= rows[:-1]))
            if bad.size:
                p = int(bad[0])
                raise ValueError(
                    f"Column {int(col_of[p])} has unsorted row indices "
                    f"({rows[p]} then {rows[p + 1]}); the kernels require "
                    "canonical CSC (sorted indices, no duplicates)."
                )
    return csc_array(A)


# -----------------------------------------------------------------------------
#         Symbolic analysis: the kernels upstream drives
# -----------------------------------------------------------------------------
class _Workspace:
    """The scratch buffers one analysis needs, allocated once and reused.

    CHOLMOD keeps these in ``Common->Flag/Head/Xwork/Iwork`` and grows them as
    needed; the sizes here mirror the workspace each ported routine documents
    (``etree``: ncol + nrow; ``postorder``: 3n; ``rowcolcounts``: 4*nrow+ncol).
    """

    def __init__(self, n, nrow, ncol):
        self.n = n
        self.parent = _lib.zeros_i(n)
        self.post = _lib.zeros_i(n)
        self.colcount = _lib.zeros_i(n)
        self.rowcount = _lib.zeros_i(n)
        self.first = _lib.zeros_i(n)
        self.level = _lib.zeros_i(n)
        self.ancestor = _lib.zeros_i(max(n, 1))
        self.prev = _lib.zeros_i(max(nrow, 1))
        self.prevnbr = _lib.zeros_i(n)
        self.setparent = _lib.zeros_i(n)
        self.prevleaf = _lib.zeros_i(n)
        self.anext = _lib.zeros_i(max(ncol, 1))
        self.head = _lib.zeros_i(n + 1)
        self.ipost = _lib.zeros_i(n)
        self.postnext = _lib.zeros_i(max(n, 1))
        self.poststack = _lib.zeros_i(max(n, 1))
        self.anz = _lib.scalar_i()


def _lib_ptrs(A):
    """The addresses of a matrix's widened CSC buffers, in one pass.

    Every kernel takes buffer addresses as plain ints, so the only thing this
    avoids is repeated Python-level work per call, not a copy: the buffers are
    the very arrays scipy handed us whenever they are already int64/float64 and
    contiguous, and ``i64``/``f64`` return the input itself in that case.
    """
    Ap = _lib.i64(A.indptr)
    Ai = _lib.i64(A.indices)
    Ax = _lib.f64(A.data)
    return _lib.addr(Ap), _lib.addr(Ai), _lib.addr(Ax), Ap, Ai, Ax


def _column_pointers(colcount, n):
    """The column pointers of a factor whose column j has colcount[j] entries.

    This is a dependent scan, so it stays in Python: for n in the tens of
    thousands the loop is tens of microseconds against a factorize in the
    milliseconds, and a numpy cumulative sum here would allocate.
    """
    Lp = _lib.zeros_i(n + 1)
    lnz = 0
    for j in range(n):
        Lp[j] = lnz
        lnz += int(colcount[j])
    Lp[n] = lnz
    return Lp, lnz


# -----------------------------------------------------------------------------
#         Matrix rearrangement (the transposes upstream does between kernels)
# -----------------------------------------------------------------------------
def _transpose(A, values=True):
    """B = A', column-compressed, via cholmod_transpose.

    ``values=False`` transposes the pattern only (CHOLMOD_TRANS_PATTERN), which
    is all the symbolic passes need; ``values=True`` also moves the values
    (CHOLMOD_TRANS_CONJ for real data, which is a plain transpose).
    """
    A = csc_array(A)
    A = _check_csc(A)
    m, n = A.shape
    if A.nnz == 0:
        return csc_array((n, m), dtype=A.dtype)
    Ap, Ai, Ax = _as_int64(A)
    Bp = _lib.zeros_i(m + 1)
    Bi = _lib.zeros_i(A.nnz)
    Bx = _lib.zeros_f(A.nnz)
    W = _lib.zeros_i(max(n, 1))
    _lib.lib().msp_transpose(
        m, n,
        _lib.addr(Ap), _lib.addr(Ai), _lib.addr(Ax),
        _lib.addr(Bp), _lib.addr(Bi), _lib.addr(Bx),
        1 if values else 0, 1,
        _lib.addr(W),
    )
    data = Bx if values else np.ones(Bi.size, dtype=np.float64)
    return csc_array((data, Bi, Bp), shape=(n, m))


# -----------------------------------------------------------------------------
#         Symbolic factorization: the three analysis passes
# -----------------------------------------------------------------------------
def _analyze_pattern(A, *, sym_kind, lower, want_pattern):
    """The symbolic analysis shared by symbfact() and CholeskyFactor.__init__.

    Runs the three kernels in upstream's order -- transpose if needed, etree,
    postorder, rowcolcounts, and finally the pattern walk if the caller wants
    the symbolic factor -- and returns everything CholeskyFactor and symbfact
    need.
    """
    A = validate_csc_input(A, ensure_double=False)
    A = _check_csc(A)
    M, N = A.shape

    if sym_kind is None:
        sym_kind = "sym"
    if sym_kind not in {"sym", "row", "col"}:
        raise ValueError(
            f"Unknown symmetry kind: {sym_kind}. Must be one of 'sym', 'row', 'col'."
        )
    if sym_kind == "sym" and M != N:
        raise ValueError(f"Expected square matrix. Got {A.shape}.")

    n = A.shape[0] if sym_kind != "col" else A.shape[1]
    if sym_kind != "sym":
        M = N  # the unsymmetric cases always analyze a square A A' or A' A
    nrow = n

    ws = _Workspace(n, M, N)

    # CHOLMOD's stype for the two kernels that consult it.  -1 means "the lower
    # triangle is the symmetric part" and is what cholmod_rowcolcounts requires;
    # 0 means the unsymmetric A A' / A' A case.
    rc_stype = -1
    transpose = False

    if sym_kind == "row":
        rc_stype = 0  # use A A'
    elif sym_kind == "col":
        rc_stype = 0  # use A' A
        transpose = True

    # cholmod_analyze transposes the pattern once and hands each kernel the view
    # it wants (upstream's `Aup = Ac; Alo = Fc`):
    #
    #   Fc = transpose(Ac);  if stype == 1: Aup = Ac, Alo = Fc
    #                       else:         Aup = Fc, Alo = Ac
    #
    # cholmod_etree reads the UPPER triangle (it rejects a lower-stype matrix)
    # and cholmod_rowcolcounts reads the LOWER one, so the two never share a
    # buffer in the symmetric case.  For the unsymmetric cases both read the
    # original A.
    if transpose:
        Aup = Alo = A
    elif sym_kind == "sym" and lower:
        # tril(A) is the symmetric part, so triu(A) == (tril(A))' == A'
        Aup = _transpose(A, values=False)
        Alo = A
    elif sym_kind == "sym":
        Aup = A
        Alo = _transpose(A, values=False)
    else:
        Aup = Alo = A

    Aup = csc_array(Aup)
    Alo = csc_array(Alo)
    Aup_p, Aup_i, _ = _as_int64(Aup)
    Alo_p, Alo_i, _ = _as_int64(Alo)

    # cholmod_etree: stype 1 for the symmetric case (Aup is triu), 0 for the
    # unsymmetric case, where all of Aup is used and the tree is that of A A'.
    _lib.lib().msp_etree(
        n, nrow,
        _lib.addr(Aup_p), _lib.addr(Aup_i),
        _lib.addr(ws.parent), _lib.addr(ws.ancestor), _lib.addr(ws.prev),
        1 if rc_stype != 0 else 0,
    )

    # cholmod_postorder.  The scratch arrays live on the workspace, never as
    # temporaries: ctypes does not hold a reference to a numpy array passed as an
    # argument, so a temporary can be freed before the kernel reads it.
    _lib.lib().msp_postorder(
        n,
        _lib.addr(ws.parent),
        _lib.addr(ws.post),
        _lib.addr(ws.head),
        _lib.addr(ws.postnext),
        _lib.addr(ws.poststack),
    )

    # cholmod_rowcolcounts reads the lower triangle for the symmetric case and
    # all of A for the unsymmetric one.
    _lib.lib().msp_rowcolcounts(
        n, Alo.shape[1],
        _lib.addr(Alo_p), _lib.addr(Alo_i),
        _lib.addr(ws.parent), _lib.addr(ws.post),
        _lib.addr(ws.colcount), _lib.addr(ws.rowcount),
        _lib.addr(ws.first), _lib.addr(ws.level),
        _lib.addr(ws.prevnbr), _lib.addr(ws.setparent), _lib.addr(ws.prevleaf),
        _lib.addr(ws.anext), _lib.addr(ws.head), _lib.addr(ws.ipost),
        rc_stype,
        _lib.addr(ws.anz),
    )

    result = {
        "A": A,
        "Aup": Aup,
        "Alo": Alo,
        "n": n,
        "parent": ws.parent.copy(),
        "post": ws.post.copy(),
        "colcount": ws.colcount.copy(),
        "rowcount": ws.rowcount.copy(),
        "first": ws.first.copy(),
        "level": ws.level.copy(),
        "anz": int(ws.anz[0]),
        "stype": 1 if rc_stype == -1 else 0,
        "rc_stype": rc_stype,
        "sym_kind": sym_kind,
        "lower": lower,
        "transpose": transpose,
        "ws": ws,
    }
    if want_pattern:
        result["L"] = _cholesky_pattern(result)
    return result


def _cholesky_pattern(an):
    """Port of upstream's ``_cholesky_l_pattern``: the symbolic factor pattern.

    CHOLMOD's cholmod_l_row_subtree gives the pattern of each row k of L; this
    places each into the correct column of a packed column-compressed L, using
    the ColCount column sizes computed by rowcolcounts.  Returns a csc_array
    whose data is boolean True, matching upstream.

    The whole walk is one ``msp_row_pattern`` call: upstream here is a Python
    loop over k that re-enters ctypes n times and then loops over the row's
    nonzeros in Python a second time, which is invisible in a kernel profile and
    costs far more than the walk itself.  The kernel does both in Mojo, and
    reports each row's length back in ``Len``.
    """
    ws = an["ws"]
    n = an["n"]
    Aup = an["Aup"]
    Ap, Ai, _, _Ap, _Ai, _Ax = _lib_ptrs(Aup)
    colcount = an["colcount"]

    Lp, lnz = _column_pointers(colcount, n)
    if lnz == 0:
        return csc_array((n, n), dtype=np.float64)

    Li = _lib.zeros_i(lnz)
    W = Lp[:-1].copy()  # a copy of the column pointers, advanced as columns fill
    Flag = _lib.zeros_i(n)
    # the kernel pushes row k's run onto the end of Stack, so it has to be able
    # to reach nnz(L), not n
    Stack = _lib.zeros_i(max(lnz, 1))
    Len = _lib.zeros_i(n)

    # For the unsymmetric cases, row_subtree needs F = A' (or A(:,f)'); we do
    # not restrict to a column subset here, so F is the full transpose.
    Fp = Fi = 0
    if an["stype"] == 0:
        F = csc_array(_transpose(Aup, values=False))
        # hold the widened buffers: ctypes keeps no reference to them
        Fp_a, Fi_a, _Fx_a = _lib.i64(F.indptr), _lib.i64(F.indices), _lib.f64(F.data)
        Fp, Fi = _lib.addr(Fp_a), _lib.addr(Fi_a)

    _lib.lib().msp_row_pattern(
        n,
        Ap, Ai,
        Fp, Fi,
        _lib.addr(an["parent"]),
        _lib.addr(Lp), _lib.addr(Li),
        _lib.addr(W), _lib.addr(Len),
        _lib.addr(Flag), _lib.addr(Stack),
        an["stype"],
    )

    data = np.ones(lnz, dtype=bool)
    return csc_array((data, Li, Lp), shape=(n, n))


# -----------------------------------------------------------------------------
#         CholeskyFactor
# -----------------------------------------------------------------------------
class CholeskyFactor:
    """The main object used for creating and manipulating a Cholesky factor.

    The constructor computes the symbolic analysis of the matrix.  The numeric
    factorization is not computed until :meth:`factorize` is called.

    Parameters
    ----------
    A : (N, N) array_like or sparse array
        An array convertible to a sparse matrix in CSC format.
    sym_kind : str in {"sym", "row", "col"}, optional
        The type of factorization to analyze: ``sym`` for the symmetric case,
        ``row`` for :math:`A A^{\\top}`, ``col`` for :math:`A^{\\top} A`.
    supernodal_mode : str in {"auto", "simplicial", "supernodal"}, optional
        Accepted for API compatibility; this port is simplicial throughout.
    lower : bool, optional
        If True, use the lower triangular part of ``A``.
    order : None or str, optional
        The permutation algorithm.  Only ``natural`` and ``None`` (the identity)
        are implemented; see README, "Coverage".
    """

    def __init__(
        self,
        A,
        *,
        lower: bool = True,
        order: str | None = "default",
        sym_kind: str | None = None,
        supernodal_mode: str | None = None,
    ):
        A = validate_csc_input(A, ensure_double=False)
        A = _check_csc(A)
        # The kernels read A's values as float64 through the C ABI.  A complex
        # matrix would be reinterpreted bit for bit and produce a plausible but
        # wrong factor, so it is refused here rather than after conversion.
        if A.dtype.kind == "c":
            raise ValueError(
                f"This port factors real float64 data; got {A.dtype}. "
                "See README, 'Coverage'."
            )

        if sym_kind is None:
            sym_kind = "sym"
        if supernodal_mode is None:
            supernodal_mode = "auto"

        if sym_kind not in {"sym", "row", "col"}:
            raise ValueError(
                f"Unknown symmetry kind: {sym_kind}. "
                "Must be one of 'sym', 'row', 'col'."
            )
        if sym_kind == "sym" and A.shape[0] != A.shape[1]:
            raise ValueError(f"Expected square matrix. Got {A.shape}.")
        if supernodal_mode not in _SUPERNODAL_MODES:
            raise ValueError(
                f"Unknown factorization mode: {supernodal_mode}. "
                f"Must be one of {set(_SUPERNODAL_MODES)}."
            )
        _check_ordering(order)

        # Matrix of all zeros is structurally not positive definite
        if A.shape[0] > 0 and A.nnz == 0:
            raise CholmodNotPositiveDefiniteError(
                "Input matrix not positive definite."
            )

        self._is_lower = lower
        self._sym_kind = sym_kind
        self._rcond = -1.0
        self._is_numeric = False
        self._is_ll = False
        self._order = "natural"
        self._perm = None
        self._colcount = None
        self._factor = None
        self._L = None
        self._D = None
        self._factor_views = None

        self._analysis = _analyze_pattern(
            A, sym_kind=sym_kind, lower=lower, want_pattern=False
        )
        self._n = self._analysis["n"]
        self._colcount = self._analysis["colcount"]
        self._perm = np.arange(self._n, dtype=np.int64)

    # -- properties ----------------------------------------------------------

    @property
    def is_ll(self):
        return self._is_ll

    @property
    def is_super(self):
        # this port is simplicial throughout
        return False

    @property
    def is_numeric(self):
        return self._is_numeric

    @property
    def is_lower(self):
        return self._is_lower

    @property
    def itype(self):
        return np.dtype(np.int64)

    @property
    def dtype(self):
        return np.dtype(np.float64)

    @property
    def N(self):
        return self._n

    @property
    def colcount(self):
        return self._colcount

    @property
    def nnz(self):
        return int(np.sum(self._colcount))

    @property
    def order(self):
        return self._order

    @property
    def perm(self):
        return self._perm

    @property
    def rcond(self):
        return self._rcond

    @property
    def sym_kind(self):
        return self._sym_kind

    def __repr__(self):
        return (
            f"CholeskyFactor(N={self.N}, nnz={self.nnz}, is_ll={self.is_ll}, "
            f"is_super={self.is_super}, itype=np.{self.itype.name}, "
            f"dtype=np.{self.dtype.name}, order={self.order})"
        )

    def __str__(self):
        lines = [
            f"Cholesky factorization of size {self.N}x{self.N}",
            f"  Nonzeros: {self.nnz}",
            f"  Form:     {'LL.T' if self.is_ll else 'LDL.T'}",
            f"  Triangle: {'lower' if self.is_lower else 'upper'}",
            f"  Storage:  simplicial",
            f"  itype:    np.{self.itype.name}",
            f"  dtype:    np.{self.dtype.name}",
            f"  order:    {self.order}",
        ]
        return "\n".join(lines)

    # -- internal state ------------------------------------------------------

    def _require_factorized(self):
        if not self.is_numeric:
            raise CholmodError(
                "Factor is symbolic. Call `factorize` before updating."
            )

    def _clear_cache(self):
        self._factor = None
        self._factor_views = None
        self._L = None
        self._D = None

    def _materialize(self):
        """Assemble the packed factor into the public L / D views."""
        self._L = self._factor
        self._D = None
        return self._L

    # -- public methods ------------------------------------------------------

    def copy(self):
        """Return a deep copy of this factor."""
        other = CholeskyFactor.__new__(CholeskyFactor)
        other._is_lower = self._is_lower
        other._sym_kind = self._sym_kind
        other._rcond = self._rcond
        other._is_numeric = self._is_numeric
        other._is_ll = self._is_ll
        other._order = self._order
        other._n = self._n
        other._colcount = self._colcount.copy()
        other._perm = self._perm.copy()
        other._analysis = dict(self._analysis)
        other._factor = None if self._factor is None else self._factor.copy()
        other._L = None
        other._D = None
        other._factor_views = None
        return other

    def change_factor(self, kind=None):
        """Change the type of factorization between ``LL.T`` and ``LDL.T``.

        The symbolic analysis is reused; this does not refactorize.
        """
        self._require_factorized()
        if kind is None:
            kind = "LDL" if self.is_ll else "LL"
        if kind not in ("LL", "LDL"):
            raise ValueError("kind must be 'LL' or 'LDL'.")
        self._clear_cache()
        self._is_ll = kind == "LL"
        return self

    def get_factor(self, kind=None, lower=None):
        """Return a copy of the factor in the requested format.

        ``kind='LL'`` returns the Cholesky factor ``L`` such that
        :math:`L L^\\top = P A P^\\top`.  ``kind='LDL'`` returns ``(L, D)``
        such that :math:`L D L^\\top = P A P^\\top`.  With ``lower=False`` the
        upper triangular factor is returned instead.
        """
        self._require_factorized()
        if kind is None:
            kind = "LL" if self.is_ll else "LDL"
        if kind not in ("LL", "LDL"):
            raise ValueError("kind must be 'LL' or 'LDL'.")
        if lower is None:
            lower = self.is_lower
        return self._get_factor_impl(kind, lower)

    def _get_factor_impl(self, kind, lower):
        L = self._materialize()
        if kind == "LL":
            out = L.copy()
            out.eliminate_zeros()
            if not lower:
                out = out.T.conj()
            return out
        else:
            # Extract L and D from the combined LD factor.  In this port the
            # packed factor stores D(k,k) as the first (diagonal) entry of column
            # k with a unit L diagonal, exactly like CHOLMOD's LD form.
            LD = L.copy()
            D = diags_array(LD.diagonal())
            LD.setdiag(1.0)
            LD.eliminate_zeros()
            if not lower:
                LD = LD.T.conj()
            return LD, D

    def get_perm(self):
        """Return a copy of the permutation vector used in the factorization."""
        return self.perm.copy()

    def factorize(self, A, ldl=None, beta=0.0):
        """Compute the numerical factorization of ``A``.

        ``ldl=False`` (the default on the first call) computes the Cholesky
        factorization :math:`R^\\top R = P A P^\\top` (or :math:`L L^\\top`);
        ``ldl=True`` computes :math:`R^\\top D R` (or :math:`L D L^\\top`).
        ``beta`` is added to the diagonal before factorization.
        """
        A = validate_csc_input(A, ensure_double=False)
        A = _check_csc(A)
        if A.dtype.kind == "c":
            raise ValueError(
                f"This port factors real float64 data; got {A.dtype}. "
                "See README, 'Coverage'."
            )

        if ldl is None:
            ldl = not self.is_ll if self.is_numeric else False
        if not isinstance(ldl, bool):
            raise ValueError("ldl must be a boolean value.")
        self._clear_cache()
        self._is_ll = not ldl

        an = self._analysis
        if A.shape != an["A"].shape:
            raise ValueError(
                f"Cannot factor a matrix of shape {A.shape} with a factor built "
                f"for {an['A'].shape}."
            )
        n = an["n"]

        # cholmod_factorize permutes/transposes A if necessary, because
        # cholmod_rowfac only accepts the UPPER triangle ("symmetric lower
        # triangular form not supported" in cholmod_rowfac.c).  With
        # lower=False that is a no-op; with lower=True the matrix is transposed
        # here, which is where upstream puts that divergence.
        if an["lower"]:
            Mat = csc_array(_transpose(A, values=True))
        else:
            Mat = A
        Mat_ap, Mat_ai, Mat_ax, _Ap, _Ai, _Ax = _lib_ptrs(Mat)

        # Allocate the packed factor to exactly nnz(L) entries from the symbolic
        # pass, so the numeric factor is a pure write into known memory.
        Lp, lnz = _column_pointers(an["colcount"], n)
        Li = _lib.zeros_i(max(lnz, 1))
        Lx = _lib.zeros_f(max(lnz, 1))
        Lnz = _lib.zeros_i(n)
        Flag = _lib.zeros_i(n)
        Stack = _lib.zeros_i(n)
        W = _lib.zeros_f(n)
        minor = _lib.scalar_i()
        fl = _lib.scalar_f()

        # For the unsymmetric cases, the numeric pass needs F = A' (row case) or
        # A (col case, which is A' of the transposed matrix).  Upstream's
        # cholmod_factorize_p takes A and F, with F = A' for the row case.
        Fp = Fi = Fx = 0
        if an["stype"] == 0:
            if an["sym_kind"] == "row":
                F = csc_array(_transpose(Mat, values=True))
            else:
                F = Mat
            # hold the widened buffers: ctypes keeps no reference to them
            Fp_a, Fi_a, Fx_a = _lib.i64(F.indptr), _lib.i64(F.indices), _lib.f64(F.data)
            Fp, Fi, Fx = _lib.addr(Fp_a), _lib.addr(Fi_a), _lib.addr(Fx_a)

        _lib.lib().msp_rowfac(
            n, float(beta),
            Mat_ap, Mat_ai, Mat_ax,
            Fp, Fi, Fx,
            _lib.addr(Lp), _lib.addr(Li), _lib.addr(Lx), _lib.addr(Lnz),
            _lib.addr(Flag), _lib.addr(Stack), _lib.addr(W),
            1 if self.is_ll else 0,
            _lib.addr(minor), _lib.addr(fl),
            an["stype"],
        )

        if int(minor[0]) < n:
            raise CholmodNotPositiveDefiniteError(
                "Input matrix is not positive definite."
            )

        # the packed factor is column-compressed; expose it as a csc_array whose
        # diagonal entry is first in each column
        self._factor = csc_array((Lx[:lnz], Li[:lnz], Lp), shape=(n, n))
        self._is_numeric = True
        # the solve driver rebuilds these buffers on every call; they do not
        # depend on b, so they are built once per factor
        self._factor_views = None

        # Update rcond: (L.diagonal().min()/L.diagonal().max())**2 for LL'
        self._update_rcond()
        return self  # for method chaining

    def _update_rcond(self):
        L = self._materialize()
        if self.is_ll:
            diag = L.diagonal()
            if diag.size and diag.min() != 0:
                self._rcond = (diag.min() / diag.max()) ** 2
            else:
                self._rcond = 0.0
        else:
            _, D = self._get_factor_impl("LDL", True)
            d = D.diagonal()
            self._rcond = float(np.sum(d > 0) / max(len(d), 1))

    def solve(self, b, system="A"):
        """Solve a linear system using the factorization.

        ``system`` is one of ``{'A', 'LDLt', 'LD', 'DLt', 'L', 'Lt', 'D'}``.
        Only ``A`` accounts for the permutation; the others solve using the
        factors directly.
        """
        self._require_factorized()

        if not (isinstance(b, np.ndarray) or issparse(b)):
            raise ValueError("b must be an ndarray or sparse matrix.")
        if b.ndim not in (1, 2):
            raise ValueError("b must be a 1D or 2D array.")

        N = self.N
        if b.shape[0] != N:
            raise ValueError("Right-hand side b must have the same number of rows as L.")
        if not np.can_cast(b.dtype, self.dtype):
            raise TypeError(f"Cannot safely cast {b.dtype=} to {self.dtype=}.")
        b = b.astype(self.dtype, copy=False)

        if system not in _SOLVE_SYSTEMS:
            raise ValueError(
                f"Unknown factorization mode: {system}. "
                f"Must be one of {set(_SOLVE_SYSTEMS)}."
            )

        if N == 0:
            return type(b)(b.shape, dtype=b.dtype)

        # check the condition number before solving
        self._check_rcond()

        return_1D = b.ndim == 1
        B = b.reshape((N, 1)) if b.ndim == 1 else b

        if issparse(B):
            X = self._solve_sparse(B, system)
        else:
            X = self._solve_dense(B, system)

        if return_1D:
            X = X[:, 0]
        return X

    def _solve_dense(self, b, system):
        """Solve the system with a dense right-hand side, one column at a time.

        Each column is copied out of ``b`` once and handed to the kernels as a
        contiguous float64 buffer, so the solve never goes back through numpy.
        The copy is not optional: a (N, 1) right-hand side is already C-contiguous
        per column, so ``ascontiguousarray`` would hand back a view onto the
        caller's buffer and the in-place solves would overwrite the input (and
        alias each other across repeated calls).
        """
        N = self.N
        ncol = b.shape[1]
        X = np.zeros((N, ncol), dtype=np.float64)
        B = _lib.f64(b)
        for j in range(ncol):
            x = np.array(B[:, j], dtype=np.float64, order="C")
            self._solve_system_1(x, system)
            X[:, j] = x
        return X

    def _solve_sparse(self, b, system):
        """Solve the system with a sparse right-hand side, one column at a time."""
        B = csc_array(b)
        N = self.N
        cols = []
        for j in range(B.shape[1]):
            x = np.zeros(N, dtype=np.float64)
            col = B[:, j]
            # solve with the sparse column by dense-treating it: the sparse solve
            # fills only the nonzeros of x, matching cholmod_spsolve
            self._solve_system_1(x, system, sparse_init=col.toarray().ravel())
            cols.append(x)
        return csc_array(np.column_stack(cols))

    def _solve_system_1(self, x, system, sparse_init=None):
        """Solve one right-hand side, dispatching on ``system``.

        This mirrors cholmod_solve's dispatch to the lsolve/ltsolve/Dsolve
        kernels, applying the same order of operations upstream uses.
        """
        N = self.N
        (Lp, Li, Lx, Lnz, Dd, _permuted), _keepalive = self._solve_buffers()

        if sparse_init is not None:
            x[:] = sparse_init

        is_ll = self.is_ll
        xa = _lib.addr(x)
        if system == "A":
            # A x = b, i.e. P A P' x = b.  Only this system applies the
            # permutation; upstream's cholmod_solve dispatches to
            # cholmod_l_permute, the triangular solves, then cholmod_l_unpermute.
            # The kernel calls stay here (rather than being skipped) so this
            # path is exercised by the tests rather than only reachable once an
            # ordering is ported; with the natural ordering both are the identity
            # and the gathers are exact copies.
            y = _lib.zeros_f(N)
            _lib.lib().msp_vec_permute(N, _lib.addr(self.perm), xa, _lib.addr(y))
            self._ldl_solve_inplace(y, Lp, Li, Lx, Lnz, Dd)
            _lib.lib().msp_vec_unpermute(N, _lib.addr(self.perm), _lib.addr(y), xa)
        elif system == "LDLt":
            # L D L' x = b (no permutation)
            self._ldl_solve_inplace(x, Lp, Li, Lx, Lnz, Dd)
        elif system == "LD":
            # L D x = b.  cholmod dispatches to ll_lsolve when the factor is
            # LL' (D is the identity) and to ldl_ldsolve when it is LDL', where
            # the kernel divides by the D stored in each column's diagonal slot.
            _lib.lib().msp_lsolve(N, Lp, Li, Lx, Lnz, xa, 1 if is_ll else 0, 0 if is_ll else 1)
        elif system == "DLt":
            # D L' x = b.  For LL' D is the identity, so this is just the
            # backward substitution; for LDL' cholmod's ldl_dltsolve divides by D
            # first, then does the unit-diagonal backward substitution.
            if not is_ll:
                _lib.lib().msp_vec_scale_inv(N, Dd, xa, xa)
            _lib.lib().msp_ltsolve(N, Lp, Li, Lx, Lnz, xa, 1 if is_ll else 0)
        elif system == "L":
            _lib.lib().msp_lsolve(N, Lp, Li, Lx, Lnz, xa, 1 if is_ll else 0, 0)
        elif system == "Lt":
            _lib.lib().msp_ltsolve(N, Lp, Li, Lx, Lnz, xa, 1 if is_ll else 0)
        elif system == "D":
            # D is the identity for an LL' factor, so the scale is by one and
            # there is nothing to divide by: the buffer is already the answer.
            if Dd is not None:
                _lib.lib().msp_vec_scale_inv(N, Dd, xa, xa)
        return x

    def _ldl_solve_inplace(self, x, Lp, Li, Lx, Lnz, Dd):
        """Solve (L L' or L D L') x = b in place, in upstream's operation order.

        LL'  : forward substitution with L, then backward substitution with L'.
        LDL' : forward substitution with L, divide by D, then backward with L'.
        """
        N = self.N
        xa = _lib.addr(x)
        if self.is_ll:
            _lib.lib().msp_lsolve(N, Lp, Li, Lx, Lnz, xa, 1, 0)
            _lib.lib().msp_ltsolve(N, Lp, Li, Lx, Lnz, xa, 1)
        else:
            _lib.lib().msp_lsolve(N, Lp, Li, Lx, Lnz, xa, 0, 0)
            _lib.lib().msp_vec_scale_inv(N, Dd, xa, xa)
            _lib.lib().msp_ltsolve(N, Lp, Li, Lx, Lnz, xa, 0)

    def _solve_buffers(self):
        """The factor's kernel-facing buffers, built once per factor.

        This returns ``(addrs, keepalive)``.  These do not depend on the
        right-hand side, so rebuilding them on every ``solve`` was pure overhead:
        an int64 copy of the indices, a diff for the column lengths, a diagonal
        extraction, and two O(n) identity checks.

        ``keepalive`` matters as much as the addresses: ctypes keeps no reference
        to a buffer whose address it is handed, and ``np.diff`` and
        ``.diagonal()`` both return fresh arrays, so caching their addresses
        without holding the arrays would read freed memory.  That is the same
        invariant the rest of this module keeps.
        """
        cached = self._factor_views
        if cached is not None:
            return cached
        L = self._materialize()
        Lp = _lib.i64(L.indptr)
        Li = _lib.i64(L.indices)
        Lx = _lib.f64(L.data)
        # CHOLMOD's lsolve uses Lnz to bound each column; in a packed csc the
        # column length is Lp[j+1]-Lp[j].
        Lnz = _lib.i64(np.diff(Lp))
        if self.is_ll:
            Dd = None
        else:
            Dd = _lib.f64(self._D_diagonal())
        permuted = not self._perm_is_identity()
        keepalive = (Lp, Li, Lx, Lnz, Dd)
        cached = (
            (
                _lib.addr(Lp), _lib.addr(Li), _lib.addr(Lx), _lib.addr(Lnz),
                None if Dd is None else _lib.addr(Dd),
                permuted,
            ),
            keepalive,
        )
        self._factor_views = cached
        return cached

    def _D_diagonal(self):
        """The diagonal D, for an LDL' factor (all ones for an LL' factor)."""
        L = self._materialize()
        if self.is_ll:
            return np.ones(self.N, dtype=np.float64)
        return L.diagonal()

    def _perm_is_identity(self):
        return bool(np.array_equal(self.perm, np.arange(self.N)))

    def _check_rcond(self):
        thresh = self.N * np.finfo(self.dtype).eps
        if self.rcond == 0:
            raise CholmodNotPositiveDefiniteError(
                "Matrix is indefinite or singular to working precision."
            )
        elif self.rcond < thresh:
            import warnings

            warnings.warn(
                "Matrix is nearly singular."
                f"  Results may be inaccurate (rcond={self.rcond:.2e}).",
                CholmodWarning,
            )

    # -- not implemented -----------------------------------------------------

    def update(self, C):
        return self._update(C, updown="up")

    def downdate(self, C):
        return self._update(C, updown="down")

    def _update(self, C, updown="up"):
        raise NotImplementedError(
            "rank-k update/downdate (cholmod_updown) is not ported; see README, "
            "'Coverage'."
        )

    def rowadd(self, k, C):
        raise NotImplementedError(
            "rowadd (cholmod_rowadd) is not ported; see README, 'Coverage'."
        )

    def rowdel(self, k):
        raise NotImplementedError(
            "rowdel (cholmod_rowdel) is not ported; see README, 'Coverage'."
        )

    def resymbol(self, A, is_permuted=True):
        raise NotImplementedError(
            "resymbol (cholmod_resymbol) is not ported; see README, 'Coverage'."
        )

    # -- determinant helpers -------------------------------------------------

    def logdet(self):
        """The natural log-determinant of the matrix."""
        self._require_factorized()
        if self.is_ll:
            L = self._materialize()
            return 2 * np.sum(np.log(L.diagonal()))
        else:
            _, D = self._get_factor_impl("LDL", True)
            return np.sum(np.log(D.diagonal()))

    def slogdet(self):
        """The sign and log-determinant of the matrix."""
        return (self.dtype.type(1.0), self.logdet())

    def det(self):
        """The determinant of the matrix."""
        return np.exp(self.logdet())

    def inv(self):
        """The inverse of the matrix, computed from the factorization."""
        return self.solve(eye_array(self.N, format="csc", dtype=self.dtype))


# -----------------------------------------------------------------------------
#         Symbolic factorization entry points
# -----------------------------------------------------------------------------
SymbolicCholFactor = namedtuple(
    "SymbolicCholFactor",
    ["count", "h", "parent", "post", "L"],
)


def symbfact(A, *, kind=None, lower: bool = False, return_factor: bool = False):
    """Symbolic factorization of a sparse matrix for Cholesky or LDL.

    Computes the elimination tree and analyzes the sparsity pattern of the
    factor.  ``kind`` selects what is analyzed: ``sym`` for the symmetric case
    (default), ``row`` for :math:`A A^\\top`, ``col`` for :math:`A^\\top A`, and
    ``lo`` for the lower-triangular symmetric case (``symbfact(A.T)``).

    Returns
    -------
    SymbolicCholFactor
        Fields ``count`` (nonzeros per column of the factor), ``h`` (height of
        the elimination tree), ``parent``, ``post``, and ``L`` (the symbolic
        factor, if ``return_factor=True``).
    """
    A = validate_csc_input(A)

    if kind is None:
        kind = "sym"
    if kind not in {"sym", "row", "col", "lo"}:
        raise ValueError(f"Unknown factorization kind: {kind}")

    M, N = A.shape
    if kind not in ["row", "col"] and M != N:
        raise ValueError(f"Input matrix A must be square, got shape {A.shape}.")

    if (kind == "row" and M == 0) or N == 0:
        empty = np.array([], dtype=np.int64)
        count, h, parent, post = empty, 0, empty, empty
        Lsym = A.copy()
        if return_factor:
            return SymbolicCholFactor(count, h, parent, post, Lsym)
        return SymbolicCholFactor(count, h, parent, post, None)

    if A.nnz == 0:
        D = N if kind == "col" else M
        count = np.zeros(D, dtype=np.int64)
        h = 1
        parent = np.full(D, -1, dtype=np.int64)
        post = np.arange(D, dtype=np.int64)
        Lsym = eye_array(D, dtype=A.dtype)
        if return_factor:
            return SymbolicCholFactor(count, h, parent, post, Lsym)
        return SymbolicCholFactor(count, h, parent, post, None)

    # Map kind -> (sym_kind, lower) for the shared analysis.
    if kind == "sym":
        sym_kind, use_lower = "sym", lower
    elif kind == "lo":
        sym_kind, use_lower = "sym", True
    elif kind == "row":
        sym_kind, use_lower = "row", False
    else:
        sym_kind, use_lower = "col", False

    an = _analyze_pattern(
        A, sym_kind=sym_kind, lower=use_lower, want_pattern=return_factor
    )

    count = an["colcount"].copy()
    parent = an["parent"].copy()
    post = an["post"].copy()
    level = an["level"]
    h = int(np.max(level)) + 1 if level.size else 1

    if return_factor:
        Lsym = an["L"]
        if not lower:
            Lsym = _transpose_pattern(Lsym)
        return SymbolicCholFactor(count, h, parent, post, Lsym)
    return SymbolicCholFactor(count, h, parent, post, None)


def _transpose_pattern(L):
    """Return R = L' as a boolean pattern csc_array (matching upstream's upper)."""
    return _transpose(L, values=False).astype(bool)


def etree(A, *, kind=None, return_post: bool = False):
    """The elimination tree of a sparse matrix, optionally postordered.

    ``kind`` selects what is analyzed: ``sym`` for the symmetric case
    (default), ``row`` for :math:`A A^\\top`, ``col`` for :math:`A^\\top A`, and
    ``lo`` for the lower-triangular symmetric case (``etree(A.T)``).

    Returns ``parent``, or ``(parent, post)`` if ``return_post=True``.
    """
    A = validate_csc_input(A)

    if kind is None:
        kind = "sym"
    if kind not in {"sym", "row", "col", "lo"}:
        raise ValueError(f"Unknown factorization kind: {kind}")

    M, N = A.shape
    if kind not in ["row", "col"] and M != N:
        raise ValueError(f"Input matrix A must be square, got shape {A.shape}.")

    if (kind == "row" and M == 0) or N == 0:
        parent = np.array([], dtype=np.int64)
        if return_post:
            return parent, parent.copy()
        return parent

    if A.nnz == 0:
        D = N if kind == "col" else M
        parent = np.full(D, -1, dtype=np.int64)
        if return_post:
            return parent, np.arange(D, dtype=np.int64)
        return parent

    if kind == "sym":
        sym_kind, use_lower = "sym", False
    elif kind == "lo":
        sym_kind, use_lower = "sym", True
    elif kind == "row":
        sym_kind, use_lower = "row", False
    else:
        sym_kind, use_lower = "col", False

    an = _analyze_pattern(A, sym_kind=sym_kind, lower=use_lower, want_pattern=False)
    parent = an["parent"].copy()
    if return_post:
        return parent, an["post"].copy()
    return parent


# -----------------------------------------------------------------------------
#         Convenience functions
# -----------------------------------------------------------------------------
def cho_factor(
    A, beta=0.0, *, lower=False, order="default", sym_kind=None, supernodal_mode=None
):
    """Compute the Cholesky factorization of a sparse matrix (LL.T form)."""
    return CholeskyFactor(
        A, lower=lower, order=order, sym_kind=sym_kind, supernodal_mode=supernodal_mode
    ).factorize(A, ldl=False, beta=beta)


def ldl_factor(
    A, beta=0.0, *, lower=True, order="default", sym_kind=None, supernodal_mode=None
):
    """Compute the LDL.T factorization of a sparse matrix."""
    return CholeskyFactor(
        A, lower=lower, order=order, sym_kind=sym_kind, supernodal_mode=supernodal_mode
    ).factorize(A, ldl=True, beta=beta)


def cholesky(
    A, beta=0.0, *, lower=False, order="default", sym_kind=None, supernodal_mode=None
):
    """Return the Cholesky factor ``(R, p)`` of a sparse matrix."""
    f = cho_factor(
        A, beta=beta, lower=lower, order=order,
        sym_kind=sym_kind, supernodal_mode=supernodal_mode,
    )
    R = f.get_factor()
    p = f.get_perm()
    return R if order is None else (R, p)


def ldl(A, beta=0.0, *, lower=True, order="default", sym_kind=None, supernodal_mode=None):
    """Return the LDL.T factorization ``(L, D, p)`` of a sparse matrix."""
    f = ldl_factor(
        A, beta=beta, lower=lower, order=order,
        sym_kind=sym_kind, supernodal_mode=supernodal_mode,
    )
    R, D = f.get_factor()
    p = f.get_perm()
    return (R, D) if order is None else (R, D, p)


def cho_solve(A, b, *, lower=False, order="default"):
    """Solve the linear system ``A x = b`` using the Cholesky factorization."""
    return cho_factor(A, lower=lower, order=order).solve(b)