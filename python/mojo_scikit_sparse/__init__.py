"""mojo-scikit-sparse: a Mojo port of the sparse Cholesky kernels in
``scikit-sparse`` (SuiteSparse CHOLMOD).

The public API mirrors ``sksparse.cholmod``: the symbolic entry points
:func:`symbfact` and :func:`etree`, the factorization object
:class:`CholeskyFactor` with :func:`cho_factor` / :func:`ldl_factor`, and the
convenience wrappers :func:`cholesky`, :func:`ldl` and :func:`cho_solve`.

The compute-bound routines -- the elimination tree, its postordering, the
row/column counts of the factor, the symbolic factor pattern, the numeric row
factorization, and the triangular solves -- are ported from CHOLMOD's C sources
into ``src/ported.mojo`` and run behind a C ABI.  This package owns the buffers
and drives the kernels.

Only ``float64`` / ``int64`` symmetric positive-definite factorizations with a
natural ordering are supported; see README, "Coverage", for the full list of
what is and is not ported.
"""

from .cholmod import (
    CholmodError,
    CholmodNotPositiveDefiniteError,
    CholmodWarning,
    CholeskyFactor,
    SymbolicCholFactor,
    cho_factor,
    cho_solve,
    cholesky,
    etree,
    ldl,
    ldl_factor,
    symbfact,
)
from ._lib import lib, library_path

__version__ = "0.1.0"

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
    "lib",
    "library_path",
    "symbfact",
]