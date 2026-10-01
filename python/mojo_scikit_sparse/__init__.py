"""mojo-scikit-sparse: sparse linear algebra kernels written in Mojo.

The heavy lifting — SpMV, the transpose, and the LDL^T factorization with its
triangular solves — runs in ``src/mojo_sparse.mojo`` behind a C ABI; this
package owns the buffers and exposes the usual analyze/factorize/solve shape of
the CHOLMOD-based ``scikit-sparse``.

Only ``float64`` and symmetric positive definite solves are covered.
"""

from ._factor import (
    Analysis,
    Factor,
    NotPositiveDefiniteError,
    NotSymmetricError,
    analyze,
    cholesky,
    factorize,
    spsolve,
)
from ._lib import lib, library_path
from ._matrix import SparseFormatError, csc_matrix, csr_matrix

__version__ = "0.1.0"

__all__ = [
    "Analysis",
    "Factor",
    "NotPositiveDefiniteError",
    "NotSymmetricError",
    "SparseFormatError",
    "analyze",
    "cholesky",
    "csc_matrix",
    "csr_matrix",
    "factorize",
    "lib",
    "library_path",
    "spsolve",
]