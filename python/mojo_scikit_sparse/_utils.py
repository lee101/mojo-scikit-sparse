"""Input validation, ported from ``sksparse.utils.validate_csc_input``.

Upstream validates in three steps: convert to a 2-D CSC array, coerce the value
dtype up to double, and sum duplicates so the indices are sorted and canonical.
CHOLMOD then relies on that canonical form (``A.sorted = True`` with the note
"requires input indices to be sorted"), so this is not optional.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import SparseEfficiencyWarning, csc_array, issparse


def validate_csc_input(A, require_square: bool = False, ensure_double: bool = True):
    """Validate and convert input matrix to CSC format.

    Parameters
    ----------
    A : (M, N) array_like
        Input matrix, converted to CSC if it is not already.
    require_square : bool, optional
        If True, the input matrix must be square (M == N).
    ensure_double : bool, optional
        If True, floating-point data is widened to double precision.

    Returns
    -------
    A : csc_array
        The input matrix converted to canonical CSC format.
    """
    # Convert dense to sparse CSC
    if not issparse(A):
        A = np.asarray(A)

    if A.ndim != 2:
        raise ValueError("Input must be 2D.")

    M, N = A.shape

    if require_square and M != N:
        raise ValueError("Input must be square.")

    try:
        if not (issparse(A) and A.format == "csc"):
            import warnings

            warnings.warn(
                f"Input matrix ({type(A)}) not in CSC format. Converting to CSC.",
                SparseEfficiencyWarning,
                stacklevel=3,
            )
            A = csc_array(A)
    except ValueError as exc:
        # csc_array() rejected the input; keep the original message, which says
        # what was wrong with it, rather than replacing it with a generic one.
        raise ValueError("Input must be convertible to CSC format.") from exc

    # Coerce bool or int data to float
    if np.issubdtype(A.dtype, np.bool_) or np.issubdtype(A.dtype, np.integer):
        A = A.astype(np.float64)

    # Coerce float32 to float64.  Upstream hands whatever the user passed to
    # CHOLMOD, which promotes on load; a float32 matrix would otherwise reach a
    # Float64 kernel and be reinterpreted bit for bit, which is how the port used
    # to silently drop half of every value.
    if np.issubdtype(A.dtype, np.floating) and A.dtype != np.float64:
        A = A.astype(np.float64)

    # Ensure double precision if requested
    if ensure_double:
        if np.issubdtype(A.dtype, np.floating):
            A = A.astype(np.float64, copy=False)
        elif np.issubdtype(A.dtype, np.complexfloating):
            A = A.astype(np.complex128, copy=False)

    # sum_duplicates sorts the indices and sums any duplicates, which is the
    # canonical form every kernel below assumes.  scipy refuses the in-place
    # merge unless the index arrays are 32-bit and it has verified they hold no
    # duplicate entries, so a matrix that is not canonical is rebuilt through a
    # coo round trip instead of failing with scipy's opaque "Output dtype not
    # compatible with inputs".
    if A.data.dtype != A.dtype:
        A = csc_array((A.data, A.indices, A.indptr), shape=A.shape, dtype=A.dtype)
    if not A.has_canonical_format:
        A = csc_array(A.tocoo())
    A.sum_duplicates()

    if not (A.has_sorted_indices and A.has_canonical_format):
        raise ValueError("Failed to convert input to canonical CSC format.")

    return A