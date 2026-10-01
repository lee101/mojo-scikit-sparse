"""Tests for the Mojo kernels and the package around them."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

import mojo_scikit_sparse as mss
from mojo_scikit_sparse import SparseFormatError


# --- fixtures --------------------------------------------------------------


def csr_from_triplets(rows, cols, vals, shape):
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    vals = np.asarray(vals, dtype=np.float64)
    order = np.argsort(rows * shape[1] + cols, kind="stable")
    rows, cols, vals = rows[order], cols[order], vals[order]
    indptr = np.zeros(shape[0] + 1, dtype=np.int64)
    if rows.size:
        np.cumsum(np.bincount(rows, minlength=shape[0]), out=indptr[1:])
    return mss.csr_matrix((vals, cols, indptr), shape=shape)


def grid5pt(nx, ny):
    """5-point Laplacian on an nx by ny grid, SPD, and it produces fill-in."""
    n = nx * ny
    rows, cols, vals = [], [], []
    for j in range(ny):
        for i in range(nx):
            r = j * nx + i
            rows.append(r)
            cols.append(r)
            vals.append(4.0)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if 0 <= a < nx and 0 <= b < ny:
                    rows.append(r)
                    cols.append(b * nx + a)
                    vals.append(-1.0)
    return csr_from_triplets(rows, cols, vals, (n, n))


def spd(n, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    return A @ A.T + n * np.eye(n)


def dense_tridiag(n, diag=4.0, off=-1.0):
    S = np.zeros((n, n))
    for i in range(n):
        S[i, i] = diag
        if i:
            S[i, i - 1] = off
        if i + 1 < n:
            S[i, i + 1] = off
    return S


@pytest.fixture(scope="module")
def lib_handle():
    return ctypes.CDLL(str(mss.library_path()))


# --- build -----------------------------------------------------------------


def test_library_exists():
    assert mss.library_path().exists(), "run `pixi run build` first"


@pytest.mark.parametrize(
    "symbol",
    [
        "msp_csc_matvec",
        "msp_csr_matvec",
        "msp_csc_transpose",
        "msp_csc_lower_solve",
        "msp_csc_lower_transpose_solve",
        "msp_vec_scale_inv",
        "msp_ldl_analyze",
        "msp_ldl_factor",
    ],
)
def test_symbol_exported(lib_handle, symbol):
    assert hasattr(lib_handle, symbol)


# --- containers ------------------------------------------------------------


@pytest.mark.parametrize("shape", [(6, 5), (1, 4), (4, 1), (3, 3)])
def test_dense_roundtrip(shape):
    rng = np.random.default_rng(0)
    dense = rng.normal(size=shape)
    dense[rng.random(shape) < 0.6] = 0.0
    for cls in (mss.csc_matrix, mss.csr_matrix):
        assert np.array_equal(cls(dense).toarray(), dense)


def test_csc_and_csr_are_transposes_of_each_other():
    dense = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert np.array_equal(mss.csc_matrix(dense).tocsr().toarray(), dense)
    assert np.array_equal(mss.csr_matrix(dense).tocsc().toarray(), dense)


@pytest.mark.parametrize("shape", [(5, 3), (3, 5)])
def test_transpose(shape):
    rng = np.random.default_rng(1)
    dense = rng.normal(size=shape)
    dense[rng.random(shape) < 0.5] = 0.0
    assert np.array_equal(mss.csc_matrix(dense).T.toarray(), dense.T)
    assert np.array_equal(mss.csr_matrix(dense).T.toarray(), dense.T)


def test_transpose_identity_on_symmetric_matrix():
    dense = dense_tridiag(6)
    assert np.allclose(mss.csr_matrix(dense).T.toarray(), dense)


def test_unsorted_and_duplicate_entries_are_canonicalised():
    # Row 0 stores column 2 twice and out of order; the values must sum.
    data = np.array([5.0, 1.0, 2.0, 7.0])
    indices = np.array([2, 0, 2, 1])
    indptr = np.array([0, 3, 4])
    A = mss.csr_matrix((data, indices, indptr), shape=(2, 3))
    assert np.array_equal(A.indices, [0, 2, 1])
    assert np.array_equal(A.data, [1.0, 7.0, 7.0])
    assert np.array_equal(A.toarray(), np.array([[1.0, 0.0, 7.0], [0.0, 7.0, 0.0]]))


@pytest.mark.parametrize(
    "triple,shape",
    [
        # indptr does not start at zero
        (([1.0], [0], [1, 1]), (1, 1)),
        # indptr is not non-decreasing
        (([1.0], [0], [0, 2, 1]), (2, 1)),
        # indptr points past the end of indices
        (([1.0], [0], [0, 3]), (1, 1)),
        # index out of range
        (([1.0, 2.0], [0, 5], [0, 2]), (1, 2)),
    ],
)
def test_malformed_input_rejected(triple, shape):
    with pytest.raises(SparseFormatError):
        mss.csr_matrix(triple, shape=shape)


def test_wrong_indptr_length_rejected():
    with pytest.raises(SparseFormatError):
        mss.csr_matrix(([1.0], [0], [0, 1, 1]), shape=(1, 1))


def test_compressed_input_requires_shape():
    with pytest.raises(ValueError):
        mss.csr_matrix(([1.0], [0], [0, 1]))


def test_rejects_non_sparse_operand():
    with pytest.raises(TypeError):
        mss.factorize(np.eye(3))


@pytest.mark.parametrize("shape", [(4, 4), (4, 0), (0, 4), (0, 0)])
def test_empty_shapes(shape):
    dense = np.zeros(shape)
    for cls in (mss.csc_matrix, mss.csr_matrix):
        A = cls(dense)
        assert A.shape == shape
        assert A.nnz == 0
        assert np.array_equal(A.toarray(), dense)
        assert np.array_equal(A.T.toarray(), dense.T)


@pytest.mark.parametrize("shape", [(5, 3), (3, 5), (4, 4)])
def test_matvec_matches_dense(shape):
    rng = np.random.default_rng(2)
    dense = rng.normal(size=shape)
    dense[rng.random(shape) < 0.5] = 0.0
    x = rng.normal(size=shape[1])
    assert np.allclose(mss.csc_matrix(dense) @ x, dense @ x)
    assert np.allclose(mss.csr_matrix(dense) @ x, dense @ x)


def test_matvec_of_zero_matrix():
    A = mss.csr_matrix(np.zeros((3, 4)))
    assert np.array_equal(A @ np.ones(4), np.zeros(3))


def test_matvec_with_zero_entries_in_the_vector():
    dense = np.array([[1.0, 2.0], [3.0, 4.0]])
    A = mss.csc_matrix(dense)
    assert np.allclose(A @ np.array([0.0, 5.0]), dense @ np.array([0.0, 5.0]))


def test_matvec_rejects_wrong_length():
    A = mss.csr_matrix(np.eye(3))
    with pytest.raises(ValueError):
        A @ np.ones(4)


def test_matmul_matrix_rhs_rejected():
    A = mss.csr_matrix(np.eye(3))
    with pytest.raises(NotImplementedError):
        A @ np.ones((3, 2))


def test_sparse_times_sparse_rejected():
    A = mss.csr_matrix(np.eye(3))
    with pytest.raises(NotImplementedError):
        A @ mss.csr_matrix(np.eye(3))


def test_copy_is_independent():
    A = mss.csr_matrix(dense_tridiag(4))
    B = A.copy()
    B.data[0] = 99.0
    assert A.data[0] != 99.0


# --- factorization ---------------------------------------------------------


@pytest.mark.parametrize("nx,ny", [(2, 2), (3, 3), (8, 8), (15, 15)])
def test_cholesky_reconstructs_the_matrix(nx, ny):
    A = grid5pt(nx, ny)
    factor = mss.cholesky(A)
    assert np.allclose(factor.toarray(), A.toarray())


def test_cholesky_reports_fill_in():
    A = grid5pt(6, 6)
    analysis = mss.analyze(A)
    factor = mss.cholesky(A)
    # The grid Laplacian is sparse to start with but its Cholesky factor is not.
    assert factor.nnz > A.nnz
    assert factor.nnz == analysis.nnz_L


def test_solve_a_reduces_the_residual():
    rng = np.random.default_rng(4)
    A = grid5pt(10, 10)
    b = rng.normal(size=A.shape[0])
    x = mss.cholesky(A).solve_A(b)
    assert np.max(np.abs(A @ x - b)) < 1e-10


@pytest.mark.parametrize("n", [1, 2, 5, 40])
def test_spsolve_matches_dense(n):
    dense = dense_tridiag(n) if n > 1 else np.array([[2.0]])
    A = mss.csc_matrix(dense)
    b = np.arange(1.0, n + 1.0)
    assert np.allclose(mss.spsolve(A, b), np.linalg.solve(dense, b))


@pytest.mark.parametrize("n", [4, 25])
def test_spsolve_matches_dense_on_random_spd(n):
    dense = spd(n, seed=n)
    A = mss.csc_matrix(dense)
    b = np.ones(n)
    assert np.allclose(mss.spsolve(A, b), np.linalg.solve(dense, b), atol=1e-9)


def test_triangular_solves_are_inverses_of_each_other():
    A = grid5pt(7, 7)
    factor = mss.cholesky(A)
    rng = np.random.default_rng(5)
    b = rng.normal(size=A.shape[0])
    L = factor.L.toarray() + np.eye(A.shape[0])
    assert np.allclose(L @ factor.solve_L(b), b)
    assert np.allclose(L.T @ factor.solve_Lt(b), b)


def test_solve_rejects_wrong_length():
    factor = mss.cholesky(grid5pt(4, 4))
    with pytest.raises(ValueError):
        factor.solve_A(np.ones(3))


def test_factor_is_reusable_for_several_rhs():
    A = grid5pt(6, 6)
    factor = mss.cholesky(A)
    rng = np.random.default_rng(6)
    for _ in range(3):
        b = rng.normal(size=A.shape[0])
        assert np.max(np.abs(A @ factor.solve_A(b) - b)) < 1e-10


def test_factorize_accepts_either_container():
    dense = dense_tridiag(7)
    assert np.allclose(mss.cholesky(mss.csr_matrix(dense)).toarray(), dense)
    assert np.allclose(mss.cholesky(mss.csc_matrix(dense)).toarray(), dense)


def test_lower_triangle_only_storage_is_accepted():
    dense = dense_tridiag(8)
    lower = np.tril(dense)
    factor = mss.cholesky(mss.csr_matrix(lower))
    assert np.allclose(factor.toarray(), dense)


def test_upper_triangle_only_storage_is_accepted():
    dense = dense_tridiag(8)
    factor = mss.cholesky(mss.csr_matrix(np.triu(dense)))
    assert np.allclose(factor.toarray(), dense)


def test_asymmetric_entries_rejected():
    A = mss.csr_matrix(np.array([[2.0, 1.0], [3.0, 2.0]]))
    with pytest.raises(mss.NotSymmetricError):
        mss.factorize(A)


def test_symmetric_matrix_is_accepted():
    A = mss.csr_matrix(np.array([[2.0, 1.0], [1.0, 2.0]]))
    assert np.allclose(mss.factorize(A).toarray(), A.toarray())


def test_indefinite_matrix_rejected_by_cholesky():
    A = mss.csc_matrix(np.array([[1.0, 2.0], [2.0, 1.0]]))
    assert np.allclose(mss.factorize(A).toarray(), A.toarray())
    with pytest.raises(mss.NotPositiveDefiniteError):
        mss.cholesky(A)


def test_singular_matrix_rejected():
    A = mss.csc_matrix(np.array([[1.0, 1.0], [1.0, 1.0]]))
    with pytest.raises(mss.NotPositiveDefiniteError):
        mss.cholesky(A)


def test_zero_diagonal_rejected():
    A = mss.csc_matrix(np.array([[0.0, 1.0], [1.0, 1.0]]))
    with pytest.raises(mss.NotPositiveDefiniteError):
        mss.cholesky(A)


def test_non_square_rejected():
    A = mss.csr_matrix(np.ones((3, 4)))
    with pytest.raises(ValueError):
        mss.factorize(A)


def test_zero_sized_matrix():
    factor = mss.cholesky(mss.csr_matrix(np.zeros((0, 0))))
    assert factor.shape == (0, 0)
    assert factor.nnz == 0
    assert factor.solve_A(np.zeros(0)).shape == (0,)


def test_one_by_one_matrix():
    factor = mss.cholesky(mss.csr_matrix(np.array([[9.0]])))
    assert factor.D[0] == pytest.approx(9.0)
    assert np.allclose(factor.solve_A([3.0]), [1.0 / 3.0])


def test_analysis_can_be_reused():
    A = grid5pt(5, 5)
    analysis = mss.analyze(A)
    first = analysis.factorize()
    second = analysis.factorize()
    assert np.array_equal(first.D, second.D)
    assert first.nnz == second.nnz == analysis.nnz_L
    assert analysis.Lnz.sum() == analysis.nnz_L


@pytest.mark.parametrize("n", [6, 12])
def test_analyze_predicts_the_factor_nnz(n):
    dense = spd(n, seed=n)
    analysis = mss.analyze(mss.csc_matrix(dense))
    assert analysis.nnz_L == mss.cholesky(mss.csc_matrix(dense)).nnz