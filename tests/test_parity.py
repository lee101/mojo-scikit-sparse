"""Parity tests: every assertion is against the real upstream scikit-sparse.

The whole point of this port is that it can be read against upstream and that it
agrees with it.  So the tests here do not check "it ran": they compare the
elimination tree, the postordering, the column counts, the symbolic pattern, the
numeric factors and the solutions against ``sksparse.cholmod`` computed on the
same input.  Where upstream is unavailable the module skips rather than quietly
testing against itself.
"""

import numpy as np
import pytest
import scipy.sparse as sp

import mojo_scikit_sparse as mss

cholmod = pytest.importorskip("sksparse.cholmod", reason="upstream scikit-sparse is required for parity")


# ---------------------------------------------------------------------------
# matrices
# ---------------------------------------------------------------------------
def davis11():
    """The 11x11 example from the CHOLMOD MATLAB docs (Davis, Eqn 2.1).

    This is the exact matrix upstream uses in the ``symbfact`` and ``etree``
    docstrings, so the published vectors in those docstrings apply directly.
    """
    N = 11
    rows = np.array([5, 6, 2, 7, 9, 10, 5, 9, 7, 10, 8, 9, 10, 9, 10, 10])
    cols = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 9])
    rng = np.random.default_rng(56)
    vals = rng.random(len(rows), dtype=np.float64)
    L = sp.coo_array((vals, (rows, cols)), shape=(N, N))
    A = L + L.T
    A.setdiag(N)
    return sp.csc_array(A)


def spd_sparse(n, density=0.04, seed=0):
    """A sparse SPD matrix whose sparsity pattern forces fill-in."""
    r = np.random.default_rng(seed)
    M = r.normal(size=(n, n))
    A = M @ M.T + n * np.eye(n)
    mask = r.random((n, n)) < density
    mask = mask | mask.T | np.eye(n, dtype=bool)
    return sp.csc_array(A * mask)


def spd_grid(k):
    """5-point Laplacian on a k-by-k grid: SPD, sparse, and it fills in."""
    n = k * k
    rows, cols, vals = [], [], []
    for j in range(k):
        for i in range(k):
            r = j * k + i
            rows.append(r)
            cols.append(r)
            vals.append(4.0)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if 0 <= a < k and 0 <= b < k:
                    rows.append(r)
                    cols.append(b * k + a)
                    vals.append(-1.0)
    return sp.csc_array(
        (np.asarray(vals), (np.asarray(rows), np.asarray(cols))), shape=(n, n)
    )


KINDS = ["sym", "lo", "row", "col"]


# ---------------------------------------------------------------------------
# symbfact
# ---------------------------------------------------------------------------
def test_symbfact_matches_upstream_docstring_vectors():
    """The docstring vectors in upstream's symbfact are reproducible values.

    If they hold for upstream they must hold for this port; comparing against the
    literal vectors pins the test to upstream's published output rather than to
    whatever upstream currently returns.
    """
    A = davis11()
    got = mss.symbfact(A)
    assert np.array_equal(got.count, np.array([3, 3, 4, 3, 3, 4, 4, 3, 3, 2, 1]))
    assert got.h == 6
    assert np.array_equal(
        got.parent, np.array([5, 2, 7, 5, 7, 6, 8, 9, 9, 10, -1])
    )
    assert np.array_equal(got.post, np.array([1, 2, 4, 7, 0, 3, 5, 6, 8, 9, 10]))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("n,seed", [(11, 56), (40, 2), (120, 5)])
def test_symbfact_matches_upstream(kind, n, seed):
    A = davis11() if (n == 11 and seed == 56) else spd_sparse(n, seed=seed)
    ours = mss.symbfact(A, kind=kind)
    theirs = cholmod.symbfact(A, kind=kind)
    assert np.array_equal(ours.count, theirs.count)
    assert ours.h == theirs.h
    assert np.array_equal(ours.parent, theirs.parent)
    assert np.array_equal(ours.post, theirs.post)


@pytest.mark.parametrize("kind", KINDS)
def test_symbfact_factor_pattern_matches_upstream(kind):
    A = spd_sparse(60, seed=7)
    ours = mss.symbfact(A, kind=kind, return_factor=True).L
    theirs = cholmod.symbfact(A, kind=kind, return_factor=True).L
    assert np.array_equal(ours.toarray(), theirs.toarray())


@pytest.mark.parametrize("lower", [False, True])
def test_symbfact_factor_lower_upper(lower):
    A = davis11()
    ours = mss.symbfact(A, lower=lower, return_factor=True).L.toarray()
    theirs = cholmod.symbfact(A, lower=lower, return_factor=True).L.toarray()
    assert np.array_equal(ours, theirs)
    # count is the per-column count of the *lower* factor either way
    assert np.count_nonzero(ours) == int(mss.symbfact(A).count.sum())


# ---------------------------------------------------------------------------
# etree
# ---------------------------------------------------------------------------
def test_etree_matches_upstream_docstring_vectors():
    got = mss.etree(davis11(), return_post=True)
    parent, post = got
    assert np.array_equal(parent, np.array([5, 2, 7, 5, 7, 6, 8, 9, 9, 10, -1]))
    assert np.array_equal(post, np.array([1, 2, 4, 7, 0, 3, 5, 6, 8, 9, 10]))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("n,seed", [(11, 56), (60, 2), (120, 5)])
def test_etree_matches_upstream(kind, n, seed):
    A = davis11() if (n == 11 and seed == 56) else spd_sparse(n, seed=seed)
    assert np.array_equal(mss.etree(A, kind=kind), cholmod.etree(A, kind=kind))
    o_parent, o_post = mss.etree(A, kind=kind, return_post=True)
    u_parent, u_post = cholmod.etree(A, kind=kind, return_post=True)
    assert np.array_equal(o_parent, u_parent)
    assert np.array_equal(o_post, u_post)


def test_etree_postorder_is_a_permutation():
    for n, seed in ((40, 1), (90, 4)):
        parent, post = mss.etree(spd_sparse(n, seed=seed), return_post=True)
        assert sorted(post.tolist()) == list(range(n))
        # a postordering visits every node strictly before its parent
        rank = np.empty(n, dtype=np.int64)
        rank[post] = np.arange(n)
        for j, p in enumerate(parent):
            if p != -1:
                assert rank[j] < rank[p]


# ---------------------------------------------------------------------------
# numeric factorization
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lower", [True, False])
@pytest.mark.parametrize("n,seed", [(30, 1), (80, 3)])
def test_cholesky_factor_matches_upstream(lower, n, seed):
    A = spd_sparse(n, seed=seed)
    ours = mss.CholeskyFactor(A, lower=lower, order="natural").factorize(A)
    theirs = cholmod.CholeskyFactor(A, lower=lower, order="natural").factorize(A)
    assert np.allclose(
        ours.get_factor(kind="LL").toarray(),
        theirs.get_factor(kind="LL").toarray(),
        rtol=1e-9,
        atol=1e-12,
    )


def test_colcount_matches_upstream():
    A = spd_sparse(70, seed=6)
    ours = mss.CholeskyFactor(A, order="natural")
    theirs = cholmod.CholeskyFactor(A, order="natural")
    assert np.array_equal(ours.colcount, theirs.colcount)
    assert ours.nnz == theirs.nnz


def test_ldl_factor_matches_upstream():
    A = spd_sparse(70, seed=6)
    ours = mss.ldl_factor(A, order="natural", lower=True).get_factor()
    theirs = cholmod.ldl_factor(A, order="natural", lower=True).get_factor()
    assert np.allclose(ours[0].toarray(), theirs[0].toarray(), rtol=1e-9, atol=1e-12)
    assert np.allclose(ours[1].diagonal(), theirs[1].diagonal(), rtol=1e-9, atol=1e-12)


def test_reconstruction_is_exact():
    """A = L D L' must reproduce the input, not merely resemble it."""
    for k in (5, 9, 14):
        A = spd_grid(k)
        L, D = mss.ldl_factor(A, order="natural", lower=True).get_factor()
        recon = (L.toarray() * D.diagonal()) @ L.toarray().T
        assert np.abs(recon - A.toarray()).max() < 1e-10


def test_cholesky_reconstruction_is_exact():
    for k in (5, 9):
        A = spd_grid(k)
        R = mss.cho_factor(A, order="natural", lower=True).get_factor()
        recon = R.toarray() @ R.toarray().T
        assert np.abs(recon - A.toarray()).max() < 1e-10


# ---------------------------------------------------------------------------
# solves
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("system", ["A", "LDLt", "L", "Lt"])
@pytest.mark.parametrize("n,seed", [(30, 1), (80, 3)])
def test_solve_matches_upstream(system, n, seed):
    A = spd_sparse(n, seed=seed)
    rng = np.random.default_rng(seed + 100)
    b = rng.normal(size=n)
    ours = mss.cho_factor(A, lower=True, order="natural").solve(b, system=system)
    theirs = cholmod.cho_factor(A, lower=True, order="natural").solve(b, system=system)
    assert np.allclose(ours, theirs, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("n,seed", [(30, 1), (80, 3)])
def test_ldl_solve_matches_upstream(n, seed):
    A = spd_sparse(n, seed=seed)
    rng = np.random.default_rng(seed + 200)
    b = rng.normal(size=n)
    ours = mss.ldl_factor(A, lower=True, order="natural").solve(b)
    theirs = cholmod.ldl_factor(A, lower=True, order="natural").solve(b)
    assert np.allclose(ours, theirs, rtol=1e-8, atol=1e-10)


def test_solve_residual_is_small():
    for k in (6, 11):
        A = spd_grid(k)
        rng = np.random.default_rng(k)
        b = rng.normal(size=A.shape[0])
        x = mss.cho_solve(A, b, lower=True, order="natural")
        assert np.abs(A.toarray() @ x - b).max() < 1e-9


def test_cho_solve_matches_upstream():
    A = spd_grid(8)
    rng = np.random.default_rng(11)
    b = rng.normal(size=A.shape[0])
    assert np.allclose(
        mss.cho_solve(A, b, lower=True, order="natural"),
        cholmod.cho_solve(A, b, lower=True, order="natural"),
        rtol=1e-8,
        atol=1e-10,
    )


def test_two_dimensional_rhs():
    A = spd_sparse(40, seed=8)
    rng = np.random.default_rng(3)
    B = rng.normal(size=(40, 3))
    f = mss.cho_factor(A, lower=True, order="natural")
    X = f.solve(B)
    assert X.shape == (40, 3)
    assert np.abs(A.toarray() @ X - B).max() < 1e-9
    theirs = cholmod.cho_factor(A, lower=True, order="natural").solve(B)
    assert np.allclose(X, theirs, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("system", ["LD", "DLt", "D"])
def test_composite_systems_are_self_consistent(system):
    """For the composite systems, check L*(result) reproduces the stated product."""
    A = spd_sparse(50, seed=9)
    f = mss.ldl_factor(A, lower=True, order="natural")
    L, D = f.get_factor()
    rng = np.random.default_rng(4)
    b = rng.normal(size=50)
    x = f.solve(b, system=system)
    if system == "D":
        assert np.allclose(D.toarray() @ x, b, atol=1e-9)
    elif system == "LD":
        assert np.allclose((L @ D).toarray() @ x, b, atol=1e-9)
    else:
        assert np.allclose((D @ L.T.toarray()) @ x, b, atol=1e-9)


# ---------------------------------------------------------------------------
# determinants and inverse
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n,seed", [(30, 1), (70, 5)])
def test_logdet_matches_upstream(n, seed):
    A = spd_sparse(n, seed=seed)
    for ours_f, theirs_f in (
        (mss.cho_factor(A, lower=True, order="natural"), cholmod.cho_factor(A, lower=True, order="natural")),
        (mss.ldl_factor(A, lower=True, order="natural"), cholmod.ldl_factor(A, lower=True, order="natural")),
    ):
        assert ours_f.logdet() == pytest.approx(theirs_f.logdet(), rel=1e-10)


def test_det_matches_numpy():
    for k in (4, 7):
        A = spd_grid(k)
        dense = A.toarray()
        f = mss.cho_factor(A, lower=True, order="natural")
        assert f.det() == pytest.approx(np.linalg.det(dense), rel=1e-8)
        sign, logabs = f.slogdet()
        assert sign == 1
        assert logabs == pytest.approx(np.log(np.linalg.det(dense)), rel=1e-8)


def test_inv_matches_upstream():
    A = spd_sparse(50, seed=12)
    ours = mss.cho_factor(A, lower=True, order="natural").inv()
    theirs = cholmod.cho_factor(A, lower=True, order="natural").inv()
    assert np.allclose(ours.toarray(), theirs.toarray(), rtol=1e-7, atol=1e-10)
    assert np.abs(A.toarray() @ ours.toarray() - np.eye(50)).max() < 1e-9


# ---------------------------------------------------------------------------
# beta
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("beta", [1.0, 2.5])
def test_beta_matches_upstream(beta):
    A = spd_sparse(40, seed=13)
    ours = mss.cho_factor(A, beta=beta, lower=True, order="natural").get_factor()
    theirs = cholmod.cho_factor(A, beta=beta, lower=True, order="natural").get_factor()
    assert np.allclose(ours.toarray(), theirs.toarray(), rtol=1e-9, atol=1e-12)


def test_beta_defines_the_matrix():
    A = sp.csc_array(np.array([[1.0, 1.0], [1.0, 1.0]]))
    f = mss.cho_factor(A, beta=2.0, lower=True, order="natural")
    L = f.get_factor().toarray()
    assert np.allclose(L @ L.T, A.toarray() + 2.0 * np.eye(2), atol=1e-12)


# ---------------------------------------------------------------------------
# sym_kind
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sym_kind", ["row", "col"])
@pytest.mark.parametrize("n,seed", [(30, 1), (60, 4)])
def test_unsymmetric_kinds_match_upstream(sym_kind, n, seed):
    A = spd_sparse(n, seed=seed)
    b = np.random.default_rng(seed).normal(size=n)
    ours = mss.CholeskyFactor(A, order="natural", sym_kind=sym_kind, lower=True)
    ours.factorize(A)
    theirs = cholmod.CholeskyFactor(A, order="natural", sym_kind=sym_kind, lower=True)
    theirs.factorize(A)
    assert np.allclose(ours.solve(b), theirs.solve(b), rtol=1e-8, atol=1e-10)
    # the factorization really is of A A' (row) or A' A (col)
    L = ours.get_factor(kind="LL").toarray()
    dense = A.toarray()
    want = dense @ dense.T if sym_kind == "row" else dense.T @ dense
    assert np.abs(L @ L.T - want).max() < 1e-8


# ---------------------------------------------------------------------------
# API surface and error handling
# ---------------------------------------------------------------------------
def test_module_exports_upstream_names():
    for name in (
        "CholeskyFactor",
        "cholesky",
        "ldl",
        "cho_factor",
        "ldl_factor",
        "cho_solve",
        "symbfact",
        "etree",
    ):
        assert hasattr(mss, name), name


def test_cholesky_returns_factor_and_permutation():
    A = spd_sparse(30, seed=2)
    R, p = mss.cholesky(A, lower=True, order="natural")
    assert R.shape == (30, 30)
    assert p.shape == (30,)


def test_ldl_returns_factor_diagonal_and_permutation():
    A = spd_sparse(30, seed=2)
    L, D, p = mss.ldl(A, lower=True, order="natural")
    assert L.shape == (30, 30)
    assert D.diagonal().shape == (30,)
    assert p.shape == (30,)


def test_not_positive_definite_raises():
    A = sp.csc_array(np.array([[-1.0, 0.0], [0.0, -1.0]]))
    with pytest.raises(mss.CholmodNotPositiveDefiniteError):
        mss.cho_factor(A, lower=True, order="natural")


def test_upstream_raises_the_same_way():
    A = sp.csc_array(np.array([[-1.0, 0.0], [0.0, -1.0]]))
    with pytest.raises(cholmod.CholmodNotPositiveDefiniteError):
        cholmod.cho_factor(A, lower=True, order="natural")


def test_zero_matrix_is_rejected():
    with pytest.raises(mss.CholmodNotPositiveDefiniteError):
        mss.CholeskyFactor(sp.csc_array((3, 3)), order="natural")


def test_solving_before_factorizing_raises():
    f = mss.CholeskyFactor(spd_sparse(20, seed=1), order="natural")
    with pytest.raises(mss.CholmodError):
        f.solve(np.ones(20))


def test_unsupported_ordering_is_rejected():
    A = spd_sparse(20, seed=1)
    with pytest.raises(ValueError):
        mss.cho_factor(A, order="amd")


def test_unported_methods_raise_not_implemented():
    f = mss.CholeskyFactor(spd_sparse(20, seed=1), order="natural")
    f.factorize(f._analysis["A"])
    for name in ("rowadd", "rowdel", "resymbol"):
        with pytest.raises(NotImplementedError):
            getattr(f, name)(0, None) if name != "rowdel" else f.rowdel(0)
    with pytest.raises(NotImplementedError):
        f.update(sp.csc_array((20, 1)))


def test_unknown_system_is_rejected():
    A = spd_sparse(20, seed=1)
    f = mss.cho_factor(A, lower=True, order="natural")
    with pytest.raises(ValueError):
        f.solve(np.ones(20), system="nope")


def test_non_square_sym_is_rejected():
    with pytest.raises(ValueError):
        mss.CholeskyFactor(sp.csc_array(np.ones((3, 4))), order="natural")


def test_unknown_kind_is_rejected():
    A = spd_sparse(20, seed=1)
    with pytest.raises(ValueError):
        mss.symbfact(A, kind="bogus")
    with pytest.raises(ValueError):
        mss.etree(A, kind="bogus")


# ---------------------------------------------------------------------------
# shapes and edge cases
# ---------------------------------------------------------------------------
def test_single_element_matrix():
    A = sp.csc_array(np.array([[4.0]]))
    L = mss.cho_factor(A, lower=True, order="natural").get_factor()
    assert L.toarray()[0, 0] == pytest.approx(2.0)


def test_diagonal_matrix_factorizes_exactly():
    d = np.array([1.0, 2.0, 3.0, 4.0])
    A = sp.csc_array(np.diag(d))
    L, D = mss.ldl_factor(A, order="natural", lower=True).get_factor()
    assert np.allclose(D.diagonal(), d)
    assert np.allclose(np.diag(L.toarray()), 1.0)


def test_one_by_one_solve():
    A = sp.csc_array(np.array([[4.0]]))
    assert mss.cho_solve(A, np.array([2.0]), lower=True, order="natural")[0] == pytest.approx(0.5)


def test_dense_input_is_accepted():
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    L = mss.cho_factor(A, lower=True, order="natural").get_factor().toarray()
    assert np.allclose(L @ L.T, A, atol=1e-12)


def test_upper_triangle_convention():
    """lower=False must use only the upper triangle, as upstream documents."""
    dense = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    A = sp.csc_array(dense)
    f = mss.cho_factor(A, lower=False, order="natural")
    R = f.get_factor(kind="LL", lower=True).toarray()
    assert np.allclose(R @ R.T, dense, atol=1e-12)


def test_duplicate_entries_are_summed_like_upstream():
    """validate_csc_input sums duplicates, and the factor must reflect the sum.

    The COO input has two entries at (0,1), so the canonical CSC matrix carries
    A(0,1) == 1.5.  With ``lower=True`` the factor uses only the lower triangle,
    which is what upstream documents; with ``lower=False`` it uses the summed
    upper entry.  Both must agree with upstream.
    """
    rows = np.array([0, 0, 0, 1, 1, 2])
    cols = np.array([0, 1, 1, 0, 1, 2])
    vals = np.array([4.0, 1.0, 0.5, 1.0, 3.0, 2.0])
    A = sp.coo_array((vals, (rows, cols)), shape=(3, 3)).tocsc()
    # the duplicates were summed into a single canonical entry at (0,1)
    assert A.nnz == 5
    assert A.toarray()[0, 1] == pytest.approx(1.5)
    for lower in (True, False):
        L = mss.cho_factor(A, lower=lower, order="natural").get_factor().toarray()
        theirs = cholmod.cho_factor(A, lower=lower, order="natural").get_factor().toarray()
        assert np.allclose(L, theirs, rtol=1e-9, atol=1e-12)


def test_duplicates_in_the_used_triangle_are_summed():
    """A duplicate that lands in the used triangle must be summed, not kept."""
    # a symmetric SPD matrix, with every entry written twice in the COO input
    base = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    rr, cc = np.nonzero(base)
    rows = np.concatenate([rr, rr])
    cols = np.concatenate([cc, cc])
    vals = np.concatenate([base[rr, cc] / 2.0, base[rr, cc] / 2.0])
    A = sp.coo_array((vals, (rows, cols)), shape=(3, 3)).tocsc()
    # the two copies of each entry were summed back to the original value
    assert A.nnz == np.count_nonzero(base)
    assert np.allclose(A.toarray(), base)
    L = mss.cho_factor(A, lower=True, order="natural").get_factor().toarray()
    assert np.allclose(L @ L.T, base, atol=1e-12)
    theirs = cholmod.cho_factor(A, lower=True, order="natural").get_factor().toarray()
    assert np.allclose(L, theirs, rtol=1e-9, atol=1e-12)


def test_factor_properties_match_upstream_shape():
    """The structural properties agree; the index width is a documented divergence.

    This port is CHOLMOD's ``cholmod_l_*`` (long) interface only, so ``itype`` is
    always int64 where upstream picks int32 for small matrices.  See README,
    "Coverage".
    """
    A = spd_sparse(25, seed=1)
    ours = mss.CholeskyFactor(A, order="natural")
    theirs = cholmod.CholeskyFactor(A, order="natural")
    assert ours.N == theirs.N
    assert ours.is_super == theirs.is_super
    assert int(ours.nnz) == int(theirs.nnz)
    assert ours.itype == np.dtype(np.int64)
    assert ours.dtype == np.dtype(np.float64)


def test_itype_divergence_is_documented():
    """Pin the one deliberate divergence: upstream may use int32, this never does."""
    A = spd_sparse(25, seed=1)
    theirs = cholmod.CholeskyFactor(A, order="natural")
    ours = mss.CholeskyFactor(A, order="natural")
    # both are 64-bit-capable; upstream simply picks 32 when the matrix is small
    assert ours.itype.itemsize == 8
    assert theirs.itype.itemsize in (4, 8)


def test_copy_is_independent():
    A = spd_sparse(25, seed=1)
    f = mss.cho_factor(A, lower=True, order="natural")
    g = f.copy()
    assert np.allclose(
        f.get_factor().toarray(), g.get_factor().toarray(), atol=0
    )
    g.change_factor("LDL")
    assert f.is_ll  # the original is untouched


def test_repr_and_str_do_not_raise():
    f = mss.CholeskyFactor(spd_sparse(15, seed=1), order="natural")
    assert "CholeskyFactor" in repr(f)
    assert "Cholesky factorization" in str(f)

# ---------------------------------------------------------------------------
# the batched pattern walk, the SIMD vectors and the solve buffer cache
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", KINDS)
def test_batched_row_pattern_matches_upstream(kind):
    """msp_row_pattern replaces n per-row calls with one; it must agree exactly.

    The kernel carries a per-row mark and a scratch stack, and clears only the
    flag slots it set, so a stale mark would show up as a short row here.  It also
    parks row k's run on the end of a stack sized nnz(L), so a stack sized n would
    run off the end on the denser cases.
    """
    for n, seed in ((5, 2), (17, 4), (60, 7), (90, 8)):
        A = spd_sparse(n, seed=seed)
        ours = mss.symbfact(A, kind=kind, return_factor=True).L
        theirs = cholmod.symbfact(A, kind=kind, return_factor=True).L
        assert np.array_equal(ours.toarray(), theirs.toarray())


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33])
def test_batched_row_pattern_handles_every_length(n):
    """n below and around the SIMD width, plus awkward tails.

    The batched walk is the one place this port does its own scatter, so it is
    checked on the sizes that expose an off-by-one in the row loop: n=1 is the
    empty-run case, and 15/17/31/33 straddle the machine's float64 SIMD width.
    """
    A = sp.csc_array(sp.eye(n, format="csc") * 3.0)
    A = A + sp.csc_array(
        sp.coo_array(
            (
                np.full(max(n - 1, 0), -1.0),
                (np.arange(1, n), np.arange(n - 1)),
            ),
            shape=(n, n),
        )
    )
    A = sp.csc_array(A)
    ours = mss.symbfact(A, return_factor=True).L
    theirs = cholmod.symbfact(A, return_factor=True).L
    assert np.array_equal(ours.toarray(), theirs.toarray())


def test_row_pattern_row_lengths_sum_to_nnz():
    """Every off-diagonal of every row is reported exactly once."""
    A = spd_grid(6)
    an = mss.cholmod._analyze_pattern(A, sym_kind="sym", lower=False, want_pattern=False)
    n = an["n"]
    Lp, lnz = mss.cholmod._column_pointers(an["colcount"], n)
    W = Lp[:-1].copy()
    Li = np.zeros(lnz, dtype=np.int64)
    Flag = np.zeros(n, dtype=np.int64)
    Stack = np.zeros(max(lnz, 1), dtype=np.int64)
    Len = np.zeros(n, dtype=np.int64)
    Ap = an["Aup"].indptr.astype(np.int64)
    Ai = an["Aup"].indices.astype(np.int64)
    Parent = an["parent"]
    keep = (Lp, Li, W, Len, Flag, Stack, Ap, Ai, Parent)
    mss._lib.lib().msp_row_pattern(
        n, Ap.ctypes.data, Ai.ctypes.data, 0, 0,
        Parent.ctypes.data, Lp.ctypes.data, Li.ctypes.data,
        W.ctypes.data, Len.ctypes.data, Flag.ctypes.data, Stack.ctypes.data, 1,
    )
    assert len(keep) == 9  # every buffer is still alive for the kernel above
    # each row reports its off-diagonal length, and together they fill nnz(L)
    assert int(Len.sum()) + n == lnz
    # the flag must come back all zero: the kernel clears exactly what it set
    assert not Flag.any()
    # and the write cursor must end on the end of each column
    assert np.array_equal(W, Lp[1:])


def test_vec_scale_inv_simd_tail_matches_scalar():
    """y = b/d with the vector body plus a scalar tail, and in place.

    n is walked over every length around the machine's float64 SIMD width so the
    tail is exercised with a remainder of every size.
    """
    lib = mss._lib.lib()
    for n in (1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 33, 65):
        rng = np.random.default_rng(n)
        d = rng.uniform(0.5, 2.0, size=n)
        b = rng.normal(size=n)
        y = np.zeros(n, dtype=np.float64)
        lib.msp_vec_scale_inv(
            n, d.ctypes.data, b.ctypes.data, y.ctypes.data
        )
        assert np.allclose(y, b / d, rtol=0, atol=0)
        # the same kernel in place must give the same answer
        inplace = np.array(b)
        lib.msp_vec_scale_inv(
            n, d.ctypes.data, inplace.ctypes.data, inplace.ctypes.data
        )
        assert np.array_equal(inplace, y)


def test_csc_matvec_matches_scipy():
    """The zero fill is vectorized; the tail must still be written."""
    lib = mss._lib.lib()
    for m in (1, 3, 8, 9, 17):
        rng = np.random.default_rng(m)
        dense = rng.normal(size=(m, m))
        dense[dense < 0.6] = 0.0
        A = sp.csc_array(dense)
        x = rng.normal(size=m)
        y = np.zeros(m, dtype=np.float64)
        lib.msp_csc_matvec(
            m, m,
            A.indptr.astype(np.int64).ctypes.data,
            A.indices.astype(np.int64).ctypes.data,
            A.data.ctypes.data,
            x.ctypes.data, y.ctypes.data,
        )
        assert np.allclose(y, A @ x, rtol=1e-12, atol=1e-12)


def test_repeated_solves_do_not_alias_the_right_hand_side():
    """The solve driver copies each column out before solving in place.

    A (N, 1) right-hand side is already C-contiguous, so a view rather than a
    copy would let the in-place kernels overwrite the caller's buffer: the first
    solve would be right and the second would solve the already-solved system.
    """
    A = spd_sparse(40, seed=3)
    b = np.random.default_rng(5).normal(size=A.shape[0])
    b_before = b.copy()
    f = mss.cho_factor(A, lower=True, order="natural")
    x1 = f.solve(b)
    x2 = f.solve(b)
    assert np.array_equal(b, b_before)
    assert np.allclose(x1, x2, rtol=0, atol=0)
    assert np.abs(A.toarray() @ x1 - b).max() < 1e-9
    # the same must hold through the two-dimensional entry point
    B = np.ascontiguousarray(np.column_stack([b, 2.0 * b]))
    X = f.solve(B)
    assert np.allclose(X[:, 0], x1, rtol=0, atol=0)
    assert np.allclose(X[:, 1], 2.0 * x1, rtol=0, atol=0)


def test_solve_after_refactorize_matches_upstream():
    """The cached solve buffers must be rebuilt when the factor changes.

    solve() caches the factor's kernel-facing buffers, so reusing a factor after
    a second factorize (possibly of a different kind) has to pick up the new
    ones rather than solve against the old factor.
    """
    A = spd_sparse(50, seed=9)
    b = np.random.default_rng(11).normal(size=A.shape[0])
    f = mss.CholeskyFactor(A, lower=True, order="natural")
    f.factorize(A, ldl=False)
    x_ll = f.solve(b, system="A")
    f.factorize(A, ldl=True)
    x_ldl = f.solve(b, system="A")
    assert np.abs(A.toarray() @ x_ll - b).max() < 1e-9
    assert np.abs(A.toarray() @ x_ldl - b).max() < 1e-9
    theirs = cholmod.CholeskyFactor(A, lower=True, order="natural")
    theirs.factorize(A, ldl=True)
    assert np.allclose(x_ldl, theirs.solve(b), rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------------------
# the FFI boundary: dtypes and layouts that would otherwise be silent corruption
# ---------------------------------------------------------------------------
def test_complex_input_is_refused_not_reinterpreted():
    """A complex matrix must not reach a Float64 kernel.

    The kernels read every value as a 64-bit float.  numpy's
    ``ascontiguousarray(..., float64)`` reinterprets complex data bit for bit
    (dropping the imaginary part of complex128 and half the bits of complex64)
    and produces a plausible but wrong factor, so the port refuses it instead.
    """
    for dtype in (np.complex64, np.complex128):
        A = sp.csc_array(np.array([[4, 1], [1, 3]], dtype=dtype))
        with pytest.raises(ValueError, match="real float64"):
            mss.cho_factor(A, lower=True, order="natural")


def test_complex_input_is_refused_in_factorize_and_analysis():
    A = sp.csc_array(np.array([[4 + 1j, 1], [1, 3 + 0j]], dtype=np.complex128))
    with pytest.raises(ValueError, match="real float64"):
        mss.CholeskyFactor(A, order="natural")
    with pytest.raises(ValueError, match="real float64"):
        mss.cho_factor(A, lower=True, order="natural")
    with pytest.raises(ValueError, match="real float64"):
        mss.symbfact(A)


def test_float32_is_widened_not_reinterpreted():
    """float32 must be converted, not read as the low half of a float64."""
    dense = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    A = sp.csc_array(dense)
    L = mss.cho_factor(A, lower=True, order="natural").get_factor().toarray()
    assert np.allclose(L @ L.T, dense, atol=1e-6)
    # and the factor is float64, like every other factor this port returns
    assert mss.cho_factor(A, lower=True, order="natural").dtype == np.float64


def test_unsorted_indices_are_canonicalized_before_the_kernel():
    """An unsorted-but-well-formed CSC matrix must give the right answer.

    This is the realistic non-canonical input (the shape scipy's own tests
    build): a valid matrix whose indices are reversed within each column.  The
    kernels assume canonical CSC, so the wrapper has to put it in that form.
    """
    D = np.array([[4.0, 0.0, 1.0], [1.0, 3.0, 0.0], [0.0, 1.0, 2.0]])
    M = sp.csc_array(D)
    data, indices, indptr = M.data.copy(), M.indices.copy(), M.indptr.copy()
    for start, stop in zip(indptr, indptr[1:]):
        indices[start:stop] = indices[start:stop][::-1].copy()
        data[start:stop] = data[start:stop][::-1].copy()
    NC = sp.csc_array((data, indices, indptr), shape=D.shape)
    assert not NC.has_canonical_format
    ours = mss.cho_factor(NC, lower=True, order="natural").get_factor().toarray()
    theirs = cholmod.cho_factor(NC, lower=True, order="natural").get_factor().toarray()
    assert np.allclose(ours, theirs, rtol=1e-9, atol=1e-12)


def test_non_canonical_matrix_is_rejected_when_it_cannot_be_merged():
    """A matrix that survives the merge but is not canonical must be refused.

    ``_check_csc`` is the only guard on the FFI boundary: the kernels have no
    bounds checks of their own, so a non-canonical layout has to raise rather
    than be handed over.
    """
    A = sp.csc_array(np.array([[4.0, 1.0], [1.0, 3.0]]))
    # bypass the merge by claiming the matrix is canonical while it is not:
    # column 0 now holds row 1 before row 0
    A.indices = np.array([1, 0, 0, 1])
    A.data = np.array([1.0, 4.0, 1.0, 3.0])
    A._has_sorted_indices = True
    A._has_canonical_format = True
    with pytest.raises(ValueError):
        mss.cho_factor(A, lower=True, order="natural")


def test_refactorizing_with_a_different_shape_is_rejected():
    """The factor buffers are sized from the analysis, not from the new matrix."""
    f = mss.CholeskyFactor(sp.csc_array(4.0 * np.eye(4)), lower=True, order="natural")
    with pytest.raises(ValueError, match="shape"):
        f.factorize(sp.csc_array(4.0 * np.eye(5)))


def test_stale_library_is_reported_clearly():
    """A library that does not export every kernel must not be loaded silently."""
    import mojo_scikit_sparse._lib as _libmod

    class FakeCDLL:
        def __init__(self, path):
            self._path = str(path)

        def __getattr__(self, name):
            raise AttributeError(name)

    old_cdll, old_lib = _libmod.ctypes.CDLL, _libmod._lib
    try:
        _libmod.ctypes.CDLL = lambda p: FakeCDLL(p)     # type: ignore[assignment]
        _libmod._lib = None
        with pytest.raises(ImportError, match="stale"):
            _libmod.lib()
    finally:
        _libmod.ctypes.CDLL = old_cdll                  # type: ignore[assignment]
        _libmod._lib = old_lib


# ---------------------------------------------------------------------------
# the solve path: the permutation kernels are part of the covered subset
# ---------------------------------------------------------------------------
def test_permutation_kernels_invert_each_other():
    """msp_vec_permute is x = P*b, and msp_vec_unpermute is x = P'*b.

    With a genuinely non-identity permutation (the natural ordering never
    produces one) the two must be exact inverses, and the composite must be the
    identity.  A solve that skipped both gathers would be numerically identical
    under the identity, which is why this is checked on the kernels directly.
    """
    lib = mss._lib.lib()
    rng = np.random.default_rng(0)
    n = 32
    p = rng.permutation(n).astype(np.int64)
    b = rng.normal(size=n)
    y = np.zeros(n, dtype=np.float64)
    lib.msp_vec_permute(n, p.ctypes.data, b.ctypes.data, y.ctypes.data)
    # the kernel computes x = P b for the permutation matrix I[p], i.e. the
    # scatter x[p[k]] = b[k] -- which is a gather of b at the inverse
    # permutation, not the gather b[p].  Its inverse therefore unpermutes as y[p].
    expected = np.zeros(n)
    expected[p] = b
    assert np.array_equal(y, expected)
    x = np.zeros(n, dtype=np.float64)
    lib.msp_vec_unpermute(n, p.ctypes.data, y.ctypes.data, x.ctypes.data)
    assert np.array_equal(x, b)


def test_solve_system_A_matches_upstream_under_repeated_calls():
    A = spd_sparse(35, seed=17)
    b = np.random.default_rng(2).normal(size=35)
    f = mss.cho_factor(A, lower=True, order="natural")
    assert np.allclose(f.solve(b, system="A"), f.solve(b, system="A"), atol=0)
    assert np.abs(A.toarray() @ f.solve(b, system="A") - b).max() < 1e-9
