"""Times the Mojo factorization and solve against a dense LU on the same matrix.

Run with `pixi run bench`.
"""

from __future__ import annotations

import time

import numpy as np

import mojo_scikit_sparse as mss


def grid5pt(nx, ny):
    """5-point Laplacian on an nx by ny grid: SPD, sparse, and it fills in."""
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
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    vals = np.asarray(vals, dtype=np.float64)
    order = np.lexsort((cols, rows))
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows[order], minlength=n), out=indptr[1:])
    return mss.csr_matrix((vals[order], cols[order], indptr), shape=(n, n))


def timed(fn, repeat):
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def scaling_table():
    print(f"{'grid':>8} {'n':>7} {'nnz(A)':>9} {'nnz(L)':>9} "
          f"{'analyze':>10} {'factor':>10} {'solve':>10} {'fill':>7}")
    for k in (10, 20, 30, 40, 50):
        A = grid5pt(k, k)
        n = A.shape[0]
        b = np.ones(n)
        t_analyze = timed(lambda: mss.analyze(A), 3)
        factor = mss.cholesky(A)
        t_factor = timed(lambda: mss.cholesky(A), 3)
        t_solve = timed(lambda: factor.solve_A(b), 5)
        print(f"{f'{k}x{k}':>8} {n:>7} {A.nnz:>9} {factor.nnz:>9} "
              f"{t_analyze * 1e3:>8.2f}ms {t_factor * 1e3:>8.2f}ms "
              f"{t_solve * 1e3:>8.2f}ms {factor.nnz / A.nnz:>6.2f}x")


def dense_comparison():
    # Dense LU is O(n^3), so this stops well before the sparse side becomes
    # impractical; the point is to show where the crossover sits.
    print()
    print(f"{'grid':>8} {'n':>7} {'dense LU':>10} {'factor+solve':>13} {'speedup':>9}")
    for k in (10, 15, 20, 25, 30):
        A = grid5pt(k, k)
        n = A.shape[0]
        b = np.ones(n)
        dense = A.toarray()
        t_dense = timed(lambda: np.linalg.solve(dense, b), 1)
        t_sparse = timed(lambda: mss.spsolve(A, b), 1)
        residual = np.max(np.abs(dense @ mss.spsolve(A, b) - b))
        assert residual < 1e-8, f"residual {residual:.2e}"
        print(f"{f'{k}x{k}':>8} {n:>7} {t_dense:>9.3f}s {t_sparse:>12.3f}s "
              f"{t_dense / t_sparse:>8.1f}x")


def main():
    scaling_table()
    dense_comparison()


if __name__ == "__main__":
    main()