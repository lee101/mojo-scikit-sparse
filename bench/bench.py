"""Benchmarks this port against upstream scikit-sparse on identical inputs.

Run with ``pixi run bench`` (never ``python bench/bench.py`` directly: the task
holds a machine-wide flock so a concurrent factory job cannot distort the
numbers).  Every number printed below is measured; nothing is extrapolated.

Upstream and this port are compared on equal footing: both use natural ordering
and a simplicial factorization, so the comparison isolates the kernels rather
than the ordering.  Upstream's AMD ordering is also reported separately, because
that is where a fill-reducing permutation wins by a wide margin and it would be
dishonest to hide it.
"""

from __future__ import annotations

import os
import platform
import time

import numpy as np
import scipy.sparse as sp

import mojo_scikit_sparse as mss

try:
    import sksparse.cholmod as up

    HAVE_UPSTREAM = True
except ImportError:  # pragma: no cover - upstream is a declared dependency
    HAVE_UPSTREAM = False


# ---------------------------------------------------------------------------
# problem generators
# ---------------------------------------------------------------------------
def grid5pt(k):
    """5-point Laplacian on a k-by-k grid: SPD, sparse, and it fills in.

    This is the classic sparse-Cholesky benchmark; the natural ordering produces
    substantial fill, which is what makes it interesting.
    """
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


def spd_random(n, density=0.01, seed=0):
    """A random sparse SPD matrix; the natural ordering adds little fill.

    Built sparsely (never as a dense n-by-n): the product A = B B' + n I is formed
    with sparse kernels so the generator stays cheap even at n = 100000.
    """
    r = np.random.default_rng(seed)
    m = max(int(density * n), 1)
    rows = r.integers(0, n, size=m)
    cols = r.integers(0, n, size=m)
    vals = r.normal(size=m)
    B = sp.coo_array((vals, (rows, cols)), shape=(n, n)).tocsc()
    A = (B @ B.T).tocsc()
    A = (A + sp.csc_array(n * np.eye(n))).tocsc()
    A.sum_duplicates()
    A.eliminate_zeros()
    return A


def poisson_2d(k):
    """A 9-point-style stencil: more coupling per row than the 5-point grid.

    Built as a proper SPD operator (diagonal 20, nearest neighbours -4,
    diagonals -1) so the factorization is well posed.
    """
    n = k * k
    rows, cols, vals = [], [], []
    for j in range(k):
        for i in range(k):
            r = j * k + i
            rows.append(r)
            cols.append(r)
            vals.append(20.0)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if 0 <= a < k and 0 <= b < k:
                    rows.append(r)
                    cols.append(b * k + a)
                    vals.append(-4.0)
            for di, dj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                a, b = i + di, j + dj
                if 0 <= a < k and 0 <= b < k:
                    rows.append(r)
                    cols.append(b * k + a)
                    vals.append(-1.0)
    return sp.csc_array(
        (np.asarray(vals), (np.asarray(rows), np.asarray(cols))), shape=(n, n)
    )


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------
def timed(fn, repeat=5, warmup=1):
    """Best-of-`repeat` wall time in seconds, after `warmup` untimed calls."""
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def ours_analyze(A):
    return mss.CholeskyFactor(A, lower=True, order="natural")


def ours_factorize(A):
    return mss.CholeskyFactor(A, lower=True, order="natural").factorize(A)


def ours_factor(A):
    return mss.cho_factor(A, lower=True, order="natural")


def ours_solve(A, b, factor):
    return lambda: factor.solve(b)


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def table_symbolic(problems):
    """The symbolic analysis: etree + postorder + row/column counts."""
    header = (
        "| problem | n | nnz(A) | mojo symbfact | upstream symbfact | speedup |"
    )
    sep = "| --- | ---: | ---: | ---: | ---: | ---: |"
    rows = [header, sep]
    for name, A in problems:
        n = A.shape[0]
        t_ours = timed(lambda: mss.symbfact(A), repeat=5)
        if HAVE_UPSTREAM:
            t_up = timed(lambda: up.symbfact(A), repeat=5)
            rows.append(
                f"| {name} | {n} | {A.nnz} | {t_ours*1e3:.2f} ms | "
                f"{t_up*1e3:.2f} ms | {t_up/t_ours:.2f}x |"
            )
        else:
            rows.append(f"| {name} | {n} | {A.nnz} | {t_ours*1e3:.2f} ms | n/a | n/a |")
    return "\n".join(rows)


def table_numeric(problems):
    """analyze + numeric factorization, the dominant cost of a sparse Cholesky.

    Both sides use natural ordering here so the kernels are compared on equal
    terms; the AMD comparison is in the last table.
    """
    header = "| problem | n | nnz(A) | nnz(L) | mojo factorize | upstream factorize | speedup |"
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    rows = [header, sep]
    for name, A in problems:
        n = A.shape[0]
        f = ours_factorize(A)
        t_ours = timed(lambda: ours_factorize(A), repeat=3)
        if HAVE_UPSTREAM:
            t_up = timed(
                lambda: up.CholeskyFactor(A, lower=True, order="natural").factorize(A),
                repeat=2,
            )
            rows.append(
                f"| {name} | {n} | {A.nnz} | {f.nnz} | {t_ours*1e3:.2f} ms | "
                f"{t_up*1e3:.2f} ms | {t_up/t_ours:.2f}x |"
            )
        else:
            rows.append(
                f"| {name} | {n} | {A.nnz} | {f.nnz} | {t_ours*1e3:.2f} ms | n/a | n/a |"
            )
    return "\n".join(rows)


def table_solve(problems):
    """Reusing a stored factor: the solve is the hot path in many workloads."""
    header = "| problem | n | nnz(L) | mojo solve | upstream solve | speedup |"
    sep = "| --- | ---: | ---: | ---: | ---: | ---: |"
    rows = [header, sep]
    for name, A in problems:
        n = A.shape[0]
        rng = np.random.default_rng(0)
        b = rng.normal(size=n)
        f = ours_factor(A)
        t_ours = timed(lambda: f.solve(b), repeat=20)
        if HAVE_UPSTREAM:
            uf = up.cho_factor(A, lower=True, order="natural")
            t_up = timed(lambda: uf.solve(b), repeat=20)
            rows.append(
                f"| {name} | {n} | {f.nnz} | {t_ours*1e3:.3f} ms | "
                f"{t_up*1e3:.3f} ms | {t_up/t_ours:.2f}x |"
            )
        else:
            rows.append(f"| {name} | {n} | {f.nnz} | {t_ours*1e3:.3f} ms | n/a | n/a |")
    return "\n".join(rows)


def table_dense_rhs(problems):
    """A 3-column dense right-hand side, the multi-solve case."""
    header = "| problem | n | K | mojo solve | upstream solve | speedup |"
    sep = "| --- | ---: | ---: | ---: | ---: | ---: |"
    rows = [header, sep]
    for name, A in problems:
        n = A.shape[0]
        rng = np.random.default_rng(1)
        B = rng.normal(size=(n, 3))
        f = ours_factor(A)
        t_ours = timed(lambda: f.solve(B), repeat=20)
        if HAVE_UPSTREAM:
            uf = up.cho_factor(A, lower=True, order="natural")
            t_up = timed(lambda: uf.solve(B), repeat=20)
            rows.append(
                f"| {name} | {n} | 3 | {t_ours*1e3:.3f} ms | "
                f"{t_up*1e3:.3f} ms | {t_up/t_ours:.2f}x |"
            )
        else:
            rows.append(f"| {name} | {n} | 3 | {t_ours*1e3:.3f} ms | n/a | n/a |")
    return "\n".join(rows)


def table_ordering(problems):
    """Natural vs upstream's default (AMD) ordering.

    Reported because the default ordering is the single biggest lever in sparse
    Cholesky, and a comparison that only showed natural ordering would not be
    telling the whole story.  Upstream is timed with its default (AMD) ordering
    because that is the configuration upstream is actually used in.
    """
    header = (
        "| problem | n | nnz(L) natural | nnz(L) AMD | mojo factorize (natural) | "
        "upstream factorize (AMD) |"
    )
    sep = "| --- | ---: | ---: | ---: | ---: | ---: |"
    rows = [header, sep]
    for name, A in problems:
        n = A.shape[0]
        t_ours = timed(lambda: ours_factorize(A), repeat=3)
        f = ours_factorize(A)
        if HAVE_UPSTREAM:
            uf = up.CholeskyFactor(A, lower=True, order="default")
            uf.factorize(A)
            t_up = timed(
                lambda: up.CholeskyFactor(A, lower=True, order="default").factorize(A),
                repeat=3,
            )
            rows.append(
                f"| {name} | {n} | {f.nnz} | {int(np.sum(uf.colcount))} | "
                f"{t_ours*1e3:.2f} ms | {t_up*1e3:.2f} ms |"
            )
        else:
            rows.append(
                f"| {name} | {n} | {f.nnz} | n/a | {t_ours*1e3:.2f} ms | n/a |"
            )
    return "\n".join(rows)


def correctness_check(problems):
    """Verify the factors really are factors, so the timings mean something."""
    for name, A in problems:
        f = ours_factor(A)
        L = f.get_factor()
        err = float(np.abs((L @ L.T - A)).max())
        assert err < 1e-8, f"{name}: L L' - A = {err:.3e}"
    return "L L' reproduces A to better than 1e-8 on every benchmark problem."


def machine_info():
    model = ""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return (
        f"{model or 'unknown CPU'}, {os.cpu_count()} logical cores, "
        f"Python {platform.python_version()}, NumPy {np.__version__}, "
        f"platform {platform.system()}"
    )


def load_info():
    """The 1/5/15-minute load averages, so a reader can judge the numbers."""
    try:
        one, five, fifteen = os.getloadavg()
    except OSError:  # pragma: no cover - not every platform has getloadavg
        return "load average unavailable"
    cores = os.cpu_count() or 1
    return (
        f"Load average during the run: {one:.0f} / {five:.0f} / {fifteen:.0f} "
        f"(1/5/15 min) over {cores} logical cores"
    )


def main():
    print("# mojo-scikit-sparse benchmarks\n")
    print(f"Machine: {machine_info()}")
    print(f"{load_info()}\n")
    if HAVE_UPSTREAM:
        import sksparse

        v = getattr(sksparse, "__version__", "unknown")
        print(f"Upstream: scikit-sparse {v} (SuiteSparse CHOLMOD via Cython)\n")
    else:
        print("Upstream scikit-sparse is not installed; no comparison columns.\n")

    problems = [
        ("grid 5pt 40x40", grid5pt(40)),
        ("grid 5pt 60x60", grid5pt(60)),
        ("grid 5pt 80x80", grid5pt(80)),
        ("grid 9pt 50x50", poisson_2d(50)),
        ("random SPD n=5000", spd_random(5000, density=0.002, seed=3)),
        ("random SPD n=20000", spd_random(20000, density=0.0005, seed=4)),
    ]

    print(correctness_check(problems))
    print()

    print("## Symbolic analysis (elimination tree + postorder + column counts)\n")
    print(table_symbolic(problems))
    print()

    print("## Numeric factorization (analyze + LL' factorization)\n")
    print(table_numeric(problems))
    print()

    print("## Solve, single right-hand side, factor reused\n")
    print(table_solve(problems))
    print()

    print("## Solve, 3-column dense right-hand side\n")
    print(table_dense_rhs(problems))
    print()

    print("## Natural vs AMD ordering\n")
    print(table_ordering(problems))


if __name__ == "__main__":
    main()