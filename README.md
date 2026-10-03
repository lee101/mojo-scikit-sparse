# mojo-scikit-sparse

A Mojo port of the sparse Cholesky kernels in
[scikit-sparse](https://github.com/python/cscikit-sparse) (SuiteSparse CHOLMOD),
exposed through a Python API that mirrors `sksparse.cholmod`'s names and
signatures.

The compute-bound routines -- the elimination tree, its postordering, the
row/column counts of the factor, the symbolic factor pattern, the numeric row
factorization and the triangular solves -- are ported from CHOLMOD's C sources
into `src/ported.mojo` and run behind a C ABI. The Python package owns the
buffers and drives the kernels in upstream's order.

## Install

From a source checkout, with [pixi](https://pixi.sh) (which also pins the Mojo
toolchain):

```bash
pixi install
pixi run build      # src/ported.mojo -> dist/libmojo-scikit-sparse.so
pixi run test       # parity tests against the real upstream scikit-sparse
pixi run bench      # benchmarks vs upstream (holds a machine-wide flock)
```

`pixi run test` depends on `build`, so a fresh checkout needs only
`pixi install && pixi run test`.

The shared library is compiled separately from the Python package, so a `pip`
install has to be given it. Copy (or symlink) the built library next to the
package and install from there:

```bash
cp dist/libmojo-scikit-sparse.so python/mojo_scikit_sparse/
pip install .
```

At import time `_lib` looks for the library in `MOJO_SCIKIT_SPARSE_LIB`, then
next to the package, then in the repository's `dist/`; if it finds none it says so
and names `pixi run build`.

Upstream `scikit-sparse` is a declared dependency purely so the parity tests and
the benchmark have something real to compare against; the library itself does not
import it.

## Usage

```python
import numpy as np
import scipy.sparse as sp
import mojo_scikit_sparse as mss

# a symmetric positive definite sparse matrix
n = 6
A = sp.csc_array(np.diag([4.0] * n) + np.diag([-1.0] * (n - 1), 1)
                 + np.diag([-1.0] * (n - 1), -1))

# the Cholesky factor R with R R' = A
R, perm = mss.cholesky(A, lower=True, order="natural")

# or solve directly
b = np.ones(n)
x = mss.cho_solve(A, b, lower=True, order="natural")

# the symbolic analysis on its own
count, h, parent, post = mss.symbfact(A)[:4]

# reuse a stored factor across many right-hand sides
f = mss.cho_factor(A, lower=True, order="natural")
x = f.solve(b)
logdet = f.logdet()
```

This is the same call sequence as upstream:

```python
from sksparse.cholmod import cholesky, cho_solve, symbfact   # upstream
import mojo_scikit_sparse as mss                              # this port
```

## Coverage

### Covered, and verified against upstream

Every function below is asserted against `sksparse.cholmod` on the same inputs by
a test in `tests/test_parity.py`. Where the result is integral (the elimination
tree, the postordering, the column counts, the symbolic factor pattern) the port
reproduces upstream **exactly** (`np.array_equal`); where it is floating point it
agrees to within the tolerance the test asserts (`rtol=1e-7` or tighter).

| Upstream | Here | Verified by |
| --- | --- | --- |
| `symbfact(A, kind, lower, return_factor)` | same | `test_symbfact_matches_upstream`, `test_symbfact_factor_pattern_matches_upstream`: all four `kind`s; `count`, `h`, `parent`, `post` and the symbolic `L` compared with `array_equal` |
| `etree(A, kind, return_post)` | same | `test_etree_matches_upstream`, `test_etree_postorder_is_a_permutation`: all four `kind`s, plus the docstring vectors |
| `CholeskyFactor(...)`, `.factorize(A, ldl, beta)` | same | `test_cholesky_factor_matches_upstream`, `test_ldl_factor_matches_upstream`, `test_colcount_matches_upstream`, `test_beta_matches_upstream` |
| `.get_factor(kind, lower)` | same | `test_cholesky_factor_matches_upstream`, `test_ldl_factor_matches_upstream` |
| `.get_perm()`, `.copy()`, `.change_factor(kind)` | same | `test_copy_is_independent`, `test_cholesky_returns_factor_and_permutation` |
| `cho_factor`, `ldl_factor`, `cholesky`, `ldl`, `cho_solve` | same | `test_cholesky_returns_factor_and_permutation`, `test_ldl_returns_factor_diagonal_and_permutation`, `test_cho_solve_matches_upstream` |
| `.solve(b, system=...)` | same | `test_solve_matches_upstream` (`A`, `LDLt`, `L`, `Lt`), `test_ldl_solve_matches_upstream`, `test_composite_systems_are_self_consistent` (`LD`, `DLt`, `D`), `test_two_dimensional_rhs` |
| `.logdet()`, `.slogdet()`, `.det()`, `.inv()` | same | `test_logdet_matches_upstream`, `test_det_matches_numpy`, `test_inv_matches_upstream` |
| `sym_kind="row"` / `"col"` (A·A' and A'·A) | same | `test_unsymmetric_kinds_match_upstream` |
| `beta` (shift the diagonal before factoring) | same | `test_beta_matches_upstream`, `test_beta_defines_the_matrix` |
| `CholmodError`, `CholmodNotPositiveDefiniteError`, `CholmodWarning` | same | `test_not_positive_definite_raises`, `test_upstream_raises_the_same_way`, `test_zero_matrix_is_rejected` |
| `is_ll`, `is_numeric`, `is_super`, `N`, `nnz`, `colcount`, `order`, `perm`, `itype`, `dtype`, `rcond`, `repr`, `str` | same | `test_factor_properties_match_upstream_shape`, `test_repr_and_str_do_not_raise` |

Kernels ported, in CHOLMOD's own source order, each in its own `@export`:

| CHOLMOD routine | Mojo kernel | Verified by |
| --- | --- | --- |
| `cholmod_transpose` | `msp_transpose` | every `symbfact`/`etree` case with a transpose, via `test_symbfact_matches_upstream` |
| `cholmod_etree` / `update_etree` | `msp_etree` | `test_etree_matches_upstream`, `test_etree_matches_upstream_docstring_vectors` |
| `cholmod_postorder` / `dfs` | `msp_postorder` | `test_symbfact_matches_upstream`, `test_etree_postorder_is_a_permutation` |
| `cholmod_rowcolcounts` / `initialize_node` / `process_edge` / `finalize_node` | `msp_rowcolcounts` | `test_colcount_matches_upstream` (`array_equal` on the column counts) |
| `cholmod_row_subtree` (the `SUBTREE` macro) | `msp_row_pattern` | `test_batched_row_pattern_matches_upstream`, `test_batched_row_pattern_handles_every_length`, `test_row_pattern_row_lengths_sum_to_nnz` |
| `cholmod_rowfac` (`t_cholmod_rowfac_worker.c`) | `msp_rowfac` | `test_cholesky_factor_matches_upstream`, `test_reconstruction_is_exact` |
| `cholmod_lsolve` / `cholmod_ltsolve` | `msp_lsolve`, `msp_ltsolve` | `test_solve_matches_upstream`, `test_solve_residual_is_small` |
| the D step and the permutation of `cholmod_solve` | `msp_vec_scale_inv`, `msp_vec_permute`, `msp_vec_unpermute` | `msp_vec_scale_inv`: `test_vec_scale_inv_simd_tail_matches_scalar`; the two permutations: `test_permutation_kernels_invert_each_other` |
| `csc_matvec` | `msp_csc_matvec` | `test_csc_matvec_matches_scipy` |

### Not covered, and why

These are **not** silent omissions. Each one either raises `NotImplementedError`
or a `ValueError`, rather than quietly doing something different. The two
exceptions are called out below as the places where a result can differ from
upstream.

- **Fill-reducing orderings.** `CholeskyFactor(..., order=...)` accepts only
  `"natural"`, `None`, and `"default"`. Upstream's `"default"` tries AMD then
  METIS; neither is ported, so **`"default"` is treated as `"natural"`** here.
  This is the first of the two places where a silently different *answer* is
  possible, and it is why the benchmark reports the natural-vs-AMD comparison
  explicitly: on an 80x80 grid, AMD cuts `nnz(L)` from 512079 to 120766 (4.2x), so
  a caller who silently got natural ordering where they expected AMD would see a
  much larger factor. Every other ordering (`"best"`, `"metis"`, `"nesdis"`,
  `"amd"`, `"colamd"`, `"postordered"`) is rejected with a `ValueError` rather than
  downgraded (`test_unsupported_ordering_is_rejected`).
- **`update`, `downdate`, `rowadd`, `rowdel`, `resymbol`** -- the `cholmod_updown`
  and `cholmod_rowadd` families. These are substantial separate algorithms with
  their own supernode handling; they raise `NotImplementedError`
  (`test_unported_methods_raise_not_implemented`).
- **Supernodal factorization.** `supernodal_mode` is accepted and validated, but
  the kernels are simplicial throughout, so `is_super` is always `False`. This is
  the second place where a caller can get a correct but much slower result than
  upstream: it is why the benchmark compares against CHOLMOD's supernodal path,
  and this port loses to it on most problems.
- **`int32` indices.** The kernels are CHOLMOD's `cholmod_l_*` (long) entry
  points, so `itype` is always `int64` where upstream picks `int32` below 2^31
  entries. The index values are identical; only the storage width differs, and the
  wrapper widens SciPy's `int32` index arrays before the ABI
  (`test_itype_divergence_is_documented`).
- **Complex and single-precision data.** The kernels read every value as a 64-bit
  float. `float32` input is widened, which is what CHOLMOD does on load. Complex
  input is **rejected with a `ValueError`** rather than reinterpreted bit for bit
  into a float64 kernel, which would silently drop the imaginary part
  (`test_complex_input_is_refused_not_reinterpreted`,
  `test_float32_is_widened_not_reinterpreted`).
- **Sparse right-hand sides.** `.solve(b)` accepts a sparse `b` and returns the
  right answer, but it is solved one dense column at a time rather than by a
  ported `cholmod_spsolve`, so it is slower than upstream on that path.
- **CHOLMOD's 2- and 3-column supernode paths** in `lsolve`/`ltsolve`. Only the
  single-column path is ported; those supernode paths are a performance
  optimization over exactly the same operations.
- **`fset` / restricted supernodal solves** and the rest of CHOLMOD's optional
  machinery, which upstream does not expose either.

## How it works

**FFI strategy.** `src/ported.mojo` is a single compilation unit exporting twelve
`abi("C")` functions via `@export`. Buffers cross the ABI as plain `Int` addresses
and are rebuilt as `Pointer[Float64, AnyOrigin[mut=True]]` inside each kernel;
nothing crosses by pointer type, and no kernel allocates. Every buffer is owned by
a numpy array on the Python side. `python/mojo_scikit_sparse/_lib.py` holds one
table of `ctypes` signatures whose entries are positional and must match the Mojo
parameter list exactly.

Four invariants live on the Python side rather than in the kernels, because
getting any of them wrong is a silent memory error rather than an exception:

- **Index arrays are widened to `int64` before crossing the ABI.** The kernels are
  the `cholmod_l_*` (long) ports and read every index as an 8-byte `Int`. SciPy
  hands back `int32` index arrays for small matrices, so passing one straight
  through reads half of every entry as garbage and segfaults.
- **Every buffer passed to a kernel is a named object.** ctypes keeps no reference
  to a numpy array passed as an argument, so an array constructed inline as a
  temporary can be freed before the kernel reads it. Scratch is allocated into a
  local or a workspace attribute first, then its address is taken; `_solve_buffers`
  returns a `keepalive` tuple for exactly this reason.
- **The input is canonical CSC.** The kernels have no bounds checks of their own,
  so `_check_csc` verifies that `indptr`/`indices`/`data` agree and that row
  indices are strictly increasing within each column -- the port of CHOLMOD's own
  `A->sorted` requirement. Non-canonical input is rebuilt (duplicates summed,
  indices sorted); a matrix that still fails the check raises rather than being
  handed over (`test_non_canonical_matrix_is_rejected_when_it_cannot_be_merged`).
- **Values are float64.** `f64` refuses complex data rather than letting numpy
  reinterpret the bits, and the analysis and factorization entry points reject a
  complex matrix before any address crosses the ABI.

**Memory layout.** Everything is CSC (`indptr`, `indices`, `data`) with `int64`
indices and `float64` values -- the layout `sksparse` itself uses, so a matrix can
be handed to either library with no conversion. The factor `L` is stored packed
and column-compressed with the **diagonal entry first in each column**, which is
what CHOLMOD's `Lp`/`Li`/`Lx` plus `Lnz` describe and what lets the triangular
solves walk columns directly. `Lnz` is the write cursor during `msp_rowfac` and is
read back as `Lp[j+1]-Lp[j]` by the solve path.

**Why the wrapper transposes.** CHOLMOD's `cholmod_etree` and `cholmod_rowfac`
only accept the *upper* triangle (`cholmod_rowcolcounts` only accepts the lower
one -- each rejects the other with `CHOLMOD_INVALID`). Upstream resolves this
inside `cholmod_analyze` and `cholmod_factorize`, which transpose once and hand
each routine the view it wants. This port keeps that structure visible in
`cholmod.py`: `_analyze_pattern` computes `Aup`/`Alo` and passes each to the
kernel that requires it, and `factorize` transposes when `lower=True`. The kernels
themselves stay a direct transcription of the C.

**Where the time goes.** `msp_rowfac` is a left-looking row factorization whose
inner loop is a gather (`W[t] -= Lx[p] * y` with `t = Li[p]`), which is why the
numeric factorization scales with the density of the factor's columns rather than
its flop count; on a nearly diagonal matrix that is O(n^2). The triangular solves
are memory-bound scalar loops. Neither has a data-parallel form that a compiler
can exploit, which is the short answer for the benchmark below.

## Benchmarks

Run with `pixi run bench`. Every number below comes from one run of that command
on the machine described here; nothing is extrapolated. `bench/bench.py` prints
the same tables.

**Machine:** Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz, 72 logical cores,
Python 3.13.14, NumPy 2.5.1, Linux. Upstream is scikit-sparse 0.5.0 (SuiteSparse
CHOLMOD via Cython).

**Measurement caveat.** The `bench` task holds a machine-wide `flock`
(`/tmp/mojo-bench.lock`) so a concurrent job cannot start a second benchmark. That
only stops other *benchmarks*; it cannot quiet the host. The run quoted below was
taken with a load average of 74 / 83 / 83 (1/5/15 min) on 72 cores -- the machine
was busy with unrelated work throughout. Both sides of every comparison are
single-threaded and were measured under exactly the same conditions, so the ratios
are fair, but the absolute milliseconds are pessimistic for both and should be
reproduced on an idle host before being quoted.

**Run-to-run spread.** The tables come from one recorded run, as above. Three runs
of this benchmark on this same loaded host were compared cell by cell:

| what | worst spread over 3 runs |
| --- | --- |
| this port's own timings | 1.6x, on one row of the single-rhs solve table |
| upstream's timings | 1.8x, on the same table's rows |
| upstream's grid 80x80 *natural* factorize | 1.6x (7.1 s to 11.5 s) |

So the port's numbers are no noisier than upstream's, but on this host neither
side is reproducible to better than about 1.5x on the smaller rows, and the 223x
grid 80x80 row is the least reproducible number in this document. Treat the
millisecond columns as indicative and the ratios as the result.

Both sides use natural ordering in the first four tables so the *kernels* are
compared on equal terms. The last table reports upstream's default (AMD) ordering
separately, because ordering dominates sparse Cholesky in practice.

Every problem is verified before timing: `L L'` reproduces `A` to better than 1e-8
on all of them, so these are timings of real factorizations.

### Symbolic analysis (elimination tree + postorder + column counts)

| problem | n | nnz(A) | mojo symbfact | upstream symbfact | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| grid 5pt 40x40 | 1600 | 7840 | 0.51 ms | 0.12 ms | 0.24x |
| grid 5pt 60x60 | 3600 | 17760 | 0.75 ms | 0.25 ms | 0.34x |
| grid 5pt 80x80 | 6400 | 31680 | 1.03 ms | 0.43 ms | 0.42x |
| grid 9pt 50x50 | 2500 | 21904 | 0.69 ms | 0.24 ms | 0.36x |
| random SPD n=5000 | 5000 | 5000 | 0.57 ms | 0.14 ms | 0.24x |
| random SPD n=20000 | 20000 | 20000 | 1.39 ms | 0.52 ms | 0.38x |

### Numeric factorization (analyze + LL' factorization)

| problem | n | nnz(A) | nnz(L) | mojo factorize | upstream factorize | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| grid 5pt 40x40 | 1600 | 7840 | 64039 | 4.76 ms | 2.49 ms | 0.52x |
| grid 5pt 60x60 | 3600 | 17760 | 216059 | 18.05 ms | 7.56 ms | 0.42x |
| grid 5pt 80x80 | 6400 | 31680 | 512079 | 51.51 ms | 11492.36 ms | 223.09x |
| grid 9pt 50x50 | 2500 | 21904 | 127450 | 10.58 ms | 5.38 ms | 0.51x |
| random SPD n=5000 | 5000 | 5000 | 5000 | 13.04 ms | 0.32 ms | 0.02x |
| random SPD n=20000 | 20000 | 20000 | 20000 | 178.66 ms | 1.12 ms | 0.01x |

### Solve, single right-hand side, factor reused

| problem | n | nnz(L) | mojo solve | upstream solve | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| grid 5pt 40x40 | 1600 | 64039 | 0.232 ms | 0.166 ms | 0.72x |
| grid 5pt 60x60 | 3600 | 216059 | 0.685 ms | 0.503 ms | 0.73x |
| grid 5pt 80x80 | 6400 | 512079 | 1.505 ms | 1.714 ms | 1.14x |
| grid 9pt 50x50 | 2500 | 127450 | 0.525 ms | 0.464 ms | 0.88x |
| random SPD n=5000 | 5000 | 5000 | 0.104 ms | 0.062 ms | 0.60x |
| random SPD n=20000 | 20000 | 20000 | 0.180 ms | 0.108 ms | 0.60x |

### Solve, 3-column dense right-hand side

| problem | n | K | mojo solve | upstream solve | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| grid 5pt 40x40 | 1600 | 3 | 0.669 ms | 0.249 ms | 0.37x |
| grid 5pt 60x60 | 3600 | 3 | 2.108 ms | 1.009 ms | 0.48x |
| grid 5pt 80x80 | 6400 | 3 | 4.827 ms | 1.903 ms | 0.39x |
| grid 9pt 50x50 | 2500 | 3 | 1.237 ms | 0.474 ms | 0.38x |
| random SPD n=5000 | 5000 | 3 | 0.206 ms | 0.093 ms | 0.45x |
| random SPD n=20000 | 20000 | 3 | 0.610 ms | 0.428 ms | 0.70x |

### Natural vs AMD ordering

| problem | n | nnz(L) natural | nnz(L) AMD | mojo factorize (natural) | upstream factorize (AMD) |
| --- | ---: | ---: | ---: | ---: | ---: |
| grid 5pt 40x40 | 1600 | 64039 | 20771 | 4.51 ms | 1.56 ms |
| grid 5pt 60x60 | 3600 | 216059 | 59765 | 17.88 ms | 4.21 ms |
| grid 5pt 80x80 | 6400 | 512079 | 120766 | 50.95 ms | 217.12 ms |
| grid 9pt 50x50 | 2500 | 127450 | 58129 | 10.03 ms | 3.72 ms |
| random SPD n=5000 | 5000 | 5000 | 5000 | 12.76 ms | 0.51 ms |
| random SPD n=20000 | 20000 | 20000 | 20000 | 174.32 ms | 2.05 ms |

### Reading these numbers honestly

**This port is slower than upstream on most of these, and the reasons are
specific rather than general.**

- **The symbolic analysis is 2.6-4.2x slower than upstream.** `cholmod_etree`,
  `cholmod_postorder` and `cholmod_rowcolcounts` are short, pointer-chasing
  algorithms over integer arrays. There is no vectorizable arithmetic in them and
  no data-parallelism to exploit -- the elimination tree is inherently sequential
  -- so there is nothing for a compiler to do that CHOLMOD's tuned C does not
  already do. The absolute cost is small (1.4 ms even at n=20000).
- **The numeric factorization is 2.0-2.4x slower on the grid problems.** Same
  reason: `cholmod_rowfac` is a sequential left-looking elimination with a gather
  in its inner loop, and this port keeps upstream's loop structure rather than
  restructuring it.
- **The one large win (223x on grid 80x80) is an upstream pathology, not a Mojo
  advantage.** Upstream's *natural*-ordering simplicial path takes 11.5 s where its
  own AMD path takes 0.2 s on the same matrix; that ratio came out between 138x
  and 266x across the runs taken. CHOLMOD's natural simplicial factorization
  degrades sharply once the factor is large. Against the configuration upstream is
  actually used in -- AMD -- this port is slower on the grid problems too
  (17.9 ms vs 4.2 ms at n=3600). The 223x row is reported because hiding it would
  misrepresent the comparison, not because it is the number to quote.
- **The 0.01-0.02x on the random SPD cases is the clearest loss.** These matrices
  are nearly diagonal (`nnz(L) == nnz(A)`), so every column of L is dense and
  `msp_rowfac`'s forward substitution is O(n) per column with no locality:
  O(n^2) = 4x10^8 operations at n=20000. Upstream's supernodal path with
  multithreaded BLAS does not suffer this. This is the clearest case where the port
  should not be used.
- **Solves are 0.6-1.1x.** A triangular solve is memory-bound; the kernels here are
  scalar, single-threaded loops over the factor, while upstream's supernodal solves
  reach multithreaded BLAS. Solving is also per-column on the Python side, which
  costs something at these sizes and is why the 3-column table is worse than the
  1-column one. The single grid 80x80 row where the port is ahead (1.14x) is the
  one cell that reversed sign between runs (0.77x in another run, when upstream
  measured 1.18 ms instead of 1.71 ms), so it is noise rather than a result.

The honest summary: this port is correct and complete for its subset, it wins only
where upstream's natural ordering collapses, and it loses to CHOLMOD's supernodal
BLAS on the kernel-bound cases. The next pass can fix the losses -- the numeric
factorization and the solves are the right targets, and the structure is intact
enough to accelerate without rewriting the algorithms.

## Tests

`pixi run test` runs 123 tests from `tests/test_parity.py`. Every one of them
either compares against upstream `scikit-sparse` on the same input or checks an
invariant that has no upstream counterpart -- the elimination tree, the
postordering, the column counts, the symbolic pattern, the numeric LL' and LDL'
factors, all seven solve systems, `det`/`logdet`/`inv`, the unsymmetric
`sym_kind`s, the error paths, the SIMD tails, and the FFI boundary guards. If
upstream `scikit-sparse` is not importable the whole module skips rather than
quietly testing the library against itself.

## License

MIT; see `LICENSE`. Upstream `scikit-sparse` is BSD-2-Clause and CHOLMOD is
LGPL-2.1-or-later; this is an independent reimplementation in Mojo and does not
link against either.