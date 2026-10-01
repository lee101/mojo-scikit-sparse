# Sparse linear algebra kernels for mojo-scikit-sparse, exported over the C ABI.
#
# Buffers cross the ABI as Int addresses and are rebuilt as pointers inside the
# wrapper.  Callers own every allocation; nothing here returns heap memory.

comptime F64P = Pointer[Float64, AnyOrigin[mut=True]]
comptime I64P = Pointer[Int, AnyOrigin[mut=True]]


@export("msp_csc_matvec")
def msp_csc_matvec(m: Int, n: Int, Ap_: Int, Ai_: Int, Ax_: Int, x_: Int, y_: Int) abi("C"):
    # y = A * x for A stored column-compressed.  Rows are zeroed once, then each
    # column contributes an axpy scattered into y.  A column whose x entry is
    # exactly zero contributes nothing, so it is skipped outright.
    var Ap = I64P(unsafe_from_address=Ap_)
    var Ai = I64P(unsafe_from_address=Ai_)
    var Ax = F64P(unsafe_from_address=Ax_)
    var x = F64P(unsafe_from_address=x_)
    var y = F64P(unsafe_from_address=y_)
    for i in range(m):
        y.unsafe_store(i, 0.0)
    for j in range(n):
        var xj = x.unsafe_load(j)
        if xj == 0.0:
            continue
        for p in range(Ap.unsafe_load(j), Ap.unsafe_load(j + 1)):
            var i = Ai.unsafe_load(p)
            y.unsafe_store(i, y.unsafe_load(i) + Ax.unsafe_load(p) * xj)


@export("msp_csr_matvec")
def msp_csr_matvec(m: Int, n: Int, Ap_: Int, Ai_: Int, Ax_: Int, x_: Int, y_: Int) abi("C"):
    # y = A * x for A stored row-compressed.  One pass over nnz with a single
    # accumulator per row.
    var Ap = I64P(unsafe_from_address=Ap_)
    var Ai = I64P(unsafe_from_address=Ai_)
    var Ax = F64P(unsafe_from_address=Ax_)
    var x = F64P(unsafe_from_address=x_)
    var y = F64P(unsafe_from_address=y_)
    for i in range(m):
        var s = 0.0
        for p in range(Ap.unsafe_load(i), Ap.unsafe_load(i + 1)):
            s += Ax.unsafe_load(p) * x.unsafe_load(Ai.unsafe_load(p))
        y.unsafe_store(i, s)


@export("msp_csc_transpose")
def msp_csc_transpose(m: Int, n: Int, Ap_: Int, Ai_: Int, Ax_: Int, Bp_: Int, Bi_: Int, Bx_: Int, w_: Int) abi("C"):
    # B = A^T with B column-compressed; nnz(B) == nnz(A) and B has shape n x m.
    # Counting sort through row pointers, then a fill pass.
    var Ap = I64P(unsafe_from_address=Ap_)
    var Ai = I64P(unsafe_from_address=Ai_)
    var Ax = F64P(unsafe_from_address=Ax_)
    var Bp = I64P(unsafe_from_address=Bp_)
    var Bi = I64P(unsafe_from_address=Bi_)
    var Bx = F64P(unsafe_from_address=Bx_)
    Bp.unsafe_store(0, 0)
    for p in range(Ap.unsafe_load(n)):
        var i = Ai.unsafe_load(p)
        Bp.unsafe_store(i + 1, Bp.unsafe_load(i + 1) + 1)
    for i in range(m):
        Bp.unsafe_store(i + 1, Bp.unsafe_load(i + 1) + Bp.unsafe_load(i))
    var w = I64P(unsafe_from_address=w_)
    for i in range(m):
        w.unsafe_store(i, Bp.unsafe_load(i))
    for j in range(n):
        for p in range(Ap.unsafe_load(j), Ap.unsafe_load(j + 1)):
            var i = Ai.unsafe_load(p)
            var q = w.unsafe_load(i)
            Bi.unsafe_store(q, j)
            Bx.unsafe_store(q, Ax.unsafe_load(p))
            w.unsafe_store(i, q + 1)


@export("msp_csc_lower_solve")
def msp_csc_lower_solve(n: Int, Ap_: Int, Ai_: Int, Ax_: Int, b_: Int, x_: Int) abi("C"):
    # Solves L x = b for unit-diagonal lower triangular L stored
    # column-compressed; x is initialised from b.
    #
    # Scatter down the columns.  Column k of a lower triangular L holds only
    # rows i > k, and by the time column k is reached every contribution from
    # columns j < k has already landed in x[k], so x[k] is final.
    var Ap = I64P(unsafe_from_address=Ap_)
    var Ai = I64P(unsafe_from_address=Ai_)
    var Ax = F64P(unsafe_from_address=Ax_)
    var b = F64P(unsafe_from_address=b_)
    var x = F64P(unsafe_from_address=x_)
    for i in range(n):
        x.unsafe_store(i, b.unsafe_load(i))
    var k = 0
    while k < n:
        var xk = x.unsafe_load(k)
        for p in range(Ap.unsafe_load(k), Ap.unsafe_load(k + 1)):
            var i = Ai.unsafe_load(p)
            x.unsafe_store(i, x.unsafe_load(i) - Ax.unsafe_load(p) * xk)
        k += 1


@export("msp_csc_lower_transpose_solve")
def msp_csc_lower_transpose_solve(n: Int, Ap_: Int, Ai_: Int, Ax_: Int, b_: Int, x_: Int) abi("C"):
    # Solves L^T x = b for unit-diagonal lower triangular L stored
    # column-compressed.
    #
    # (L^T x)(k) = sum_i L(i,k) x(i), and column k of a lower triangular L
    # holds exactly those L(i,k) for i > k.  Walking the columns downwards
    # therefore reads a gather whose inputs are all final, which is why this
    # direction cannot use the forward solve's scatter.
    var Ap = I64P(unsafe_from_address=Ap_)
    var Ai = I64P(unsafe_from_address=Ai_)
    var Ax = F64P(unsafe_from_address=Ax_)
    var b = F64P(unsafe_from_address=b_)
    var x = F64P(unsafe_from_address=x_)
    var k = n - 1
    while k >= 0:
        var s = b.unsafe_load(k)
        for p in range(Ap.unsafe_load(k), Ap.unsafe_load(k + 1)):
            s -= Ax.unsafe_load(p) * x.unsafe_load(Ai.unsafe_load(p))
        x.unsafe_store(k, s)
        k -= 1


@export("msp_vec_scale_inv")
def msp_vec_scale_inv(n: Int, d_: Int, b_: Int, y_: Int) abi("C"):
    # y = b / d, elementwise.
    var d = F64P(unsafe_from_address=d_)
    var b = F64P(unsafe_from_address=b_)
    var y = F64P(unsafe_from_address=y_)
    for i in range(n):
        y.unsafe_store(i, b.unsafe_load(i) / d.unsafe_load(i))


@export("msp_ldl_analyze")
def msp_ldl_analyze(n: Int, Ap_: Int, Ai_: Int, parent_: Int, Lnz_: Int, mark_: Int) abi("C"):
    # Symbolic pass for A = L*D*L^T with L unit lower triangular in CSR.
    #
    # Lnz[i] is the number of strictly-lower entries in row i of L.  Rows are
    # walked in increasing order and the elimination tree (parent[]), built
    # lazily, gives the fill pattern: a walk up from column j stops at the first
    # node already stamped for row i.  parent[v] is the smallest k > v with
    # L(k,v) != 0, so stamping it the first time v is reached builds the tree.
    var Ap = I64P(unsafe_from_address=Ap_)
    var Ai = I64P(unsafe_from_address=Ai_)
    var parent = I64P(unsafe_from_address=parent_)
    var Lnz = I64P(unsafe_from_address=Lnz_)
    var mark = I64P(unsafe_from_address=mark_)
    for i in range(n):
        parent.unsafe_store(i, -1)
        mark.unsafe_store(i, -1)
    for i in range(n):
        mark.unsafe_store(i, i)
        var lnz = 0
        for p in range(Ap.unsafe_load(i), Ap.unsafe_load(i + 1)):
            var j = Ai.unsafe_load(p)
            if j >= i:
                continue
            var v = j
            while mark.unsafe_load(v) != i:
                mark.unsafe_store(v, i)
                lnz += 1
                if parent.unsafe_load(v) == -1:
                    parent.unsafe_store(v, i)
                v = parent.unsafe_load(v)
        Lnz.unsafe_store(i, lnz)


@always_inline
def _sort_int(v: I64P, tmp: I64P, count: Int):
    # Bottom-up merge sort of v[0:count]; std.algorithm has no sort on this
    # toolchain, and the per-row factor pattern needs a guaranteed O(m log m).
    var width = 1
    while width < count:
        var lo = 0
        while lo < count:
            var mid = min(lo + width, count)
            var hi = min(lo + 2 * width, count)
            var a = lo
            var b = mid
            var k = lo
            while a < mid and b < hi:
                if v.unsafe_load(a) <= v.unsafe_load(b):
                    tmp.unsafe_store(k, v.unsafe_load(a))
                    a += 1
                else:
                    tmp.unsafe_store(k, v.unsafe_load(b))
                    b += 1
                k += 1
            while a < mid:
                tmp.unsafe_store(k, v.unsafe_load(a))
                a += 1
                k += 1
            while b < hi:
                tmp.unsafe_store(k, v.unsafe_load(b))
                b += 1
                k += 1
            lo += 2 * width
        var t = 0
        while t < count:
            v.unsafe_store(t, tmp.unsafe_load(t))
            t += 1
        width *= 2


@export("msp_ldl_factor")
def msp_ldl_factor(
    n: Int,
    Ap_: Int,
    Ai_: Int,
    Ax_: Int,
    Lp_: Int,
    Li_: Int,
    Lx_: Int,
    D_: Int,
    mark_: Int,
    val_: Int,
    parent_: Int,
    pat_: Int,
    ptmp_: Int,
    nnz_: Int,
) abi("C"):
    # Numeric up-looking LDL^T for symmetric A stored row-compressed.  Only the
    # lower triangle and the diagonal of A are read; the upper triangle is
    # ignored, matching the CHOLMOD convention.
    #
    # Row i of L is built in two steps.  First the elimination tree gives the
    # pattern: walking up from each column j < i of A stamps every node on the
    # way, which is exactly the set of columns where L(i,j) can be nonzero --
    # fill included, not just what A stores.  Those nodes are then sorted,
    # because L(i,j) = (A(i,j) - sum_{k<j} L(i,k) D(k) L(j,k)) / D(j) needs
    # L(i,k) for every k < j, and mark[k] == i together with val[k] turns the
    # sum over row j of L into a gather over contiguous memory.
    #
    # The pattern matches the one msp_ldl_analyze counted, so the caller's
    # Lp/Li/Lx sizing is exact; the cursor actually consumed is written back to
    # nnz_ so the caller can check that invariant.
    var Ap = I64P(unsafe_from_address=Ap_)
    var Ai = I64P(unsafe_from_address=Ai_)
    var Ax = F64P(unsafe_from_address=Ax_)
    var Lp = I64P(unsafe_from_address=Lp_)
    var Li = I64P(unsafe_from_address=Li_)
    var Lx = F64P(unsafe_from_address=Lx_)
    var D = F64P(unsafe_from_address=D_)
    var mark = I64P(unsafe_from_address=mark_)
    var val = F64P(unsafe_from_address=val_)
    var parent = I64P(unsafe_from_address=parent_)
    var pat = I64P(unsafe_from_address=pat_)
    var ptmp = I64P(unsafe_from_address=ptmp_)
    for i in range(n):
        mark.unsafe_store(i, -1)
        parent.unsafe_store(i, -1)
    var cursor = 0
    Lp.unsafe_store(0, 0)
    for i in range(n):
        Lp.unsafe_store(i, cursor)
        mark.unsafe_store(i, i)
        val.unsafe_store(i, 1.0)
        var dax = 0.0
        var pl = 0
        for p in range(Ap.unsafe_load(i), Ap.unsafe_load(i + 1)):
            var j = Ai.unsafe_load(p)
            if j > i:
                continue
            if j == i:
                dax = Ax.unsafe_load(p)
                continue
            var v = j
            while mark.unsafe_load(v) != i:
                mark.unsafe_store(v, i)
                pat.unsafe_store(pl, v)
                pl += 1
                if parent.unsafe_load(v) == -1:
                    parent.unsafe_store(v, i)
                v = parent.unsafe_load(v)
        _sort_int(pat, ptmp, pl)
        var pa = Ap.unsafe_load(i)
        for q in range(pl):
            var j = pat.unsafe_load(q)
            while pa < Ap.unsafe_load(i + 1) and Ai.unsafe_load(pa) < j:
                pa += 1
            var v = 0.0
            if pa < Ap.unsafe_load(i + 1) and Ai.unsafe_load(pa) == j:
                v = Ax.unsafe_load(pa)
            for p2 in range(Lp.unsafe_load(j), Lp.unsafe_load(j + 1)):
                var k = Li.unsafe_load(p2)
                if mark.unsafe_load(k) == i:
                    v -= val.unsafe_load(k) * Lx.unsafe_load(p2) * D.unsafe_load(k)
            var lij = v / D.unsafe_load(j)
            val.unsafe_store(j, lij)
            Li.unsafe_store(cursor, j)
            Lx.unsafe_store(cursor, lij)
            cursor += 1
        var d = dax
        for p2 in range(Lp.unsafe_load(i), cursor):
            var l = Lx.unsafe_load(p2)
            d -= l * l * D.unsafe_load(Li.unsafe_load(p2))
        D.unsafe_store(i, d)
        Lp.unsafe_store(i + 1, cursor)
    (I64P(unsafe_from_address=nnz_)).unsafe_store(0, cursor)