# Sparse Cholesky kernels for mojo-scikit-sparse, ported from SuiteSparse CHOLMOD.
#
# Upstream is scikit-sparse (sksparse/cholmod.pyx), which is a thin Cython shell
# around CHOLMOD.  Everything below is a port of the CHOLMOD routine named in the
# comment above it, in CHOLMOD's own source order, so the two can be read side by
# side.  The exception is cholmod_analyze: upstream never spells out its ordering
# choice, and only order="natural" is ported here (see README, "Coverage").
#
# Buffers cross the C ABI as Int addresses and are rebuilt as pointers inside the
# wrapper.  Callers own every allocation; nothing here returns heap memory.
# Int arrays are int64 (CHOLMOD's cholmod_long_ interface, what scipy uses for
# matrices larger than 2**31) and value arrays are Float64.

comptime F64P = Pointer[Float64, AnyOrigin[mut=True]]
comptime I64P = Pointer[Int, AnyOrigin[mut=True]]
# The machine's SIMD width for float64; MOJO_NOTES section 1.
comptime FW = simd_width_of[DType.float64]()
# The matching width for int64, used where an index scan is the hot loop.
comptime IW = simd_width_of[DType.int64]()

from std.math import sqrt
from std.sys import simd_width_of

# CHOLMOD's EMPTY sentinel for an unset parent pointer.
comptime EMPTY: Int = -1

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#     cholmod_transpose
# -----------------------------------------------------------------------------
@export("msp_transpose")
def msp_transpose(
    m: Int, n: Int,
    Ap_: Int, Ai_: Int, Ax_: Int,
    Bp_: Int, Bi_: Int, Bx_: Int,
    values_: Int, packed_: Int, w_: Int,
) abi("C"):
    # B = A' (pattern only when values_ == 0), B column-compressed.
    #
    # Upstream reaches cholmod_transpose (CHOLMOD_TRANS_PATTERN or
    # CHOLMOD_TRANS_CONJ) from four places: the unsymmetric analyze of A A', the
    # unsymmetric analyze of A' A, symbfact(kind="lo"), and etree(kind="lo").
    # All four only ever transpose a matrix whose columns are visited in
    # increasing order, and all four then re-derive what they need from the
    # transposed pattern, so they share this one kernel rather than four copies.
    var Ap = I64P(unsafe_from_address=Ap_)
    var Ai = I64P(unsafe_from_address=Ai_)
    var Bp = I64P(unsafe_from_address=Bp_)
    var Bi = I64P(unsafe_from_address=Bi_)
    var Bx = F64P(unsafe_from_address=Bx_)
    var w = I64P(unsafe_from_address=w_)

    Bp.unsafe_store(0, 0)
    # count the entries in each row of A; that is each column of B
    var anz = Ap.unsafe_load(n)
    var p = 0
    while p < anz:
        var i = Ai.unsafe_load(p)
        Bp.unsafe_store(i + 1, Bp.unsafe_load(i + 1) + 1)
        p += 1
    # prefix sum into Bp.  This is a dependent scan, so the loop-carried value is
    # kept in a register (the serial chain is the loop-carried carried dependency
    # anyway) instead of the read-modify-write of memory it came from.  Vectorizing
    # it needs cross-lane carries, so there is nothing to widen here.
    var run = Bp.unsafe_load(0)
    var i = 0
    while i < m:
        run += Bp.unsafe_load(i + 1)
        Bp.unsafe_store(i + 1, run)
        i += 1
    for i in range(m):
        w.unsafe_store(i, Bp.unsafe_load(i))
    # fill, walking A column by column so each column of B ends up sorted
    var Ax = F64P(unsafe_from_address=Ax_)
    for j in range(n):
        for p in range(Ap.unsafe_load(j), Ap.unsafe_load(j + 1)):
            var i = Ai.unsafe_load(p)
            var q = w.unsafe_load(i)
            Bi.unsafe_store(q, j)
            if values_:
                Bx.unsafe_store(q, Ax.unsafe_load(p))
            w.unsafe_store(i, q + 1)


# -----------------------------------------------------------------------------
#     cholmod_etree / update_etree
# -----------------------------------------------------------------------------
@always_inline
def _update_etree(k_in: Int, i: Int, Parent: I64P, Ancestor: I64P):
    # process the edge (k,i) in the input graph; traverse the path from k to the
    # root of the partially-constructed tree, compressing as it goes
    var k = k_in
    while True:
        var a = Ancestor.unsafe_load(k)
        if a == i:
            # final ancestor reached; no change to tree
            return
        Ancestor.unsafe_store(k, i)
        if a == EMPTY:
            # final ancestor undefined; this is a new edge in the tree
            Parent.unsafe_store(k, i)
            return
        k = a


@export("msp_etree")
def msp_etree(n: Int, nrow_: Int, Ap_: Int, Ai_: Int, Parent_: Int, Ancestor_: Int, Prev_: Int, stype_: Int) abi("C"):
    # Elimination tree of A (symmetric, stype_ > 0) or of A'A (stype_ == 0).
    #
    # Upstream's etree() sets stype = 1 for kind "sym"/"col" and stype = 0 with
    # the transpose for kind "row"; for kind "lo" it transposes and takes the
    # symmetric branch.  The symmetric branch is fed triu(A); the caller
    # therefore passes the transpose when the requested triangle is the lower
    # one.  Caller supplies Parent, Ancestor, and Prev (size nrow, for the
    # unsymmetric path-graph construction) as scratch.
    var Ap = I64P(unsafe_from_address=Ap_)
    var Ai = I64P(unsafe_from_address=Ai_)
    var Parent = I64P(unsafe_from_address=Parent_)
    var Ancestor = I64P(unsafe_from_address=Ancestor_)
    var nrow = nrow_
    var stype = stype_

    for j in range(n):
        Parent.unsafe_store(j, EMPTY)
        Ancestor.unsafe_store(j, EMPTY)

    if stype > 0:
        # symmetric (upper) case: compute etree (A)
        for j in range(n):
            # for each row i in column j of triu(A), excluding the diagonal
            for p in range(Ap.unsafe_load(j), Ap.unsafe_load(j + 1)):
                var i = Ai.unsafe_load(p)
                if i < j:
                    _update_etree(i, j, Parent, Ancestor)
    else:
        # unsymmetric case: compute etree (A'A)
        var Prev = I64P(unsafe_from_address=Prev_)
        for i in range(nrow):
            Prev.unsafe_store(i, EMPTY)
        for j in range(n):
            # for each row i in column j of A
            for p in range(Ap.unsafe_load(j), Ap.unsafe_load(j + 1)):
                # a graph is constructed dynamically with one path per row of A.
                # At node i of this path-graph, all edges (jprev,j) are
                # considered, where jprev < j.
                var i = Ai.unsafe_load(p)
                var jprev = Prev.unsafe_load(i)
                if jprev != EMPTY:
                    _update_etree(jprev, j, Parent, Ancestor)
                Prev.unsafe_store(i, j)


# -----------------------------------------------------------------------------
#     cholmod_postorder / dfs
# -----------------------------------------------------------------------------
@always_inline
def _dfs(p_in: Int, k_in: Int, Post: I64P, Head: I64P, Next: I64P, Pstack: I64P) -> Int:
    # return the new value of k; non-recursive depth-first search
    var p = p_in
    var k = k_in
    # put the root node on the stack
    Pstack.unsafe_store(0, p)
    var phead = 0
    while phead >= 0:
        # grab the node p from top of the stack and get its youngest child j
        p = Pstack.unsafe_load(phead)
        var j = Head.unsafe_load(p)
        if j == EMPTY:
            # all children of p ordered; remove p from stack and order it
            phead -= 1
            Post.unsafe_store(k, p)
            k += 1
        else:
            # leave p on the stack; start a DFS at child j
            Head.unsafe_store(p, Next.unsafe_load(j))
            phead += 1
            Pstack.unsafe_store(phead, j)
    return k


@export("msp_postorder")
def msp_postorder(n: Int, Parent_: Int, Post_: Int, Head_: Int, Next_: Int, Pstack_: Int) abi("C"):
    # Postorder a tree whose Parent[j] = p if p is the parent of j, and
    # Parent[j] = EMPTY (-1) if j is a root.  On output Post[0..n-1] is a
    # complete permutation vector.  Caller supplies Head, Next, Pstack scratch.
    var Parent = I64P(unsafe_from_address=Parent_)
    var Post = I64P(unsafe_from_address=Post_)
    var Head = I64P(unsafe_from_address=Head_)
    var Next = I64P(unsafe_from_address=Next_)
    var Pstack = I64P(unsafe_from_address=Pstack_)

    for i in range(n + 1):
        Head.unsafe_store(i, EMPTY)
    # construct a link list of children for each node, in reverse order so
    # children come out in ascending order in each list
    for j in range(n - 1, -1, -1):
        var p = Parent.unsafe_load(j)
        if p >= 0 and p < n:
            Next.unsafe_store(j, Head.unsafe_load(p))
            Head.unsafe_store(p, j)

    # start a DFS at each root node of the etree
    var k = 0
    for j in range(n):
        if Parent.unsafe_load(j) == EMPTY:
            k = _dfs(j, k, Post, Head, Next, Pstack)

    for j in range(n):
        Head.unsafe_store(j, EMPTY)


# -----------------------------------------------------------------------------
#     cholmod_rowcolcounts / initialize_node / process_edge / finalize_node
# -----------------------------------------------------------------------------
@always_inline
def _initialize_node(k: Int, Post: I64P, Parent: I64P, ColCount: I64P, PrevNbr: I64P) -> Int:
    # initial work for the kth node in the postordered etree
    var p = Post.unsafe_load(k)
    var parent = Parent.unsafe_load(p)
    if parent != EMPTY:
        ColCount.unsafe_store(parent, ColCount.unsafe_load(parent) - 1)
    # flag node p to exclude self edges (p,p)
    PrevNbr.unsafe_store(p, k)
    return p


@always_inline
def _process_edge(
    p: Int, u: Int, k: Int,
    First: I64P, PrevNbr: I64P, ColCount: I64P, PrevLeaf: I64P,
    RowCount: I64P, SetParent: I64P, Level: I64P,
):
    # edge (p,u) is being processed; p < u is a descendant of its ancestor u in
    # the etree, and node p is the kth node in the postordered etree
    if First.unsafe_load(p) > PrevNbr.unsafe_load(u):
        # p is a leaf of the subtree of u
        ColCount.unsafe_store(p, ColCount.unsafe_load(p) + 1)
        var prevleaf = PrevLeaf.unsafe_load(u)
        var q = u
        if prevleaf != EMPTY:
            # q = FIND (prevleaf): root of the SetParent tree containing prevleaf
            q = prevleaf
            while q != SetParent.unsafe_load(q):
                q = SetParent.unsafe_load(q)
            # the root q has been found; re-traverse and perform path compression
            var s = prevleaf
            while s != q:
                var sparent = SetParent.unsafe_load(s)
                SetParent.unsafe_store(s, q)
                s = sparent
            ColCount.unsafe_store(q, ColCount.unsafe_load(q) - 1)
        # increment RowCount by the length of the path from p to q
        RowCount.unsafe_store(u, RowCount.unsafe_load(u) + (Level.unsafe_load(p) - Level.unsafe_load(q)))
        PrevLeaf.unsafe_store(u, p)
    # flag u as having been processed at step k
    PrevNbr.unsafe_store(u, k)


@always_inline
def _finalize_node(p: Int, Parent: I64P, SetParent: I64P):
    # all nodes in the SetParent tree rooted at p now have as their final root
    # the node Parent[p]: this computes UNION (p, Parent[p])
    if Parent.unsafe_load(p) != EMPTY:
        SetParent.unsafe_store(p, Parent.unsafe_load(p))


@export("msp_rowcolcounts")
def msp_rowcolcounts(
    nrow_: Int, ncol_: Int,
    Ap_: Int, Ai_: Int,
    Parent_: Int, Post_: Int,
    ColCount_: Int, RowCount_: Int, First_: Int, Level_: Int,
    PrevNbr_: Int, SetParent_: Int, PrevLeaf_: Int, Anext_: Int, Head_: Int, Ipost_: Int,
    stype_: Int, anz_: Int,
) abi("C"):
    # Compute the row and column counts of the Cholesky factor L of A or A A'.
    # The etree and its postordering must already be computed (see msp_etree and
    # msp_postorder) and are given as inputs.
    #
    # For the symmetric case (LL' = A), A is accessed by column and only the
    # lower triangular part of A is used.  For the unsymmetric case
    # (LL' = AA'), A is accessed by column and all of A is used; the etree must
    # be the etree of A A' (etree == 0 in upstream's naming).
    #
    # Caller supplies the scratch arrays PrevNbr, SetParent, PrevLeaf, Anext,
    # Head, Ipost, and the RowCount output; anz_ receives the number of entries
    # in triu(A) (the symmetric case only), which CHOLMOD returns in Common.anz.
    var nrow = nrow_
    var ncol = ncol_
    var Ap = I64P(unsafe_from_address=Ap_)
    var Ai = I64P(unsafe_from_address=Ai_)
    var Parent = I64P(unsafe_from_address=Parent_)
    var Post = I64P(unsafe_from_address=Post_)
    var ColCount = I64P(unsafe_from_address=ColCount_)
    var First = I64P(unsafe_from_address=First_)
    var Level = I64P(unsafe_from_address=Level_)
    var PrevNbr = I64P(unsafe_from_address=PrevNbr_)
    var SetParent = I64P(unsafe_from_address=SetParent_)
    var PrevLeaf = I64P(unsafe_from_address=PrevLeaf_)
    var Head = I64P(unsafe_from_address=Head_)
    var stype = stype_

    # find the first descendant and level of each node in the tree
    for i in range(nrow):
        First.unsafe_store(i, EMPTY)
    for k in range(nrow):
        # node i of the etree is the kth node in the postordered etree
        var i = Post.unsafe_load(k)
        # i is a leaf if First[i] is still EMPTY; ColCount[i] starts at 1 if i is
        # a leaf, zero otherwise
        if First.unsafe_load(i) == EMPTY:
            ColCount.unsafe_store(i, 1)
        else:
            ColCount.unsafe_store(i, 0)
        # traverse the path from node i to the root, stopping at a node r whose
        # First[r] is already defined
        var len = 0
        var r = i
        while r != EMPTY and First.unsafe_load(r) == EMPTY:
            First.unsafe_store(r, k)
            len += 1
            r = Parent.unsafe_load(r)
        if r == EMPTY:
            # we hit a root node, the level of which is zero
            len -= 1
        else:
            # we stopped at node r, where Level[r] is already defined
            len += Level.unsafe_load(r)
        # re-traverse the path from node i to r; set the level of each node
        var s = i
        while s != r:
            Level.unsafe_store(s, len)
            len -= 1
            s = Parent.unsafe_load(s)

    # AA' case: sort columns of A according to first postordered row index
    var unsym = stype == 0
    if unsym:
        var Ipost = I64P(unsafe_from_address=Ipost_)
        var Anext = I64P(unsafe_from_address=Anext_)
        for k in range(nrow):
            Ipost.unsafe_store(Post.unsafe_load(k), k)
        for j in range(ncol):
            Anext.unsafe_store(j, EMPTY)
        for jj in range(ncol):
            var j = jj
            # column j: find the smallest postordered row (if any) and place the
            # column in that bucket's link list
            var p = Ap.unsafe_load(j)
            var pend = Ap.unsafe_load(j + 1)
            if pend > p:
                var k = Ipost.unsafe_load(Ai.unsafe_load(p))
                p += 1
                while p < pend:
                    var inew = Ipost.unsafe_load(Ai.unsafe_load(p))
                    if inew < k:
                        k = inew
                    p += 1
                Anext.unsafe_store(j, Head.unsafe_load(k))
                Head.unsafe_store(k, j)

    # compute the row counts and node weights
    var RowCount = I64P(unsafe_from_address=RowCount_)
    for i in range(nrow):
        RowCount.unsafe_store(i, 1)
    for i in range(nrow):
        PrevLeaf.unsafe_store(i, EMPTY)
        PrevNbr.unsafe_store(i, EMPTY)
        SetParent.unsafe_store(i, i)  # every node is in its own set

    if not unsym:
        # symmetric case: LL' = A
        # also determine the number of entries in triu(A)
        var anz = nrow
        for k in range(nrow):
            # j is the kth node in the postordered etree
            var j = _initialize_node(k, Post, Parent, ColCount, PrevNbr)
            # for all nonzeros A(i,j) below the diagonal, in column j of A
            for p in range(Ap.unsafe_load(j), Ap.unsafe_load(j + 1)):
                var i = Ai.unsafe_load(p)
                if i > j:
                    # j is a descendant of i in etree(A)
                    anz += 1
                    _process_edge(
                        j, i, k, First, PrevNbr, ColCount, PrevLeaf,
                        RowCount, SetParent, Level,
                    )
            _finalize_node(j, Parent, SetParent)
        I64P(unsafe_from_address=anz_).unsafe_store(0, anz)
    else:
        # unsymmetric case: LL' = AA'
        var Anext = I64P(unsafe_from_address=Anext_)
        for k in range(nrow):
            # inode is the kth node in the postordered etree
            var inode = _initialize_node(k, Post, Parent, ColCount, PrevNbr)
            # for all cols j whose first postordered row is k
            var j = Head.unsafe_load(k)
            while j != EMPTY:
                var jnext = Anext.unsafe_load(j)
                # for all rows i in column j
                for p in range(Ap.unsafe_load(j), Ap.unsafe_load(j + 1)):
                    var i = Ai.unsafe_load(p)
                    # has i already been considered at this step k
                    if PrevNbr.unsafe_load(i) < k:
                        # inode is a descendant of i in etree(AA')
                        _process_edge(
                            inode, i, k, First, PrevNbr, ColCount, PrevLeaf,
                            RowCount, SetParent, Level,
                        )
                j = jnext
            # clear link list k
            Head.unsafe_store(k, EMPTY)
            _finalize_node(inode, Parent, SetParent)

    # finish computing the column counts
    for j in range(nrow):
        var parent = Parent.unsafe_load(j)
        if parent != EMPTY:
            ColCount.unsafe_store(parent, ColCount.unsafe_load(parent) + ColCount.unsafe_load(j))

# -----------------------------------------------------------------------------
#     cholmod_l_row_subtree (the SUBTREE macro) and the batched pattern walk
# -----------------------------------------------------------------------------
@always_inline
def _subtree(
    Ap: I64P, Ai: I64P, Parent: I64P, Flag: I64P, Stack: I64P,
    mark: Int, k: Int, p_in: Int, pend: Int, top_in: Int,
) -> Int:
    # The SUBTREE macro from cholmod_rowfac.c, verbatim in structure.  SCATTER
    # is the empty operation (this variant computes the pattern only) and
    # PARENT(i) is Parent[i].
    var p = p_in
    var top = top_in
    while p < pend:
        var i = Ai.unsafe_load(p)
        if i <= k:
            # SCATTER (nothing)
            # start at node i and traverse up the subtree, stop at node k
            var length = 0
            while i < k and i != EMPTY and Flag.unsafe_load(i) < mark:
                # L(k,i) is nonzero, and seen for the first time
                Stack.unsafe_store(length, i)  # place i on the stack
                length += 1
                Flag.unsafe_store(i, mark)  # mark i as visited
                i = Parent.unsafe_load(i)  # traverse up the etree
            # move the path down to the bottom of the stack
            while length > 0:
                top -= 1
                length -= 1
                Stack.unsafe_store(top, Stack.unsafe_load(length))
        p += 1
    return top


@export("msp_row_pattern")
def msp_row_pattern(
    n_: Int,
    Ap_: Int, Ai_: Int,
    Fp_: Int, Fi_: Int,
    Parent_: Int,
    Lp_: Int, Li_: Int,
    W_: Int, Len_: Int,
    Flag_: Int, Stack_: Int,
    stype_: Int,
) abi("C"):
    # The nonzero pattern of rows 0..n-1 of the Cholesky factor L, written
    # straight into a packed column-compressed Li in one call.  This is
    # cholmod_row_subtree unrolled over every k, with the driver's per-element
    # scatter folded in: upstream's _cholesky_l_pattern places row k's column j
    # at Li[W[j]] and advances W[j], and because k is the outer loop each column
    # ends up with its entries in increasing k and its diagonal added last.
    #
    # Row k's walk lives at the top of Stack, which is where the original
    # cholmod_row_subtree put it: the run of off-diagonal columns of row k is
    # pushed onto the end of the buffer and grows downwards.  The buffer has to
    # reach nnz(L), not n, because row k's run can be longer than n: L(k,j) for
    # j<k occupies a slot in column j, so the run's length is at most Lp[k], and
    # Lp[k] runs up to nnz(L).  (CHOLMOD sizes its stack Lnzmax + n for the same
    # reason.)  Zero is the empty-stack sentinel and row indices run from 1, so 0
    # can never be a real row index: that is what lets the kernel clear only the
    # Flag slots it set instead of sweeping all n of them, which is what cost
    # O(n^2) before.
    #
    # Caller supplies the scratch: Flag (n), Stack (nnz(L)), Len (n) and W (n,
    # the per-column write cursor, starting at Lp).  Flag must be all zero on
    # entry.  Len[k] receives nnz(row k) - 1, the off-diagonal length of row k.
    var n = n_
    var Ap = I64P(unsafe_from_address=Ap_)
    var Ai = I64P(unsafe_from_address=Ai_)
    var Parent = I64P(unsafe_from_address=Parent_)
    var Lp = I64P(unsafe_from_address=Lp_)
    var Li = I64P(unsafe_from_address=Li_)
    var W = I64P(unsafe_from_address=W_)
    var Len = I64P(unsafe_from_address=Len_)
    var Flag = I64P(unsafe_from_address=Flag_)
    var Stack = I64P(unsafe_from_address=Stack_)
    var stype = stype_
    var Fp = I64P(unsafe_from_address=Fp_)
    var Fi = I64P(unsafe_from_address=Fi_)

    # a mark of 2*k+2 keeps every row's stamp distinct, and above the widest mark
    # of the pass it is still even so an unmarked Flag[i] (0) never looks visited
    for k in range(n):
        var lim = Lp.unsafe_load(k)
        var mark = 2 * k + 2
        # the run for row k is pushed onto the end of Stack, so its start is the
        # stack base lim; _subtree returns where the run ended up
        var top = lim
        if stype > 0:
            # symmetric upper case: scatter the kth column of triu(A)
            top = _subtree(
                Ap, Ai, Parent, Flag, Stack, mark, k,
                Ap.unsafe_load(k), Ap.unsafe_load(k + 1), lim,
            )
        else:
            # unsymmetric case: for each nonzero F(t,k), scatter column t of A
            for pf in range(Fp.unsafe_load(k), Fp.unsafe_load(k + 1)):
                var t = Fi.unsafe_load(pf)
                top = _subtree(
                    Ap, Ai, Parent, Flag, Stack, mark, k,
                    Ap.unsafe_load(t), Ap.unsafe_load(t + 1), top,
                )
        # the stack holds exactly the off-diagonal columns of row k, in strictly
        # decreasing order: the walk pushes each new path at the bottom and pops
        # it down, and a column already on the stack is never pushed again
        var s = top
        while s < lim:
            var j = Stack.unsafe_load(s)
            var w = W.unsafe_load(j)
            Li.unsafe_store(w, k)
            W.unsafe_store(j, w + 1)
            # reset the marks this row set, walking its own stack: every node
            # it pushed is on the stack, so no other row's mark is touched
            Flag.unsafe_store(j, 0)
            s += 1
        # the diagonal of row k, last in its column
        var wk = W.unsafe_load(k)
        Li.unsafe_store(wk, k)
        W.unsafe_store(k, wk + 1)
        Len.unsafe_store(k, lim - top)


# -----------------------------------------------------------------------------
#     cholmod_l_rowfac (via t_cholmod_rowfac_worker.c)
# -----------------------------------------------------------------------------
@export("msp_rowfac")
def msp_rowfac(
    n_: Int, beta: Float64,
    Ap_: Int, Ai_: Int, Ax_: Int,
    Fp_: Int, Fi_: Int, Fx_: Int,
    Lp_: Int, Li_: Int, Lx_: Int, Lnz_: Int,
    Flag_: Int, Stack_: Int, W_: Int,
    is_ll_: Int,
    minor_out_: Int, fl_out_: Int,
    stype_: Int,
) abi("C"):
    # Compute the LDL' or LL' factorization by rows, exactly as
    # cholmod_l_rowfac does.  L is packed column-compressed with the diagonal
    # entry first in each column (that is what Lp/Li/Lx and Lnz describe).
    #
    # A is the lower triangle of the (permuted) matrix, column-compressed.
    # Only the lower triangle and the diagonal of A are read.  beta is added to
    # the diagonal before factoring, matching cholmod_factorize_p.
    #
    # stype_ > 0 selects the symmetric case, in which F is unused and
    # Fp_/Fi_/Fx_ may be 0.  stype_ == 0 selects the unsymmetric case, in which
    # F = A' (or A(:,f)') is required and the k'th column of A A' is scattered.
    #
    # On output minor_out receives the index at which the factor stopped being
    # positive definite (n if it got all the way), and fl_out the flop count.
    var n = n_
    var Ap = I64P(unsafe_from_address=Ap_)
    var Ai = I64P(unsafe_from_address=Ai_)
    var Ax = F64P(unsafe_from_address=Ax_)
    var Lp = I64P(unsafe_from_address=Lp_)
    var Li = I64P(unsafe_from_address=Li_)
    var Lx = F64P(unsafe_from_address=Lx_)
    var Lnz = I64P(unsafe_from_address=Lnz_)
    var Flag = I64P(unsafe_from_address=Flag_)
    var Stack = I64P(unsafe_from_address=Stack_)
    var W = F64P(unsafe_from_address=W_)
    var is_ll = is_ll_ != 0
    var stype = stype_
    var minor = n
    var fl = 0.0
    # CHOLMOD's mark is persistent and monotonically increasing across the whole
    # factorization; one Flag array serves every k, cleared each time.
    var mark = 1

    for k in range(n):
        # column k of L is currently empty
        Lnz.unsafe_store(k, 1)

        var top = n  # Stack is empty
        Flag.unsafe_store(k, mark)  # do not include the diagonal entry in Stack

        # compute pattern of kth row of L and scatter the kth input column
        if stype > 0:
            # symmetric: scatter kth col of triu (beta*I+AA'), get pattern L(k,:)
            top = _rowfac_subtree(Ap, Ai, Ax, W, Lnz, Li, Lp, Flag, Stack, mark, k, Ap.unsafe_load(k), Ap.unsafe_load(k + 1), top, 0.0, 0)
        else:
            # unsymmetric: scatter kth col of triu (beta*I+AA'), where the i'th
            # column of A is scaled by F(t,k); W = A A' (or A' A transposed in)
            var Fp = I64P(unsafe_from_address=Fp_)
            var Fi = I64P(unsafe_from_address=Fi_)
            var Fx = F64P(unsafe_from_address=Fx_)
            for pf in range(Fp.unsafe_load(k), Fp.unsafe_load(k + 1)):
                # get nonzero entry F (t,k)
                var t = Fi.unsafe_load(pf)
                var fx = Fx.unsafe_load(pf)
                top = _rowfac_subtree(Ap, Ai, Ax, W, Lnz, Li, Lp, Flag, Stack, mark, k, Ap.unsafe_load(t), Ap.unsafe_load(t + 1), top, fx, 1)

        for i in range(n):
            if Flag.unsafe_load(i) == mark:
                Flag.unsafe_store(i, 0)
        mark += 1

        # compute the kth row of L and store it in column form.
        # Solve L(0:k-1, 0:k-1) y(0:k-1) = b(0:k-1) where b(0:k) is in W and
        # Stack.  For LDL': L(k,0:k-1) = y(0:k-1) ./ D(0:k-1) and
        # D(k) = b(k) - L(k,0:k-1) y(0:k-1).  For LL': L(k,0:k-1) = y(0:k-1)
        # and L(k,k) = sqrt(b(k) - L(k,0:k-1) L(0:k-1,k)).
        var dk = W.unsafe_load(k) + beta
        W.unsafe_store(k, 0.0)

        var s = top
        while s < n:
            # get i for each nonzero entry L(k,i)
            var i = Stack.unsafe_load(s)
            # y = W[i]; W[i] = 0
            var y = W.unsafe_load(i)
            W.unsafe_store(i, 0.0)

            var lnz = Lnz.unsafe_load(i)
            var p = Lp.unsafe_load(i)
            # Lnz(i) is the write cursor for column i (the end of what has been
            # written so far), so column i runs from Lp(i) to that cursor
            var pend = p + lnz
            # di is the diagonal entry L(i,i) or D(i,i)
            var di = Lx.unsafe_load(p)
            var lx = 0.0

            if i >= minor or di == 0.0:
                # the matrix is already not positive definite: skip column i
                lx = 0.0
                p = pend
            elif is_ll:
                fl += 2.0 * Float64(pend - p - 1) + 3.0
                # forward solve using L(i:(k-1),i); divide by L(i,i)
                y = y / di
                p += 1
                while p < pend:
                    var t = Li.unsafe_load(p)
                    W.unsafe_store(t, W.unsafe_load(t) - Lx.unsafe_load(p) * y)
                    p += 1
                # do not scale L; compute the dot product for L(k,k)
                lx = y
                dk = dk - lx * y
            else:
                fl += 2.0 * Float64(pend - p - 1) + 3.0
                # forward solve using D(i,i) and L((i+1):(k-1),i)
                p += 1
                while p < pend:
                    var t = Li.unsafe_load(p)
                    W.unsafe_store(t, W.unsafe_load(t) - Lx.unsafe_load(p) * y)
                    p += 1
                # scale L(k,0:k-1) for LDL'; compute D(k,k)
                lx = y / di
                dk = dk - lx * y

            # store L(k,i) in the column form matrix of L
            Li.unsafe_store(p, k)
            Lx.unsafe_store(p, lx)
            Lnz.unsafe_store(i, lnz + 1)
            s += 1

        # ensure the diagonal exists, and store it in L
        var pd = Lp.unsafe_load(k)
        Li.unsafe_store(pd, k)

        if k >= minor:
            # the matrix is already not positive definite
            dk = 0.0
        elif (is_ll and dk <= 0.0) or (not is_ll and dk == 0.0):
            # the matrix has just been found to be not positive definite
            dk = 0.0
            minor = k

        if is_ll:
            # this is counted as one flop, below
            dk = _sqrt(dk)
            fl += 1.0

        # Lx[p] = D(k,k) = d
        Lx.unsafe_store(pd, dk)

    I64P(unsafe_from_address=minor_out_).unsafe_store(0, minor)
    F64P(unsafe_from_address=fl_out_).unsafe_store(0, fl)


@always_inline
def _rowfac_subtree(
    Ap: I64P, Ai: I64P, Ax: F64P, W: F64P,
    Lnz: I64P, Li: I64P, Lp: I64P,
    Flag: I64P, Stack: I64P,
    mark: Int, k: Int, p_in: Int, pend: Int, top_in: Int,
    fx: Float64, multadd: Int,
) -> Int:
    # The scatter-and-SUBTREE loop of t_cholmod_rowfac_worker.c.  SCATTER is
    # ASSIGN (plain copy) in the symmetric case and MULTADD (scaled accumulate)
    # in the unsymmetric case; PARENT(i) is Li[Lp[i]+1] via _rowfac_parent.
    var p = p_in
    var top = top_in
    while p < pend:
        var i = Ai.unsafe_load(p)
        if i <= k:
            if multadd:
                # SCATTER MULTADD: W[i] += Ax[p] * fx
                W.unsafe_store(i, W.unsafe_load(i) + Ax.unsafe_load(p) * fx)
            else:
                # SCATTER ASSIGN: W[i] = Ax[p]
                W.unsafe_store(i, Ax.unsafe_load(p))
            # start at node i and traverse up the subtree, stop at node k
            var length = 0
            while i < k and i != EMPTY and Flag.unsafe_load(i) < mark:
                # L(k,i) is nonzero, and seen for the first time
                Stack.unsafe_store(length, i)
                length += 1
                Flag.unsafe_store(i, mark)
                i = _rowfac_parent(Lnz, Li, Lp, i)
            # move the path down to the bottom of the stack
            while length > 0:
                top -= 1
                length -= 1
                Stack.unsafe_store(top, Stack.unsafe_load(length))
        p += 1
    return top


@always_inline
def _rowfac_parent(Lnz: I64P, Li: I64P, Lp: I64P, i: Int) -> Int:
    # use Li[Lp[i]+1] for the etree: if column i of L has only its diagonal
    # entry, node i has no parent yet
    if Lnz.unsafe_load(i) > 1:
        return Li.unsafe_load(Lp.unsafe_load(i) + 1)
    return EMPTY


@always_inline
def _sqrt(x: Float64) -> Float64:
    # CHOLMOD applies sqrt to the diagonal of an LL' factor.  Wrapped so the
    # import lives in one place.
    return sqrt(x)


# -----------------------------------------------------------------------------
#     cholmod_l_lsolve (via t_cholmod_lsolve_template.c)
# -----------------------------------------------------------------------------
@export("msp_lsolve")
def msp_lsolve(n: Int, Lp_: Int, Li_: Int, Lx_: Int, Lnz_: Int, X_: Int, is_ll_: Int, is_ld_: Int) abi("C"):
    # Solve L x = b in place (X is b on input and x on output).
    #
    # is_ll_: L has a non-unit diagonal and this is an LL' factor, so x[j] is
    #   divided by L(j,j) and the off-diagonal entries of column j are L(j,i).
    # is_ld_: L is the combined LD factor of an LDL' factorization, so column j
    #   stores both L(j,j) (its first entry, holding D(j,j)*1) and the rest.
    # Otherwise L has a unit diagonal (plain LDL' L), so only the off-diagonal
    # entries are used and x[j] is left as the accumulated value.
    #
    # Only the single-column path is ported; CHOLMOD's 2- and 3-column supernode
    # paths are a pure performance optimization over exactly these operations.
    var Lp = I64P(unsafe_from_address=Lp_)
    var Li = I64P(unsafe_from_address=Li_)
    var Lx = F64P(unsafe_from_address=Lx_)
    var Lnz = I64P(unsafe_from_address=Lnz_)
    var X = F64P(unsafe_from_address=X_)
    var is_ll = is_ll_ != 0
    var is_ld = is_ld_ != 0

    var j = 0
    while j < n:
        # get the start, end, and length of column j
        var p = Lp.unsafe_load(j)
        var lnz = Lnz.unsafe_load(j)
        var pend = p + lnz

        # solve with a single column of L
        var y = X.unsafe_load(j)
        if is_ll:
            y = y / Lx.unsafe_load(p)
            X.unsafe_store(j, y)
        elif is_ld:
            X.unsafe_store(j, y / Lx.unsafe_load(p))
        p += 1
        while p < pend:
            X.unsafe_store(Li.unsafe_load(p), X.unsafe_load(Li.unsafe_load(p)) - Lx.unsafe_load(p) * y)
            p += 1
        j += 1  # advance to the next column of L


# -----------------------------------------------------------------------------
#     cholmod_l_ltsolve (via t_cholmod_ltsolve_template.c)
# -----------------------------------------------------------------------------
@export("msp_ltsolve")
def msp_ltsolve(n: Int, Lp_: Int, Li_: Int, Lx_: Int, Lnz_: Int, X_: Int, is_ll_: Int) abi("C"):
    # Solve L' x = b in place (X is b on input and x on output).
    #
    # For an LL' factor (is_ll_ != 0) the diagonal is divided out as well; for
    # a unit-diagonal LDL' factor it is not.  Only the single-column path is
    # ported; CHOLMOD's 2- and 3-column supernode paths are a pure performance
    # optimization over exactly these operations.
    var Lp = I64P(unsafe_from_address=Lp_)
    var Li = I64P(unsafe_from_address=Li_)
    var Lx = F64P(unsafe_from_address=Lx_)
    var Lnz = I64P(unsafe_from_address=Lnz_)
    var X = F64P(unsafe_from_address=X_)
    var is_ll = is_ll_ != 0

    var j = n - 1
    while j >= 0:
        # get the start, end, and length of column j
        var p = Lp.unsafe_load(j)
        var lnz = Lnz.unsafe_load(j)
        var pend = p + lnz

        # solve with a single column of L
        var y = X.unsafe_load(j)
        var d = Lx.unsafe_load(p)
        p += 1
        while p < pend:
            y -= Lx.unsafe_load(p) * X.unsafe_load(Li.unsafe_load(p))
            p += 1
        if is_ll:
            X.unsafe_store(j, y / d)
        else:
            X.unsafe_store(j, y)
        j -= 1  # advance to the next column of L


# -----------------------------------------------------------------------------
#     csc_matvec: y = A*x, for a column-compressed A
# -----------------------------------------------------------------------------
@export("msp_csc_matvec")
def msp_csc_matvec(m: Int, n: Int, Ap_: Int, Ai_: Int, Ax_: Int, x_: Int, y_: Int) abi("C"):
    # y = A*x for A stored column-compressed.  Rows are zeroed once, then each
    # column contributes an axpy scattered into y.  A column whose x entry is
    # exactly zero contributes nothing, so it is skipped outright.
    var Ap = I64P(unsafe_from_address=Ap_)
    var Ai = I64P(unsafe_from_address=Ai_)
    var Ax = F64P(unsafe_from_address=Ax_)
    var x = F64P(unsafe_from_address=x_)
    var y = F64P(unsafe_from_address=y_)
    var i = 0
    while i + FW <= m:
        y.unsafe_store(i, SIMD[DType.float64, FW](0.0))
        i += FW
    while i < m:
        y.unsafe_store(i, 0.0)
        i += 1
    for j in range(n):
        var xj = x.unsafe_load(j)
        if xj == 0.0:
            continue
        for p in range(Ap.unsafe_load(j), Ap.unsafe_load(j + 1)):
            var r = Ai.unsafe_load(p)
            y.unsafe_store(r, y.unsafe_load(r) + Ax.unsafe_load(p) * xj)


# -----------------------------------------------------------------------------
#     the arithmetic the solve driver needs, in the same C ABI
# -----------------------------------------------------------------------------
@export("msp_vec_scale_inv")
def msp_vec_scale_inv(n: Int, d_: Int, b_: Int, y_: Int) abi("C"):
    # y = b / d, elementwise (the D solve, and the LDL' scaling step).  When b
    # and y are the same buffer this is an in-place scale, which is what the
    # solve driver does rather than allocating a temporary per call.
    var d = F64P(unsafe_from_address=d_)
    var b = F64P(unsafe_from_address=b_)
    var y = F64P(unsafe_from_address=y_)
    var i = 0
    while i + FW <= n:
        y.unsafe_store(i, b.unsafe_load[width=FW](i) / d.unsafe_load[width=FW](i))
        i += FW
    while i < n:
        y.unsafe_store(i, b.unsafe_load(i) / d.unsafe_load(i))
        i += 1


@export("msp_vec_permute")
def msp_vec_permute(n: Int, p_: Int, b_: Int, x_: Int) abi("C"):
    # x = P*b where P is the permutation matrix I[p] for the permutation vector
    # p: x[p[k]] = b[k].  CHOLMOD's P is exactly I[p].
    var p = I64P(unsafe_from_address=p_)
    var b = F64P(unsafe_from_address=b_)
    var x = F64P(unsafe_from_address=x_)
    for k in range(n):
        x.unsafe_store(p.unsafe_load(k), b.unsafe_load(k))


@export("msp_vec_unpermute")
def msp_vec_unpermute(n: Int, p_: Int, b_: Int, x_: Int) abi("C"):
    # x = P'*b for the same P: x[k] = b[p[k]]
    var p = I64P(unsafe_from_address=p_)
    var b = F64P(unsafe_from_address=b_)
    var x = F64P(unsafe_from_address=x_)
    for k in range(n):
        x.unsafe_store(k, b.unsafe_load(p.unsafe_load(k)))
