# author: xbresson
# code link: https://github.com/xbresson/CE7454_2019/blob/master/codes/labs_lecture14/lab01_ChebGCNs/lib/coarsening.py

import numpy as np
import scipy.sparse
import sklearn.metrics


def laplacian(W, normalized=True):
    """Return graph Laplacian"""

    # Degree matrix.
    d = W.sum(axis=0)

    # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

    assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr.csr_matrix
    return L


def rescale_L(L, lmax=2):
    """Rescale Laplacian eigenvalues to [-1,1]"""
    M, M = L.shape
    I = scipy.sparse.identity(M, format="csr", dtype=L.dtype)
    L /= lmax * 2
    L -= I
    return L


def lmax_L(L):
    """Compute largest Laplacian eigenvalue"""
    return scipy.sparse.linalg.eigsh(
        L, k=1, which="LM", return_eigenvectors=False
    )[0]


# graph coarsening with Heavy Edge Matching
def coarsen(A, levels):
    graphs, parents = HEM(A, levels)
    perms = compute_perm(parents)

    laplacians = []
    for i, A in enumerate(graphs):
        M, M = A.shape

        if i < levels:
            A = perm_adjacency(A, perms[i])

        A = A.tocsr()
        A.eliminate_zeros()
        Mnew, Mnew = A.shape
        print(
            "Layer {0}: M_{0} = |V| = {1} nodes ({2} added), |E| = {3} edges".format(
                i, Mnew, Mnew - M, A.nnz // 2
            )
        )

        L = laplacian(A, normalized=True)
        laplacians.append(L)

    return laplacians, perms[0] if len(perms) > 0 else None


def HEM(W, levels, rid=None):
    """
    Coarsen a graph multiple times using the Heavy Edge Matching (HEM).
    Input
    W: symmetric sparse weight (adjacency) matrix
    levels: the number of coarsened graphs
    Output
    graph[0]: original graph of size N_1
    graph[2]: coarser graph of size N_2 < N_1
    graph[levels]: coarsest graph of Size N_levels < ... < N_2 < N_1
    parents[i] is a vector of size N_i with entries ranging from 1 to N_{i+1}
        which indicate the parents in the coarser graph[i+1]
    nd_sz{i} is a vector of size N_i that contains the size of the supernode in the graph{i}
    Note
    if "graph" is a list of length k, then "parents" will be a list of length k-1
    """

    N, N = W.shape

    if rid is None:
        rid = np.random.permutation(range(N))

    ss = np.array(W.sum(axis=0)).squeeze()
    rid = np.argsort(ss)

    parents = []
    degree = W.sum(axis=0) - W.diagonal()
    graphs = []
    graphs.append(W)

    print("Heavy Edge Matching coarsening with Xavier version")

    for _ in range(levels):
        # CHOOSE THE WEIGHTS FOR THE PAIRING
        # weights = ones(N,1)       # metis weights
        weights = degree  # graclus weights
        # weights = supernode_size  # other possibility
        weights = np.array(weights).squeeze()

        # PAIR THE VERTICES AND CONSTRUCT THE ROOT VECTOR
        idx_row, idx_col, val = scipy.sparse.find(W)
        cc = idx_row
        rr = idx_col
        vv = val

        # TO BE SPEEDUP
        if not (list(cc) == list(np.sort(cc))):
            tmp = cc
            cc = rr
            rr = tmp

        cluster_id = HEM_one_level(cc, rr, vv, rid, weights)  # cc is ordered
        parents.append(cluster_id)

        # COMPUTE THE EDGES WEIGHTS FOR THE NEW GRAPH
        nrr = cluster_id[rr]
        ncc = cluster_id[cc]
        nvv = vv
        Nnew = cluster_id.max() + 1
        # CSR is more appropriate: row,val pairs appear multiple times
        W = scipy.sparse.csr_matrix((nvv, (nrr, ncc)), shape=(Nnew, Nnew))
        W.eliminate_zeros()

        # Add new graph to the list of all coarsened graphs
        graphs.append(W)
        N, N = W.shape

        # COMPUTE THE DEGREE (OMIT OR NOT SELF LOOPS)
        degree = W.sum(axis=0)
        # degree = W.sum(axis=0) - W.diagonal()

        # CHOOSE THE ORDER IN WHICH VERTICES WILL BE VISTED AT THE NEXT PASS
        # [~, rid]=sort(ss);     # arthur strategy
        # [~, rid]=sort(supernode_size);    #  thomas strategy
        # rid=randperm(N);                  #  metis/graclus strategy
        ss = np.array(W.sum(axis=0)).squeeze()
        rid = np.argsort(ss)

    return graphs, parents


# Coarsen a graph given by rr,cc,vv.  rr is assumed to be ordered
def HEM_one_level(rr, cc, vv, rid, weights):
    nnz = rr.shape[0]
    N = rr[nnz - 1] + 1

    marked = np.zeros(N, np.bool_)
    rowstart = np.zeros(N, np.int32)
    rowlength = np.zeros(N, np.int32)
    cluster_id = np.zeros(N, np.int32)

    oldval = rr[0]
    count = 0
    clustercount = 0

    for ii in range(nnz):
        rowlength[count] = rowlength[count] + 1
        if rr[ii] > oldval:
            oldval = rr[ii]
            rowstart[count + 1] = ii
            count = count + 1

    for ii in range(N):
        tid = rid[ii]
        if not marked[tid]:
            wmax = 0.0
            rs = rowstart[tid]
            marked[tid] = True
            bestneighbor = -1
            for jj in range(rowlength[tid]):
                nid = cc[rs + jj]
                if marked[nid]:
                    tval = 0.0
                else:
                    # First approach
                    if 2 == 1:
                        tval = vv[rs + jj] * (
                            1.0 / weights[tid] + 1.0 / weights[nid]
                        )

                    # Second approach
                    if 1 == 1:
                        Wij = vv[rs + jj]
                        Wii = vv[rowstart[tid]]
                        Wjj = vv[rowstart[nid]]
                        di = weights[tid]
                        dj = weights[nid]
                        tval = (2.0 * Wij + Wii + Wjj) * 1.0 / (di + dj + 1e-9)

                if tval > wmax:
                    wmax = tval
                    bestneighbor = nid

            cluster_id[tid] = clustercount

            if bestneighbor > -1:
                cluster_id[bestneighbor] = clustercount
                marked[bestneighbor] = True

            clustercount += 1

    return cluster_id


def compute_perm(parents):
    """
    Return a list of indices to reorder the adjacency and data matrices so
    that the union of two neighbors from layer to layer forms a binary tree.
    """

    # Order of last layer is random (chosen by the clustering algorithm).
    indices = []
    if len(parents) > 0:
        M_last = max(parents[-1]) + 1
        indices.append(list(range(M_last)))

    for parent in parents[::-1]:
        # Fake nodes go after real ones.
        pool_singeltons = len(parent)

        indices_layer = []
        for i in indices[-1]:
            indices_node = list(np.where(parent == i)[0])
            assert 0 <= len(indices_node) <= 2

            # Add a node to go with a singelton.
            if len(indices_node) == 1:
                indices_node.append(pool_singeltons)
                pool_singeltons += 1

            # Add two nodes as children of a singelton in the parent.
            elif len(indices_node) == 0:
                indices_node.append(pool_singeltons + 0)
                indices_node.append(pool_singeltons + 1)
                pool_singeltons += 2

            indices_layer.extend(indices_node)
        indices.append(indices_layer)

    # Sanity checks.
    for i, indices_layer in enumerate(indices):
        M = M_last * 2**i
        # Reduction by 2 at each layer (binary tree).
        assert len(indices[0] == M)
        # The new ordering does not omit an indice.
        assert sorted(indices_layer) == list(range(M))

    return indices[::-1]


assert compute_perm(
    [np.array([4, 1, 1, 2, 2, 3, 0, 0, 3]), np.array([2, 1, 0, 1, 0])]
) == [[3, 4, 0, 9, 1, 2, 5, 8, 6, 7, 10, 11], [2, 4, 1, 3, 0, 5], [0, 1, 2]]


def perm_adjacency(A, indices):
    """
    Permute adjacency matrix, i.e. exchange node ids,
    so that binary unions form the clustering tree.
    """
    if indices is None:
        return A

    M, M = A.shape
    Mnew = len(indices)
    A = A.tocoo()

    # Add Mnew - M isolated vertices.
    rows = scipy.sparse.coo_matrix((Mnew - M, M), dtype=np.float32)
    cols = scipy.sparse.coo_matrix((Mnew, Mnew - M), dtype=np.float32)
    A = scipy.sparse.vstack([A, rows])
    A = scipy.sparse.hstack([A, cols])

    # Permute the rows and the columns.
    perm = np.argsort(indices)
    A.row = np.array(perm)[A.row]
    A.col = np.array(perm)[A.col]

    assert np.abs(A - A.T).mean() < 1e-8  # 1e-9
    assert type(A) is scipy.sparse.coo.coo_matrix
    return A


def perm_data(x, indices):
    """
    Permute data matrix, i.e. exchange node ids,
    so that binary unions form the clustering tree.
    """
    if indices is None:
        return x

    N, M = x.shape
    Mnew = len(indices)
    assert Mnew >= M
    xnew = np.empty((N, Mnew))
    for i, j in enumerate(indices):
        # Existing vertex, i.e. real data.
        if j < M:
            xnew[:, i] = x[:, j]
        # Fake vertex because of singeltons.
        # They will stay 0 so that max pooling chooses the singelton.
        # Or -infty ?
        else:
            xnew[:, i] = np.zeros(N)
    return xnew
