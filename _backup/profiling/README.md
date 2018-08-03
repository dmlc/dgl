# Profiling the performance of graph-tool, igraph and Networkx

## Settings
We profiled graph-tool 2.26, igraph 0.7.1 and Networkx 2.1 (Python 3.6) on an Intel(R) Xeon(R) CPU E5-2620 v3 @ 2.40GHz with 6 cores.

Benchmark scripts and sources are obtained from:
https://graph-tool.skewed.de/performance

The graph used in profiling is the strongly connected component of the PGP web of trust network circa November 2009. It is a directed graph, with 39,796 vertices and 301,498 edges.

We did not use the betweenness benchmark due to the time it takes (10+ hours for NetworkX).

For the PageRank benchmark of DGL (PageRank with SpMV), we assumed tensorized storage of adjacency matrix and degree vector.
DGL was profiled with both CPU-only and GPU-enabled tensorflow (version 1.8.0).
DGL (CPU) was profiled on the same CPU as graph-tool, igraph and NetworkX, and DGL (GPU) was profiled on a GeForce GTX TITAN X.

## Profiling
```
bash pgp.sh
bash profile.sh
```

## Results
Average time per call:

| Algorithm                   | graph-tool | igraph  | NetworkX | DGL (CPU) | DGL (GPU) |
| --------------------------- | ---------- | ------- | -------- | --------- | --------- |
| Single-source shortest path | 0.004 s    | 0.017 s | 0.484 s  | N/A       | N/A       |
| PageRank                    | 0.009 s    | 0.239 s | 4.207 s  | 0.040 s   | 0.081 s   |
| K-core                      | 0.010 s    | 0.029 s | 1.178 s  | N/A       | N/A       |
| Minimum spanning tree       | 0.022 s    | 0.030 s | 1.961 s  | N/A       | N/A       |

Profiling pagerank_scipy:

| Operation                 | Time    |
| ------------------------- | ------- |
| nx.to_scipy_sparse_matrix | 1.34 s  |
| np.ndarray to dict        | 0.01 s  |
| Power iteration           | 0.001 s |
| PageRank                  | 1.35 s  |
