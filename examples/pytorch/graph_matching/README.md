# Graph Matching Routines

Implementation of various algorithms to compute the Graph Edit Distance (GED) between two DGLGraphs G1 and G2. The graph edit distance between two graphs is a generalization of the string edit distance between strings. The following four algorithms are implemented:

 - astar: Calculates exact GED using A* graph traversal algorithm, the heuristic used is the one proposed in (Riesen and Bunke, 2009) [1].
 - beam: Calculates approximate GED using A* graph traversal algorithm, with a threshold on the size of the open list. [2]
 - bipartite: Calculates approximate GED using linear assignment on the nodes, with Jonker-Volgerand (JV) algorithm. [3]
 - hausdorff: Approximation of graph edit distance based on Hausdorff matching [4].

### Dependencies
  - lapjv (https://github.com/src-d/lapjv): We use the lapjv implementation to solve assignment problem, because of its scalability. Another option is to use the hungarian algorithm provided by scipy (scipy.optimize.linear_sum_assignment).

### Usage

Examples of usage are provided in examples.py. The function signature and an example is also given below:

```sh
graph_edit_distance(G1, G2, node_substitution_cost=None, edge_substitution_cost=None, G1_node_deletion_cost=None, G2_node_insertion_cost=None, G1_edge_deletion_cost=None, G2_edge_insertion_cost=None, algorithm='bipartite', max_beam_size=100)
"""
Parameters
----------
G1, G2: DGLGraphs

node_substitution_cost, edge_substitution_cost : 2D numpy arrays
node_substitution_cost[i,j] is the cost of substitution node i of G1 with node j of G2, similar definition for edge_substitution_cost. If None, default cost of 0 is used.

G1_node_deletion_cost, G1_edge_deletion_cost : 1D numpy arrays
G1_node_deletion_cost[i] is the cost of deletion of node i of G1, similar definition for G1_edge_deletion_cost. If None, default cost of 1 is used.
    
G2_node_insertion_cost, G2_edge_insertion_cost : 1D numpy arrays
G2_node_insertion_cost[i] is the cost of insertion of node i of G2, similar definition for G2_edge_insertion_cost. If None, default cost of 1 is used.

algorithm : string
Algorithm to use to calculate the edit distance. Can be either 'astar', 'beam', 'bipartite' or 'hausdorff'.

max_beam_size : int
Maximum number of nodes in the open list, in case the algorithm is 'beam'.
    
Returns
-------
A tuple of three objects: (edit_distance, node_mapping, edge_mapping)
edit distance is the calculated edit distance (float).
node_mapping is a tuple of size two, containing the node assignments of the two graphs respectively. eg., node_mapping[0][i] is the node mapping of node i of graph G1 (None means that the node is deleted). Similar definition for the edge_mapping.
For 'hausdorff', node_mapping and edge_mapping are returned as None, as this approximation does not return a unique edit path.

Examples
--------
>>> src1 = [0, 1, 2, 3, 4, 5];
>>> dst1 = [1, 2, 3, 4, 5, 6];
>>> src2 = [0, 1, 3, 4, 5];
>>> dst2 = [1, 2, 4, 5, 6];

>>> G1 = dgl.DGLGraph((src1, dst1))
>>> G2 = dgl.DGLGraph((src2, dst2))
>>> distance, node_mapping, edge_mapping = graph_edit_distance(G1, G1, algorithm='astar')
>>> print(distance)
0.0
>>> distance, node_mapping, edge_mapping = graph_edit_distance(G1, G2, algorithm='astar')
>>> print(distance)
1.0
```
### References
    [1] Riesen, Kaspar, Stefan Fankhauser, and Horst Bunke. "Speeding Up Graph Edit Distance Computation with a Bipartite Heuristic." MLG. 2007.
    [2] Neuhaus, Michel, Kaspar Riesen, and Horst Bunke. "Fast suboptimal algorithms for the computation of graph edit distance." Joint IAPR International Workshops on Statistical Techniques in Pattern Recognition (SPR) and Structural and Syntactic Pattern Recognition (SSPR). 2006.
    [3] Fankhauser, Stefan, Kaspar Riesen, and Horst Bunke. "Speeding up graph edit distance computation through fast bipartite matching." International Workshop on Graph-Based Representations in Pattern Recognition. 2011.
    [4] Fischer, Andreas, et al. "A hausdorff heuristic for efficient computation of graph edit distance." Joint IAPR International Workshops on Statistical Techniques in Pattern Recognition (SPR) and Structural and Syntactic Pattern Recognition (SSPR). 2014.



