# Migration Guide for DGL 0.5

## Breaking changes

The following changes may break existing codes if the related APIs are used. Note that **most of the removed APIs have quite rare use cases** and have quite easy replacements.

1. DGLGraph now requires the graph structure and feature data to have the same device placement. If the given node/edge feature tensors have different devices as the graph’s, dgl.ndata and dgl.edata will raise an error as follow:
    ```bash
    dgl._ffi.base.DGLError: Cannot assign node feature "x" on device cpu to a graph on device cuda:0.
    Call DGLGraph.to() to copy the graph to the same device.
    ```
    To fix it, copy either the graph (using the `DGLGraph.to` API) or the feature tensors to the same device.

1. Changes to `dgl.graph`:
    * No longer accept SciPy matrix/NetworkX graph as the input data. Use `dgl.from_scipy`/`dgl.from_networkx` instead.
    * `ntype` and `etype` are removed from the arguments. To construct graphs with named node/edge types, use `dgl.heterograph`.
        ```python
        g = dgl.heterograph(('user', 'follows', 'user') : ...)
        ```
    * `validate` is removed from the arguments. DGL now always checks whether the num_nodes is greater than the largest node ID if specified.
1. `dgl.bipartite` is removed.
    * To create a uni-directional bipartite graph, use `dgl.heterograph`. E.g.,
        ```python
        g = dgl.hetrograph(('user', 'rates', 'movie'): ...)
        ```
    * To create a uni-directional bipartite graph from a SciPy matrix, use the new API `dgl.bipartite_from_scipy`.
    * To create a uni-directional bipartite graph from a NetworkX graph, use the new API `dgl.bipartite_from_networkx`.
1. Changes to `dgl.heterograph`:
    * No longer accept SciPy matrix/NetworkX graph as the input data. Use the `from_*` APIs to create graphs first and then pass their edges to the `dgl.heterograph` API. E.g.,
        ```python
        nx_g = ...  # some networkx graph
        spmat = ...  # some scipy matrix
        g1 = dgl.from_networkx(nx_g)
        g2 = dgl.bipartite_from_scipy(spmat)
        g = dgl.heterograph({('user', 'follows', 'user') : g1.edges(),
                             ('user', 'rates', 'movie') : g2.edges()})
        ```
1. `dgl.hetero_from_relations` is removed. Use `dgl.heterograph` instead.
1. From 0.5, subgraphs extracted via DGL APIs automatically inherits node and edge features from the parent graph. DGL also saves the original nodes/edge IDs in `subg.ndata[dgl.NID]` and `subg.edata[dgl.EID]` if nodes/edges are relabeled. This new behavior makes the following `DGLGraph` methods useless and we thus remove them:
    * `DGLGraph.parent`, `DGLGraph.parent_nid`, `DGLGraph.parent_eid`, `DGLGraph.map_to_subgraph_nid`, `DGLGraph.copy_from_parent`, `DGLGraph.copy_to_parent` and `DGLGraph.detach_parent`.
1. Other removed DGLGraph APIs:
    * `DGLGraph.from_networkx`. Use `dgl.from_networkx` to construct a DGLGraph from a NetworkX graph.
    * `DGLGraph.from_scipy_sparse_matrix`. Use `dgl.from_scipy` to construct a DGLGraph from a SciPy matrix.
    * `DGLGraph.register_apply_node_func` , `DGLGraph.register_apply_edge_func`, `DGLGraph.register_message_func` and `DGLGraph.register_reduce_func`. Please specify them directly as the arguments of the message passing APIs.
        ```python
        g = ...  # some graph
        # before 0.5
        g.register_message_func(mfunc)
        g.register_reduce_func(rfunc)
        g.update_all()
        
        # starting from 0.5
        g.update_all(mfunc, rfunc)
        ```
    * `DGLGraph.group_apply_edges`. To normalize edge weights within the neighborhood of each destination node, use `dgl.nn.edge_softmax`. To normalize edge weights within the neighborhood of each source node, use `dgl.reverse` first before the edge softmax.
    * `DGLGraph.send` and `DGLGraph.recv`. There are rarely any cases where send and recv must be invoked separately. Use `DGLGraph.send_and_recv` or `DGLGraph.update_all` for message passing.
    * `DGLGraph.multi_recv`, `DGLGraph.multi_pull`, `DGLGraph.multi_send_and_recv`. To perform message passing on a part  of the nodes and edges, use `dgl.node_subgraph` or `dgl.edge_subgraph` to extract the subset first and then call `DGLGraph.multi_update_all`.
    * `DGLGraph.clear`. Use `dgl.graph(([], []))`` to create a new empty graph.
    * `DGLGraph.subgraphs`. Use `DGLGraph.subgraph`.
    * `DGLGraph.batch_num_nodes` and `DGLGraph.batch_num_edges` are now functions that accept node/edge type as the only argument for getting batching information of a heterograph.
    * `DGLGraph.flatten`. To create a new graph without batching information, use `new_g = gl.graph(old_g.edges())``.
1. The reduce function `dgl.function.prod` is removed.
1. `dgl.add_self_loop` will NOT remove existing self loops automatically. It is recommanded to call `dgl.remove_self_loop` before invoking `dgl.add_self_loop`.



## Deprecations

Will not break old codes but will throw deprecation warning.

### Core APIs

1. Creating a graph using `dgl.DGLGraph(data)` is deprecated. Use `dgl.graph(data)`.
1. Deprecated `DGLGraph` methods:
    - `DGLGraph.to_networkx` -> `dgl.to_networkx`
    - `DGLGraph.readonly` and `DGLGraph.is_readonly`. Before 0.5, this flag is a hint for more efficient implementation. From 0.5, the efficiency issue has been resolved so they become useless. 
    - `DGLGraph.__len__` -> `DGLGraph.number_of_nodes`
    - `dgl.DGLGraph.__contains__` -> `DGLGraph.has_nodes`
    - `DGLGraph.add_node` -> `DGLGraph.add_nodes`
    - `DGLGraph.add_edge` -> `DGLGraph.add_edges`
    - `DGLGraph.has_node` -> `DGLGraph.has_nodes`
    - `DGLGraph.has_edge_between` -> `DGLGraph.has_edges_between` 
    - `DGLGraph.edge_id` -> `dgl.DGLGraph.edge_ids`.
    - `DGLGraph.in_degree` -> `dgl.DGLGraph.in_degrees`.
    - `DGLGraph.out_degree` -> `dgl.DGLGraph.out_degrees`.
1. `dgl.to_simple_graph` -> `dgl.to_simple`.
1. `dgl.to_homo` -> `dgl.to_homogeneous`.
1. `dgl.to_hetero` -> `dgl.to_heterogeneous`.
1. `dgl.as_heterograph` and `dgl.as_immutable_graph` are deprecated as `dgl.DGLGraph` and `dgl.DGLHeteroGraph` are now merged.
1. `dgl.batch_hetero` -> `dgl.batch`
1. `dgl.unbatch_hetero` -> `dgl.unbatch`
1. The `node_attrs` / `edge_attrs` arguments of `dgl.batch` are renamed to `ndata` / `edata`.
1. The arguments `share_ndata` and `share_edata` of `dgl.reverse` are renamed to `copy_ndata` and `copy_edata`.

### Dataset APIs

For all the current datsets, their class attributes such as `graph`, `feat`, etc. are deprecated. The recommended usage is to get them from each sample:
```python
# Before 0.5
dataset = dgl.data.CoraFull()
g = dataset.graph
feat = dataset.feat
...

# From 0.5
dataset = dgl.data.CoraFullDataset()  # in 0.5, all the classes have a "Dataset" in the name.
g = dataset[0]  # is directly a DGLGraph object
feat = g.ndata['feat']
...
```

**Other changes**
* ``dgl.data.SST`` is deprecated and replaced by ``dgl.data.SSTDataset``. The attribute ``trees`` is deprecated and replaced by ``__getitem__``. The attribute ``num_vocabs`` is deprecated and replaced by ``vocab_size``
