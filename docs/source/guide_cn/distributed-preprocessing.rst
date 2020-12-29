.. _guide_cn-distributed-preprocessing:

7.1 分布式训练所需的图数据预处理
------------------------------------------

:ref:`(English Version) <guide-distributed-preprocessing>`

DGL requires preprocessing the graph data for distributed training, including two steps:
1) partition a graph into subgraphs, 2) assign nodes/edges with new Ids. DGL provides
a partitioning API that performs the two steps. The API supports both random partitioning
and a `Metis <http://glaros.dtc.umn.edu/gkhome/views/metis>`__-based partitioning.
The benefit of Metis partitioning is that it can generate
partitions with minimal edge cuts that reduces network communication for distributed training
and inference. DGL uses the latest version of Metis with the options optimized for the real-world
graphs with power-law distribution. After partitioning, the API constructs the partitioned results
in a format that is easy to load during the training.

DGL要求预处理图数据以进行分布式训练，这包括两个步骤：1)将一张图划分为多张子图，2)为节点/边分配新的ID。
DGL提供了一个API以执行这两个步骤。该API同时支持随机划分和一个基于
`Metis <http://glaros.dtc.umn.edu/gkhome/views/metis>`__ 的划分。Metis划分的好处在于，
它可以用最少的边分割生成划分，从而减少了用于分布式训练和推理的网络通信。DGL使用最新版本的Metis，
并针对真实世界中具有幂律分布的图进行了优化。 在图划分后，API以易于在训练期间加载的格式构造划分结果。

**Note**: The graph partition API currently runs on one machine. Therefore, if a graph is large,
users will need a large machine to partition a graph. In the future, DGL will support distributed
graph partitioning.

**Note**: 图划分API当前在一台机器上运行。 因此如果一张图很大，用户将需要一台大型机器来对图进行划分。
未来DGL将支持分布式图划分。

By default, the partition API assigns new IDs to the nodes and edges in the input graph to help locate
nodes/edges during distributed training/inference. After assigning IDs, the partition API shuffles
all node data and edge data accordingly. During the training, users just use the new node/edge IDs.
However, the original IDs are still accessible through ``g.ndata['orig_id']`` and ``g.edata['orig_id']``,
where ``g`` is a DistGraph object (see the section of DistGraph).

默认情况下，为了在分布式训练/推理期间定位节点/边，API将新ID分配给输入图的节点和边。
分配ID后，该API会相应地打乱所有节点数据和边数据。在训练期间，用户只需使用新的节点/边ID。
与此同时，用户仍然可以通过 ``g.ndata['orig_id']`` 和 ``g.edata['orig_id']`` 获取原始ID。
其中 ``g`` 是 ``DistGraph`` 对象（请参见7.2里的DistGraph小节）。

The partitioned results are stored in multiple files in the output directory. It always contains
a JSON file called xxx.json, where xxx is the graph name provided to the partition API. The JSON file
contains all the partition configurations. If the partition API does not assign new IDs to nodes and edges,
it generates two additional Numpy files: `node_map.npy` and `edge_map.npy`, which stores the mapping between
node/edge IDs and partition IDs. The Numpy arrays in the two files are large for a graph with billions of
nodes and edges because they have an entry for each node and edge in the graph. Inside the folders for
each partition, there are three files that store the partition data in the DGL format. `graph.dgl` stores
the graph structure of the partition as well as some metadata on nodes and edges. `node_feats.dgl` and
`edge_feats.dgl` stores all features of nodes and edges that belong to the partition. 

DGL将划分结果存储在输出目录中的多个文件中。它始终包含一个名为xxx.json的JSON文件，其中xxx是提供给划分API的图名称。
JSON文件包含所有划分配置。如果该API没有为节点和边分配新ID，它将生成两个额外的NumPy文件：`node_map.npy` 和 `edge_map.npy`。
它们存储节点/边ID与分区ID之间的映射。对于具有十亿级数量节点和边的图，两个文件中的NumPy数组很大。
这是因为图中的每个节点和边都对应一个条目。在每个分区的文件夹内，有三个文件以DGL格式存储分区数据。
`graph.dgl` 存储分区的图结构以及节点和边上的一些元数据。`node_feats.dgl` 和 `edge_feats.dgl` 存储属于该分区的节点和边的所有特征。

.. code-block:: none

    data_root_dir/
        |-- xxx.json             # partition configuration file in JSON # JSON中的分区配置文件
        |-- node_map.npy         # partition id of each node stored in a numpy array (optional) 存储在NumPy数组中的每个节点的分区ID（可选）
        |-- edge_map.npy         # partition id of each edge stored in a numpy array (optional) 存储在NumPy数组中的每个边的分区ID（可选）
        |-- part0/               # data for partition 0 划分0的数据
            |-- node_feats.dgl   # node features stored in binary format 以二进制格式存储的节点特征
            |-- edge_feats.dgl   # edge features stored in binary format 以二进制格式存储的边特征
            |-- graph.dgl        # graph structure of this partition stored in binary format 以二进制格式存储的子图结构
        |-- part1/               # data for partition 1 划分1的数据
            |-- node_feats.dgl
            |-- edge_feats.dgl
            |-- graph.dgl

Load balancing

负载均衡
~~~~~~~~~~~~~~

When partitioning a graph, by default, Metis only balances the number of nodes in each partition.
This can result in suboptimal configuration, depending on the task at hand. For example, in the case
of semi-supervised node classification, a trainer performs computation on a subset of labeled nodes in
a local partition. A partitioning that only balances nodes in a graph (both labeled and unlabeled), may
end up with computational load imbalance. To get a balanced workload in each partition, the partition API
allows balancing between partitions with respect to the number of nodes in each node type, by specifying
``balance_ntypes`` in :func:`dgl.distributed.partition_graph`. Users can take advantage of this and consider
nodes in the training set, validation set and test set are of different node types.

在对图进行划分时，默认情况下，Metis仅平衡每个子图中的节点数。根据当前的任务情况，这可能导致配置欠佳。
例如，在半监督节点分类的情况下，这训练器对局部分区中带标签节点的子集执行计算。
一个仅平衡图中节点（带标签和未带标签）的划分可能会导致计算负载不平衡。为了在每个分区中获得平衡的工作负载，
划分API通过在 :func:`dgl.distributed.partition_graph` 中指定 ``balance_ntypes``
在每个节点类型中的节点数上实现分区间的平衡。用户可以利用这一点将训练集、验证集和测试集中的节点看作不同类型的节点。

The following example considers nodes inside the training set and outside the training set are two types of nodes:

以下示例将训练集内和训练集外的节点看作两种类型的节点：

.. code:: python

    dgl.distributed.partition_graph(g, ‘graph_name’, 4, ‘/tmp/test’, balance_ntypes=g.ndata[‘train_mask’])

In addition to balancing the node types, :func:`dgl.distributed.partition_graph` also allows balancing
between in-degrees of nodes of different node types by specifying ``balance_edges``. This balances
the number of edges incident to the nodes of different types.

除了平衡节点的类型之外， :func:`dgl.distributed.partition_graph` 还允许通过指定
``balance_edges`` 来平衡每个类型节点在子图中的入度。这平衡了不同类型节点的连边数量。

**Note**: The graph name passed to :func:`dgl.distributed.partition_graph` is an important argument.
The graph name will be used by :class:`dgl.distributed.DistGraph` to identify a distributed graph.
A legal graph name should only contain alphabetic characters and underscores.

**Note**: 传给 :func:`dgl.distributed.partition_graph` 的图名称是一个重要的参数。
:class:`dgl.distributed.DistGraph` 使用该名称来识别一个分布式的图。一个有效的图名称应该仅包含字母和下划线。
