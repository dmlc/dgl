.. _guide_cn-distributed-apis:

7.2 分布式的API
--------------------

:ref:`(English Version) <guide-distributed-apis>`

This section covers the distributed APIs used in the training script. DGL provides three distributed
data structures and various APIs for initialization, distributed sampling and workload split.
For distributed training/inference, DGL provides three distributed data structures:
:class:`~dgl.distributed.DistGraph` for distributed graphs, :class:`~dgl.distributed.DistTensor` for
distributed tensors and :class:`~dgl.distributed.DistEmbedding` for distributed learnable embeddings.

本节介绍了在训练脚本中使用的分布式计算API。DGL提供了三种分布式数据结构和多种API，用于初始化、分布式采样和工作负载拆分。
对于分布式训练/推断，DGL提供了三种分布式数据结构：用于分布式图的 :class:`~dgl.distributed.DistGraph`、
用于分布式张量的 :class:`~dgl.distributed.DistTensor` 和用于分布式可学习嵌入的
:class:`~dgl.distributed.DistEmbedding`。

Initialization of the DGL distributed module

DGL分布式模块的初始化
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~dgl.distributed.initialize` initializes the distributed module. When the training script runs
in the trainer mode, this API builds connections with DGL servers and creates sampler processes;
when the script runs in the server mode, this API runs the server code and never returns. This API
has to be called before any of DGL's distributed APIs. When working with Pytorch,
:func:`~dgl.distributed.initialize` has to be invoked before ``torch.distributed.init_process_group``.
Typically, the initialization APIs should be invoked in the following order:

:func:`~dgl.distributed.initialize` 可以用于初始化分布式模块。当训练脚本在训练器模式下运行时，
这个API会与DGL服务器建立连接并创建采样器进程。当脚本在服务器模式下运行时，这个API将运行服务器代码，
并且永不返回。必须在DGL的任何其他分布式API之前，调用此API。在使用PyTorch时，必须在
``torch.distributed.init_process_group`` 之前调用 :func:`~dgl.distributed.initialize`。
通常，初始化API应按以下顺序调用：

.. code:: python

    dgl.distributed.initialize('ip_config.txt', num_workers=4)
    th.distributed.init_process_group(backend='gloo')

**Note**: If the training script contains user-defined functions (UDFs) that have to be invoked on
the servers (see the section of DistTensor and DistEmbedding for more details), these UDFs have to
be declared before :func:`~dgl.distributed.initialize`.

**Note**: 如果训练脚本里包含需要在服务器(细节内容可以在下面的DistTensor和DistEmbedding章节里查看)上调用的用户定义函数(UDF)，
这些UDF必须在 :func:`~dgl.distributed.initialize` 之前声明。

Distributed graph

分布式图
~~~~~~~~~~~~~~~~~

:class:`~dgl.distributed.DistGraph` is a Python class to access the graph structure and node/edge features
in a cluster of machines. Each machine is responsible for one and only one partition. It loads
the partition data (the graph structure and the node data and edge data in the partition) and makes
it accessible to all trainers in the cluster. :class:`~dgl.distributed.DistGraph` provides a small subset
of :class:`~dgl.DGLGraph` APIs for data access.

:class:`~dgl.distributed.DistGraph` 是一个Python类，用于访问计算机集群中的图结构和节点/边特征。每台计算机负责一个且只负责一个分区。
它加载分区数据(包括分区中的图结构，节点数据和边数据)，并使集群中的所有训练器均可访问。
:class:`~dgl.distributed.DistGraph` 提供了 :class:`~dgl.DGLGraph` API的一小部分以方便数据访问。

**Note**: :class:`~dgl.distributed.DistGraph` currently only supports graphs of one node type and one edge type.

**Note**: :class:`~dgl.distributed.DistGraph` 当前仅支持一种节点类型和一种边类型的图。

Distributed mode vs. standalone mode

分布式模式与独立模式
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~dgl.distributed.DistGraph` can run in two modes: distributed mode and standalone mode.
When a user executes a training script in a Python command line or Jupyter Notebook, it runs in
a standalone mode. That is, it runs all computation in a single process and does not communicate
with any other processes. Thus, the standalone mode requires the input graph to have only one partition.
This mode is mainly used for development and testing (e.g., develop and run the code in Jupyter Notebook).
When a user executes a training script with a launch script (see the section of launch script),
:class:`~dgl.distributed.DistGraph` runs in the distributed mode. The launch tool starts servers
(node/edge feature access and graph sampling) behind the scene and loads the partition data in
each machine automatically. :class:`~dgl.distributed.DistGraph` connects with the servers in the cluster
of machines and access them through the network.

:class:`~dgl.distributed.DistGraph` 可以在两种模式下运行：分布式模式和独立模式。
当用户在Python命令行或Jupyter Notebook中执行训练脚本时，它将以独立模式运行。也就是说，它在单个进程中运行所有计算，
并且不与任何其他进程通信。因此，独立模式要求输入图仅具有一个分区。此模式主要用于开发和测试
(例如，在Jupyter Notebook中开发和运行代码)。当用户使用启动脚本执行训练脚本时(请参见启动脚本部分)，
:class:`~dgl.distributed.DistGraph` 将以分布式模式运行。启动工具在后台启动服务器(包括，节点/边特征访问和图采样)，
并将分区数据自动加载到每台计算机中。:class:`~dgl.distributed.DistGraph` 与集群中的服务器连接并通过网络访问它们。

DistGraph creation

创建DistGraph
^^^^^^^^^^^^^^^^^^

In the distributed mode, the creation of :class:`~dgl.distributed.DistGraph` requires the graph name used
during graph partitioning. The graph name identifies the graph loaded in the cluster.

在分布式模式下，:class:`~dgl.distributed.DistGraph` 的创建需要（定义）在图划分期间使用的图名称。
图名称标识了集群中所需加载的图。

.. code:: python

    import dgl
    g = dgl.distributed.DistGraph('graph_name')

When running in the standalone mode, it loads the graph data in the local machine. Therefore, users need
to provide the partition configuration file, which contains all information about the input graph.

在独立模式下运行时，DistGraph将图数据加载到本地计算机中。因此，用户需要提供分区配置文件，其中包含有关输入图的所有信息。

.. code:: python

    import dgl
    g = dgl.distributed.DistGraph('graph_name', part_config='data/graph_name.json')

**Note**: In the current implementation, DGL only allows the creation of a single DistGraph object. The behavior
of destroying a DistGraph and creating a new one is undefined.

**Note**: 在当前实现中，DGL仅允许创建单个DistGraph对象。销毁DistGraph并创建一个新DistGraph的行为没有被定义。

Access graph structure

访问图结构
^^^^^^^^^^^^^^^^^^^^^^

:class:`~dgl.distributed.DistGraph` provides a very small number of APIs to access the graph structure.
Currently, most APIs provide graph information, such as the number of nodes and edges. The main use case
of DistGraph is to run sampling APIs to support mini-batch training (see the section of distributed
graph sampling).

:class:`~dgl.distributed.DistGraph` 提供了一点API来访问图结构。当前，它们主要被用来提供图信息，例如节点和边的数量。
DistGraph的主要应用场景是运行采样API以支持小批量训练（请参阅分布式图采样一节）。

.. code:: python

    print(g.number_of_nodes())

Access node/edge data

访问节点/边数据
^^^^^^^^^^^^^^^^^^^^^

Like :class:`~dgl.DGLGraph`, :class:`~dgl.distributed.DistGraph` provides ``ndata`` and ``edata``
to access data in nodes and edges.
The difference is that ``ndata``/``edata`` in :class:`~dgl.distributed.DistGraph` returns
:class:`~dgl.distributed.DistTensor`, instead of the tensor of the underlying framework.
Users can also assign a new :class:`~dgl.distributed.DistTensor` to
:class:`~dgl.distributed.DistGraph` as node data or edge data.

与 :class:`~dgl.DGLGraph`一样， :class:`~dgl.distributed.DistGraph` 提供
``ndata`` 和 ``edata`` 来访问节点和边中的数据。它们的区别在于
:class:`~dgl.distributed.DistGraph` 中的 ``ndata`` / ``edata`` 返回 :class:`~dgl.distributed.DistTensor`，
而不是底层框架里的张量。用户还可以将新的 :class:`~dgl.distributed.DistTensor` 分配给
:class:`~dgl.distributed.DistGraph` 作为节点数据或边数据。

.. code:: python

    g.ndata['train_mask']
    <dgl.distributed.dist_graph.DistTensor at 0x7fec820937b8>
    g.ndata['train_mask'][0]
    tensor([1], dtype=torch.uint8)

Distributed Tensor

分布式张量
~~~~~~~~~~~~~~~~~

As mentioned earlier, DGL shards node/edge features and stores them in a cluster of machines.
DGL provides distributed tensors with a tensor-like interface to access the partitioned
node/edge features in the cluster. In the distributed setting, DGL only supports dense node/edge
features.

如前所述，在分布式模式下，DGL会分片节点/边特征并将其存储在计算机集群中。
DGL为分布式张量提供了类似于张量的接口，以访问群集中的分区节点/边特征。
在分布式设置中，DGL仅支持密集节点/边特征。

:class:`~dgl.distributed.DistTensor` manages the dense tensors partitioned and stored in
multiple machines. Right now, a distributed tensor has to be associated with nodes or edges
of a graph. In other words, the number of rows in a DistTensor has to be the same as the number
of nodes or the number of edges in a graph. The following code creates a distributed tensor.
In addition to the shape and dtype for the tensor, a user can also provide a unique tensor name.
This name is useful if a user wants to reference a persistent distributed tensor (the one exists
in the cluster even if the :class:`~dgl.distributed.DistTensor` object disappears).

:class:`~dgl.distributed.DistTensor` 管理在多个计算机中分区和存储的密集张量。
目前，分布式张量必须与图的节点或边相关联。换句话说，DistTensor中的行数必须与图中的节点数或边数相同。
以下代码创建一个分布式张量。 除了张量的形状和数据类型之外，用户还可以提供唯一的张量名称。
如果用户要引用一个固定的分布式张量(即使 :class:`~dgl.distributed.DistTensor` 对象消失，该名称仍存在于群集中)，
则(使用这样的)名称就很有用。

.. code:: python

    tensor = dgl.distributed.DistTensor((g.number_of_nodes(), 10), th.float32, name=’test’)

**Note**: :class:`~dgl.distributed.DistTensor` creation is a synchronized operation. All trainers
have to invoke the creation and the creation succeeds only when all trainers call it. 

**Note**: :class:`~dgl.distributed.DistTensor` 创建是一个同步操作。所有训练器都必须调用创建，
并且只有当所有训练器都调用它时，此创建过程才能成功。

A user can add a :class:`~dgl.distributed.DistTensor` to a :class:`~dgl.distributed.DistGraph`
object as one of the node data or edge data.

用户可以将 :class:`~dgl.distributed.DistTensor` 作为节点数据或边数据之一添加到
:class:`~dgl.distributed.DistGraph` 对象。

.. code:: python

    g.ndata['feat'] = tensor

**Note**: The node data name and the tensor name do not have to be the same. The former identifies
node data from :class:`~dgl.distributed.DistGraph` (in the trainer process) while the latter
identifies a distributed tensor in DGL servers.

**Note**: 节点数据名称和张量名称不必相同。 前者从 :class:`~dgl.distributed.DistGraph` 中标识节点数据(在训练器进程中)，
而后者则标识DGL服务器中的分布式张量。

:class:`~dgl.distributed.DistTensor` provides a small set of functions. It has the same APIs as
regular tensors to access its metadata, such as the shape and dtype.
:class:`~dgl.distributed.DistTensor` supports indexed reads and writes but does not support
computation operators, such as sum and mean.

:class:`~dgl.distributed.DistTensor` 提供了一些功能。它具有与常规张量相同的API，用于访问其元数据，
例如形状和数据类型。:class:`~dgl.distributed.DistTensor` 支持索引读取和写入，
但不支持计算运算符，例如求总和以及求均值。

.. code:: python

    data = g.ndata['feat'][[1, 2, 3]]
    print(data)
    g.ndata['feat'][[3, 4, 5]] = data

**Note**: Currently, DGL does not provide protection for concurrent writes from multiple trainers
when a machine runs multiple servers. This may result in data corruption. One way to avoid concurrent
writes to the same row of data is to run one server process on a machine.

**Note**: 当前，当一台机器运行多个服务器时，DGL不提供对来自多个训练器的并发写入的保护。
这可能会导致数据损坏。避免同时写入同一行数据的一种方法是在一个计算机上只运行一个服务器进程。

Distributed Embedding

分布式嵌入
~~~~~~~~~~~~~~~~~~~~~

DGL provides :class:`~dgl.distributed.DistEmbedding` to support transductive models that require
node embeddings. Creating distributed embeddings is very similar to creating distributed tensors.

DGL提供 :class:`~dgl.distributed.DistEmbedding` 以支持需要节点嵌入的直推模型。创建分布式嵌入与创建分布式张量非常相似。

.. code:: python

    def initializer(shape, dtype):
        arr = th.zeros(shape, dtype=dtype)
        arr.uniform_(-1, 1)
        return arr
    emb = dgl.distributed.DistEmbedding(g.number_of_nodes(), 10, init_func=initializer)

Internally, distributed embeddings are built on top of distributed tensors, and, thus, has
very similar behaviors to distributed tensors. For example, when embeddings are created, they
are sharded and stored across all machines in the cluster. It can be uniquely identified by a name.

在内部，分布式嵌入建立在分布式张量之上，因此，其行为与分布式张量非常相似。
例如，创建嵌入时，会将它们分片并存储在集群中的所有计算机上。(分布式嵌入)可以通过名称唯一标识。

**Note**: The initializer function is invoked in the server process. Therefore, it has to be
declared before :class:`~dgl.distributed.initialize`.

**Note**: 服务器进程负责调用初始化函数。 因此，必须在初始化( :class:`~dgl.distributed.initialize` )之前声明它。

Because the embeddings are part of the model, a user has to attach them to an optimizer for
mini-batch training. Currently, DGL provides a sparse Adagrad optimizer
:class:`~dgl.distributed.SparseAdagrad` (DGL will add more optimizers for sparse embeddings later).
Users need to collect all distributed embeddings from a model and pass them to the sparse optimizer.
If a model has both node embeddings and regular dense model parameters and users want to perform
sparse updates on the embeddings, they need to create two optimizers, one for node embeddings and
the other for dense model parameters, as shown in the code below:

因为嵌入是模型的一部分，所以用户必须将其附加到优化器上以进行小批量训练。当前，
DGL提供了一个稀疏的Adagrad优化器 :class:`~dgl.distributed.SparseAdagrad` (DGL以后将为稀疏嵌入添加更多的优化器)。
用户需要从模型中收集所有分布式嵌入，并将它们传递给稀疏优化器。如果模型同时具有节点嵌入和规则的密集模型参数，
并且用户希望对嵌入执行稀疏更新，则他们需要创建两个优化器，一个用于节点嵌入，另一个用于密集模型参数，如以下代码所示：

.. code:: python

    sparse_optimizer = dgl.distributed.SparseAdagrad([emb], lr=lr1)
    optimizer = th.optim.Adam(model.parameters(), lr=lr2)
    feats = emb(nids)
    loss = model(feats)
    loss.backward()
    optimizer.step()
    sparse_optimizer.step()

**Note**: :class:`~dgl.distributed.DistEmbedding` is not an Pytorch nn module, so we cannot
get access to it from parameters of a Pytorch nn module.

**Note**: :class:`~dgl.distributed.DistEmbedding` 不是PyTorch的nn模块，因此用户无法从nn模块的参数访问它。

Distributed sampling

分布式采样
~~~~~~~~~~~~~~~~~~~~

DGL provides two levels of APIs for sampling nodes and edges to generate mini-batches
(see the section of mini-batch training). The low-level APIs require users to write code
to explicitly define how a layer of nodes are sampled (e.g., using :func:`dgl.sampling.sample_neighbors` ).
The high-level sampling APIs implement a few popular sampling algorithms for node classification
and link prediction tasks (e.g., :class:`~dgl.dataloading.pytorch.NodeDataloader` and
:class:`~dgl.dataloading.pytorch.EdgeDataloader` ).

DGL提供了两个级别的API，用于对节点和边进行采样以生成小批量(请参阅小批量培训部分)。
底层API要求用户编写代码以明确定义如何对节点层进行采样(例如，使用 :func:`dgl.sampling.sample_neighbors` )。
上一层采样API为节点分类和Link Prediction任务实现了一些流行的采样算法（例如
:class:`~dgl.dataloading.pytorch.NodeDataloader`
和
:class:`~dgl.dataloading.pytorch.EdgeDataloader` )。

The distributed sampling module follows the same design and provides two levels of sampling APIs.
For the lower-level sampling API, it provides :func:`~dgl.distributed.sample_neighbors` for
distributed neighborhood sampling on :class:`~dgl.distributed.DistGraph`. In addition, DGL provides
a distributed Dataloader (:class:`~dgl.distributed.DistDataLoader` ) for distributed sampling.
The distributed Dataloader has the same interface as Pytorch DataLoader except that users cannot
specify the number of worker processes when creating a dataloader. The worker processes are created
in :func:`dgl.distributed.initialize`.

分布式采样模块采用相同的设计，并提供两个级别的采样API。对于底层的采样API，它为
:class:`~dgl.distributed.DistGraph` 上的分布式邻域采样提供了
:func:`~dgl.distributed.sample_neighbors`。另外，DGL提供了用于分布式采样的分布式数据加载器(
:class:`~dgl.distributed.DistDataLoader`)。除了用户在创建数据加载器时无法指定工作进程的数量，
分布式数据加载器具有与PyTorch DataLoader相同的接口。 工作进程在 :func:`dgl.distributed.initialize` 中创建。

**Note**: When running :func:`dgl.distributed.sample_neighbors` on :class:`~dgl.distributed.DistGraph`,
the sampler cannot run in Pytorch Dataloader with multiple worker processes. The main reason is that
Pytorch Dataloader creates new sampling worker processes in every epoch, which leads to creating and
destroying :class:`~dgl.distributed.DistGraph` objects many times.

**Note**: 在 :class:`~dgl.distributed.DistGraph` 上运行 :func:`dgl.distributed.sample_neighbors` 时，
采样器无法在具有多个工作进程的PyTorch Dataloader中运行。主要原因是PyTorch Dataloader在每个训练周期都会创建新的采样工作进程，
从而导致多次创建和删除 :class:`~dgl.distributed.DistGraph` 对象。

The same high-level sampling APIs (:class:`~dgl.dataloading.pytorch.NodeDataloader` and
:class:`~dgl.dataloading.pytorch.EdgeDataloader` ) work for both :class:`~dgl.DGLGraph`
and :class:`~dgl.distributed.DistGraph`. When using :class:`~dgl.dataloading.pytorch.NodeDataloader`
and :class:`~dgl.dataloading.pytorch.EdgeDataloader`, the distributed sampling code is exactly
the same as single-process sampling.

:class:`~dgl.DGLGraph` 和 :class:`~dgl.distributed.DistGraph` 都可以使用相同的高级采样API(
:class:`~dgl.dataloading.pytorch.NodeDataloader`
和
:class:`~dgl.dataloading.pytorch.EdgeDataloader`)。使用
:class:`~dgl.dataloading.pytorch.NodeDataloader`
和
:class:`~dgl.dataloading.pytorch.EdgeDataloader` 时，分布式采样代码与单进程采样完全相同。

When using the low-level API, the sampling code is similar to single-process sampling. The only
difference is that users need to use :func:`dgl.distributed.sample_neighbors` and
:class:`~dgl.distributed.DistDataLoader`.

使用底层API时，采样代码类似于单进程采样。唯一的区别是用户需要使用
:func:`dgl.distributed.sample_neighbors`
和
:class:`~dgl.distributed.DistDataLoader`。

.. code:: python

    def sample_blocks(seeds):
        seeds = th.LongTensor(np.asarray(seeds))
        blocks = []
        for fanout in [10, 25]:
            frontier = dgl.distributed.sample_neighbors(g, seeds, fanout, replace=True)
            block = dgl.to_block(frontier, seeds)
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
            return blocks
        dataloader = dgl.distributed.DistDataLoader(dataset=train_nid,
                                                    batch_size=batch_size,
                                                    collate_fn=sample_blocks,
                                                    shuffle=True)
        for batch in dataloader:
            ...

When using the high-level API, the distributed sampling code is identical to the single-machine sampling:

使用高级API时，分布式采样代码与单机采样相同：

.. code:: python

    sampler = dgl.sampling.MultiLayerNeighborSampler([10, 25])
    dataloader = dgl.sampling.NodeDataLoader(g, train_nid, sampler,
                                             batch_size=batch_size, shuffle=True)
    for batch in dataloader:
        ... 


Split workloads

分割工作量
~~~~~~~~~~~~~~~

Users need to split the training set so that each trainer works on its own subset. Similarly,
we also need to split the validation and test set in the same way.

用户需要分割训练集，以便每个训练器都可以使用自己的训练集子集。同样，用户还需要以相同的方式分割验证和测试集。

For distributed training and evaluation, the recommended approach is to use boolean arrays to
indicate the training/validation/test set. For node classification tasks, the length of these
boolean arrays is the number of nodes in a graph and each of their elements indicates the existence
of a node in a training/validation/test set. Similar boolean arrays should be used for
link prediction tasks.

对于分布式训练和评估，推荐的方法是使用布尔数组表示训练/验证/测试集。对于节点分类任务，
这些布尔数组的长度是图中节点的数量，并且它们的每个元素都表示训练/验证/测试集中是否存在对应节点。
link prediction也应使用类似的布尔数组。

DGL provides :func:`~dgl.distributed.node_split` and :func:`~dgl.distributed.edge_split` to
split the training, validation and test set at runtime for distributed training. The two functions
take the boolean arrays as input, split them and return a portion for the local trainer.
By default, they ensure that all portions have the same number of nodes/edges. This is
important for synchronous SGD, which assumes each trainer has the same number of mini-batches.

DGL提供了 :func:`~dgl.distributed.node_split` 和 :func:`~dgl.distributed.edge_split`
在运行时拆分训练、验证和测试集，以进行分布式训练。这两个函数将布尔数组作为输入，将其拆分并为本地训练器返回一部分。
默认情况下，它们确保所有部分都具有相同数量的节点/边。这对于同步SGD非常重要，
因为它（同步SGD）假定每个训练器具有相同数量的小批量。

The example below splits the training set and returns a subset of nodes for the local process.

下面的示例拆分训练集，并返回本地进程的节点子集。

.. code:: python

    train_nids = dgl.distributed.node_split(g.ndata['train_mask'])

