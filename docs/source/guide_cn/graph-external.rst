.. _guide_cn-graph-external:

1.4 从外部源创建图
---------------

The options to construct a :class:`~dgl.DGLGraph` from external sources include:

- Conversion from external python libraries for graphs and sparse matrices (NetworkX and SciPy).
- Loading graphs from disk.

可以从外部来源构造一个 :class:`~dgl.DGLGraph` 对象，包括：

- 从用于图和稀疏矩阵的外部Python库（NetworkX 和 SciPy）创建而来。
- 从磁盘加载图数据。

The section does not cover functions that generate graphs by transforming from other
graphs. See the API reference manual for an overview of them.

本节不涉及通过转换其他图来生成图的函数，相关概述请阅读API参考手册。

Creating Graphs from External Libraries
从外部库创建图
^^^^^^^^^^^

The following code snippet is an example for creating a graph from a SciPy sparse matrix and a NetworkX graph.

以下代码片段为从SciPy稀疏矩阵和NetworkX图创建DGL图的示例。

.. code::

    >>> import dgl
    >>> import torch as th
    >>> import scipy.sparse as sp
    >>> spmat = sp.rand(100, 100, density=0.05) # 5%非零项
    >>> dgl.from_scipy(spmat)                   # 来自SciPy
    Graph(num_nodes=100, num_edges=500,
          ndata_schemes={}
          edata_schemes={})

    >>> import networkx as nx
    >>> nx_g = nx.path_graph(5) # 一条链路0-1-2-3-4
    >>> dgl.from_networkx(nx_g) # 来自NetworkX
    Graph(num_nodes=5, num_edges=8,
          ndata_schemes={}
          edata_schemes={})

Note that when constructing from the `nx.path_graph(5)`, the resulting :class:`~dgl.DGLGraph` has 8
edges instead of 4. This is because `nx.path_graph(5)` constructs an undirected NetworkX graph
:class:`networkx.Graph` while a :class:`~dgl.DGLGraph` is always directed. In converting an undirected
NetworkX graph into a :class:`~dgl.DGLGraph`, DGL internally converts undirected edges to two directed edges.
Using directed NetworkX graphs :class:`networkx.DiGraph` can avoid such behavior.

注意，当使用 `nx.path_graph(5)` 进行创建时， :class:`~dgl.DGLGraph` 对象有8条边，而非4条。
这是由于 `nx.path_graph(5)` 构建了一个无向的NetworkX图 :class:`networkx.Graph` ，而DGLGraph对象总是有向的。
当将无向的NetworkX图转换为 :class:`~dgl.DGLGraph` 对象时，DGL会在内部将1条无向边转换为2条有向边。
使用有向的NetworkX图 :class:`networkx.DiGraph` 可避免该行为。

.. code::

    >>> nxg = nx.DiGraph([(2, 1), (1, 2), (2, 3), (0, 0)])
    >>> dgl.from_networkx(nxg)
    Graph(num_nodes=4, num_edges=4,
          ndata_schemes={}
          edata_schemes={})

.. note::

    DGL internally converts SciPy matrices and NetworkX graphs to tensors to construct graphs.
    Hence, these construction methods are not meant for performance critical parts.

    DGL在内部将SciPy矩阵和NetworkX图转换为张量来创建图。因此，这些构建方法并不适用于重视性能的场景。

See APIs: :func:`dgl.from_scipy`, :func:`dgl.from_networkx`.

相关API： :func:`dgl.from_scipy` 、 :func:`dgl.from_networkx` 。

Loading Graphs from Disk

从磁盘加载图
^^^^^^^^^^

There are many data formats for storing graphs and it isn't possible to enumerate every option.
Thus, this section only gives some general pointers on certain common ones.

有多种文件格式可储存图，所以这里难以枚举所有选项。因此，本节仅给出一些常见格式的一般情况。

Comma Separated Values (CSV)

逗号分隔值（CSV）
""""""""""""""

One very common format is CSV, which stores nodes, edges, and their features in a tabular format:

CSV是一种常见的格式，以表格格式储存节点、边及其特征：

.. table:: nodes.csv

   +-----------+
   |age, title |
   +===========+
   |43, 1      |
   +-----------+
   |23, 3      |
   +-----------+
   |...        |
   +-----------+

.. table:: edges.csv

   +-----------------+
   |src, dst, weight |
   +=================+
   |0, 1, 0.4        |
   +-----------------+
   |0, 3, 0.9        |
   +-----------------+
   |...              |
   +-----------------+

There are known Python libraries (e.g. pandas) for loading this type of data into python
objects (e.g., :class:`numpy.ndarray`), which can then be used to construct a DGLGraph. If the
backend framework also provides utilities to save/load tensors from disk (e.g., :func:`torch.save`,
:func:`torch.load`), one can follow the same principle to build a graph.

许多知名Python库（如Pandas）可以将该类型数据加载到python对象（如 :class:`numpy.ndarray` ）中，
进而使用这些对象来构建DGLGraph对象。如果后端框架也提供了从磁盘中保存或加载张量的工具（如 :func:`torch.save`,  :func:`torch.load` ），
可以遵循相同的原理来构建图。

See also: `Tutorial for loading a Karate Club Network from edge pairs CSV <https://github.com/dglai/WWW20-Hands-on-Tutorial/blob/master/basic_tasks/1_load_data.ipynb>`_.
另见： `从成对的边 CSV 文件中加载 Karate Club Network 的教程 <https://github.com/dglai/WWW20-Hands-on-Tutorial/blob/master/basic_tasks/1_load_data.ipynb>`_。

JSON/GML Format

JSON/GML 格式
""""""""""""

Though not particularly fast, NetworkX provides many utilities to parse
`a variety of data formats <https://networkx.github.io/documentation/stable/reference/readwrite/index.html>`_
which indirectly allows DGL to create graphs from these sources.

尽管并不特别快，NetworkX提供了许多工具解析`各种数据格式<https://networkx.github.io/documentation/stable/reference/readwrite/index.html>`_，DGL可以间接地从这些来源创建图。

DGL Binary Format
DGL 二进制格式
""""""""""""

DGL provides APIs to save and load graphs from disk stored in binary format. Apart from the
graph structure, the APIs also handle feature data and graph-level label data. DGL also
supports checkpointing graphs directly to S3 or HDFS. The reference manual provides more
details about the usage.

DGL提供了API以从磁盘中保存和加载二进制格式的图。除了图结构，API也能处理特征数据和图级别的标签数据。
DGL也支持直接使用S3或HDFS的检查点图。参考手册提供了该用法的更多细节。

See APIs: :func:`dgl.save_graphs`, :func:`dgl.load_graphs`.
相关API： :func:`dgl.save_graphs` 、 :func:`dgl.load_graphs` 。