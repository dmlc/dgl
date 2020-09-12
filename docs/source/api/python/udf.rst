.. _apiudf:

User-defined Functions
==================================================

.. currentmodule:: dgl.udf

User-defined functions (UDFs) allow arbitrary computation following the Message Passing Paradigm
(see :ref:`guide-message-passing`). They bring more flexibility when :ref:`apifunction` cannot
realize a desired computation.

User-defined Message Function
-----------------------------

A message function takes a batch of edges as input and returns some message(s) for each edge,
which may combine the features of the edges and their end nodes. Formally, it takes the following
form

.. code::

    def message_func(edges):
        """
        Parameters
        ----------
        edges : EdgeBatch
            A batch of edges.

        Returns
        -------
        dict[str, tensor]
            The messages generated. It maps a message name to the corresponding messages of all
            edges in the batch. The order of the messages is the same as the order of the edges
            in the input argument.
        """

DGL generates :class:`~dgl.udf.EdgeBatch` instances internally, which expose the following
interface for defining ``message_func``.

.. autosummary::
    :toctree: ../../generated/

    EdgeBatch.src
    EdgeBatch.dst
    EdgeBatch.data
    EdgeBatch.edges
    EdgeBatch.batch_size

User-defined Reduce Function
----------------------------

A reduce function takes a batch of nodes as input and returns the updated features for each node,
which may combine the current node features and the messages nodes received. Formally, it takes
the following form

.. code::

    def reduce_func(nodes):
        """
        Parameters
        ----------
        nodes : NodeBatch
            A batch of nodes.

        Returns
        -------
        dict[str, tensor]
            The updated node features. It maps a feature name to the corresponding features of
            all nodes in the batch. The order of the nodes is the same as the order of the nodes
            in the input argument.
        """

DGL generates :class:`~dgl.udf.NodeBatch` instances internally, which expose the following
interface for defining ``reduce_func``.

.. autosummary::
    :toctree: ../../generated/

    NodeBatch.data
    NodeBatch.mailbox
    NodeBatch.nodes
    NodeBatch.batch_size

Degree Bucketing
----------------

DGL employs a degree-bucketing mechanism for message passing with UDFs. It groups nodes with
a same in-degree and invokes message passing for each group of nodes. As a result, one shall
not make any assumptions about the batch size of :class:`~dgl.udf.NodeBatch` instances.

For a batch of nodes, DGL stacks the incoming messages of each node along the second dimension,
ordered by edge ID.  An example goes as follows:

.. code:: python

    >>> import dgl
    >>> import torch
    >>> import dgl.function as fn
    >>> g = dgl.graph(([1, 3, 5, 0, 4, 2, 3, 3, 4, 5], [1, 1, 0, 0, 1, 2, 2, 0, 3, 3]))
    >>> g.edata['eid'] = torch.arange(10)
    >>> def reducer(nodes):
    ...     print(nodes.mailbox['eid'])
    ...     return {'n': nodes.mailbox['eid'].sum(1)}
    >>> g.update_all(fn.copy_e('eid', 'eid'), reducer)
    tensor([[5, 6],
            [8, 9]])
    tensor([[3, 7, 2],
            [0, 1, 4]])

Essentially, node #2 and node #3 are grouped into one bucket with in-degree of 2, and node
#0 and node #1 are grouped into one bucket with in-degree of 3.  Within each bucket, the
edges are ordered by the edge IDs for each node.
