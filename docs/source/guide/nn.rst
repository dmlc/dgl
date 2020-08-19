.. _guide-nn:

Chapter 3: Building GNN Modules
=====================================

DGL NN module is the building block for your GNN model. It inherents
from `Pytorch’s NN Module <https://pytorch.org/docs/1.2.0/_modules/torch/nn/modules/module.html>`__, `MXNet Gluon’s NN Block  <http://mxnet.incubator.apache.org/versions/1.6/api/python/docs/api/gluon/nn/index.html>`__ and `TensorFlow’s Keras
Layer <https://www.tensorflow.org/api_docs/python/tf/keras/layers>`__, depending on the DNN framework backend in use. In DGL NN
module, the parameter registration in construction function and tensor
operation in forward function are the same with the backend framework.
In this way, DGL code can be seamlessly integrated into the backend
framework code. The major difference lies in the message passing
operations that are unique in DGL.

DGL has integrated many commonly used
:ref:`apinn-pytorch-conv`, :ref:`apinn-pytorch-dense-conv`, :ref:`apinn-pytorch-pooling`,
and
:ref:`apinn-pytorch-util`. We welcome your contribution!

In this section, we will use
:class:`~dgl.nn.pytorch.conv.SAGEConv`
with Pytorch backend as an example to introduce how to build your own
DGL NN Module.

DGL NN Module Construction Function
-----------------------------------

The construction function will do the following:

1. Set options.
2. Register learnable paramesters or submodules.
3. Reset parameters.

.. code::

    import torch as th
    from torch import nn
    from torch.nn import init

    from .... import function as fn
    from ....base import DGLError
    from ....utils import expand_as_pair, check_eq_shape

    class SAGEConv(nn.Module):
        def __init__(self,
                     in_feats,
                     out_feats,
                     aggregator_type,
                     bias=True,
                     norm=None,
                     activation=None,
                     allow_zero_in_degree=False):
            super(SAGEConv, self).__init__()

            self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
            self._out_feats = out_feats
            self._aggre_type = aggregator_type
            self.norm = norm
            self.activation = activation
            self._allow_zero_in_degree = allow_zero_in_degree

In construction function, we first need to set the data dimensions. For
general Pytorch module, the dimensions are usually input dimension,
output dimension and hidden dimensions. For graph neural, the input
dimension can be split into source node dimension and destination node
dimension.

Besides data dimensions, a typical option for graph neural network is
aggregation type (``self._aggre_type``). Aggregation type determines how
messages on different edges are aggregated for a certain destination
node. Commonly used aggregation types include ``mean``, ``sum``,
``max``, ``min``. Some modules may apply more complicated aggregation
like a ``lstm``.

``norm`` here is a callable function for feature normalization. On the
SAGEConv paper, such normalization can be l2 norm:
:math:`h_v = h_v / \lVert h_v \rVert_2`.

.. code::

            # aggregator type: mean, max_pool, lstm, gcn
            if aggregator_type not in ['mean', 'max_pool', 'lstm', 'gcn']:
                raise KeyError('Aggregator type {} not supported.'.format(aggregator_type))
            if aggregator_type == 'max_pool':
                self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
            if aggregator_type == 'lstm':
                self.lstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True)
            if aggregator_type in ['mean', 'max_pool', 'lstm']:
                self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
            self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=bias)
            self.reset_parameters()

Register parameters and submodules. In SAGEConv, submodules vary
according to the aggregation type. Those modules are pure Pytorch nn
modules like ``nn.Linear``, ``nn.LSTM``, etc. At the end of construction
function, weight initialization is applied by calling
``reset_parameters()``.

.. code::

        def reset_parameters(self):
            """Reinitialize learnable parameters."""
            gain = nn.init.calculate_gain('relu')
            if self._aggre_type == 'max_pool':
                nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
            if self._aggre_type == 'lstm':
                self.lstm.reset_parameters()
            if self._aggre_type != 'gcn':
                nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
            nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

DGL NN Module Forward Function
----------------------------------

In NN module, ``forward()`` function does the actual message passing and
computating. Compared with Pytorch’s NN module which usually takes
tensors as the parameters, DGL NN module takes an additional parameter
`DGLGraph <https://docs.dgl.ai/api/python/graph.html>`__. The
workload for ``forward()`` function can be splitted into three parts:

-  Graph checking and graph type specification.

-  Message passing and reducing.

-  Update feature after reducing for output.

Let’s dive deep into the ``forward()`` function in SAGEConv example.

Graph checking and graph type specification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code::

        def forward(self, graph, feat):
            with graph.local_scope():
                # Graph checking
                if not self._allow_zero_in_degree:
                    if (graph.in_degrees() == 0).any():
                        raise DGLError('There are 0-in-degree nodes in the graph,
                                      'output for those nodes will be invalid.'
                                      'This is harmful for some applications, '
                                      'causing silent performance regression.'
                                      'Adding self-loop on the input graph by calling
                                      '`g = dgl.add_self_loop(g)` will resolve the issue.'
                                      'Setting ``allow_zero_in_degree`` to be `True`
                                      'when constructing this module will suppress the '
                                      'check and let the code run.')
                # Specify graph type then expand input feature according to graph type
                feat_src, feat_dst = expand_as_pair(feat, graph)

**This part of code is usually shared by all the NN modules.**

``forward()`` needs to handle many corner cases on the input that can
lead to invalid values in computing and message passing. The above
example handles the case where there are 0-in-degree nodes in the input
graph.

When a node has 0-in-degree, the ``mailbox`` will be empty and the
reduce function will not produce valid values. For example, if the
reduce function is ``max``, the output for the 0-in-degree nodes
will be ``-inf``.

DGL NN module should be reusable across different types of graph input
including: homogeneous graph, `heterogeneous
graph <https://docs.dgl.ai/tutorials/basics/5_hetero.html>`__, `subgraph
block <https://docs.dgl.ai/guide/minibatch.html>`__.

The math formulas for SAGEConv are:

.. math::


   h_{\mathcal{N}(dst)}^{(l+1)}  = \mathrm{aggregate}
           \left(\{h_{src}^{l}, \forall src \in \mathcal{N}(dst) \}\right)

.. math::

    h_{dst}^{(l+1)} = \sigma \left(W \cdot \mathrm{concat}
           (h_{dst}^{l}, h_{\mathcal{N}(dst)}^{l+1} + b) \right)

.. math::

    h_{dst}^{(l+1)} = \mathrm{norm}(h_{dst}^{l})

We need to specify the source node feature ``feat_src`` and destination
node feature ``feat_dst`` according to the graph type. The function to
specify the graph type and expand ``feat`` into ``feat_src`` and
``feat_dst`` is
`expand_as_pair() <https://github.com/dmlc/dgl/blob/master/python/dgl/utils/internal.py#L553>`__.
The detail of this function is shown below.

.. code::

    def expand_as_pair(input_, g=None):
        if isinstance(input_, tuple):
            # Bipartite graph case
            return input_
        elif g is not None and g.is_block:
            # Subgraph block case
            if isinstance(input_, Mapping):
                input_dst = {
                    k: F.narrow_row(v, 0, g.number_of_dst_nodes(k))
                    for k, v in input_.items()}
            else:
                input_dst = F.narrow_row(input_, 0, g.number_of_dst_nodes())
            return input_, input_dst
        else:
            # Homograph case
            return input_, input_

For homogeneous whole graph training, source nodes and destination nodes
are the same. They are all the nodes in the graph.

For heterogeneous case, the graph can be splitted into several bipartite
graphs, one for each relation. The relations are represented as
``(src_type, edge_type, dst_dtype)``. When we identify the input feature
``feat`` is a tuple, we will treat the graph as bipartite. The first
element in the tuple will be the source node feature and the second
element will be the destination node feature.

In mini-batch training, the computing is applied on a subgraph sampled
by given a bunch of destination nodes. The subgraph is called as
``block`` in DGL. After message passing, only those destination nodes
will be updated since they have the same neighborhood as the one they
have in the original full graph. In the block creation phase,
``dst nodes`` are in the front of the node list. We can find the
``feat_dst`` by the index ``[0:g.number_of_dst_nodes()]``.

After determining ``feat_src`` and ``feat_dst``, the computing for the
above three graph types are the same.

Message passing and reducing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code::

                if self._aggre_type == 'mean':
                    graph.srcdata['h'] = feat_src
                    graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))
                    h_neigh = graph.dstdata['neigh']
                elif self._aggre_type == 'gcn':
                    check_eq_shape(feat)
                    graph.srcdata['h'] = feat_src
                    graph.dstdata['h'] = feat_dst     # same as above if homogeneous
                    graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'neigh'))
                    # divide in_degrees
                    degs = graph.in_degrees().to(feat_dst)
                    h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
                elif self._aggre_type == 'max_pool':
                    graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
                    graph.update_all(fn.copy_u('h', 'm'), fn.max('m', 'neigh'))
                    h_neigh = graph.dstdata['neigh']
                else:
                    raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

                # GraphSAGE GCN does not require fc_self.
                if self._aggre_type == 'gcn':
                    rst = self.fc_neigh(h_neigh)
                else:
                    rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)

The code actually does message passing and reducing computing. This part
of code varies module by module. Note that all the message passings in
the above code are implemented using ``update_all()`` API and
``built-in`` message/reduce functions to fully utilize DGL’s performance
optimization as described in :ref:`guide-message-passing`.

Update feature after reducing for output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code::

                # activation
                if self.activation is not None:
                    rst = self.activation(rst)
                # normalization
                if self.norm is not None:
                    rst = self.norm(rst)
                return rst

The last part of ``forward()`` function is to update the feature after
the ``reduce function``. Common update operations are applying
activation function and normalization according to the option set in the
object construction phase.

Heterogeneous GraphConv Module
------------------------------

:class:`dgl.nn.pytorch.HeteroGraphConv`
is a module-level encapsulation to run DGL NN module on heterogeneous
graph. The implementation logic is the same as message passing level API
``multi_update_all()``:

-  DGL NN module within each relation :math:`r`.
-  Reduction that merges the results on the same node type from multiple
   relationships.

This can be formulated as:

.. math::  h_{dst}^{(l+1)} = \underset{r\in\mathcal{R}, r_{dst}=dst}{AGG} (f_r(g_r, h_{r_{src}}^l, h_{r_{dst}}^l))

where :math:`f_r` is the NN module for each relation :math:`r`,
:math:`AGG` is the aggregation function.

HeteroGraphConv implementation logic:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code::

    class HeteroGraphConv(nn.Module):
        def __init__(self, mods, aggregate='sum'):
            super(HeteroGraphConv, self).__init__()
            self.mods = nn.ModuleDict(mods)
            if isinstance(aggregate, str):
                self.agg_fn = get_aggregate_fn(aggregate)
            else:
                self.agg_fn = aggregate

The heterograph convolution takes a dictonary ``mods`` that maps each
relation to a nn module. And set the function that aggregates results on
the same node type from multiple relations.

.. code::

    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = {nty : [] for nty in g.dsttypes}

Besides input graph and input tensors, the ``forward()`` function takes
two additional dictionary parameters ``mod_args`` and ``mod_kwargs``.
These two dictionaries have the same keys as ``self.mods``. They are
used as customized parameters when calling their corresponding NN
modules in ``self.mods``\ for different types of relations.

An output dictionary is created to hold output tensor for each
destination type\ ``nty`` . Note that the value for each ``nty`` is a
list, indicating a single node type may get multiple outputs if more
than one relations have ``nty`` as the destination type. We will hold
them in list for further aggregation.

.. code::

          if g.is_block:
              src_inputs = inputs
              dst_inputs = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
          else:
              src_inputs = dst_inputs = inputs

          for stype, etype, dtype in g.canonical_etypes:
              rel_graph = g[stype, etype, dtype]
              if rel_graph.number_of_edges() == 0:
                  continue
              if stype not in src_inputs or dtype not in dst_inputs:
                  continue
              dstdata = self.mods[etype](
                  rel_graph,
                  (src_inputs[stype], dst_inputs[dtype]),
                  *mod_args.get(etype, ()),
                  **mod_kwargs.get(etype, {}))
              outputs[dtype].append(dstdata)

The input ``g`` can be a heterogeneous graph or a subgraph block from a
heterogeneous graph. As in ordinary NN module, the ``forward()``
function need to handle different input graph types separately.

Each relation is represented as a ``canonical_etype``, which is
``(stype, etype, dtype)``. Using ``canonical_etype`` as the key, we can
extract out a bipartite graph ``rel_graph``. For bipartite graph, the
input feature will be organized as a tuple
``(src_inputs[stype], dst_inputs[dtype])``. The NN module for each
relation is called and the output is saved. To avoid unnecessary call,
relations with no edge or no node with the its src type will be skipped.

.. code::

        rsts = {}
        for nty, alist in outputs.items():
            if len(alist) != 0:
                rsts[nty] = self.agg_fn(alist, nty)

Finally, the results on the same destination node type from multiple
relationships are aggregated using ``self.agg_fn`` function.

HeteroGraphConv examplar usage code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a heterograph
^^^^^^^^^^^^^^^^^^^^

.. code::

    >>> import dgl
    >>> g = dgl.heterograph({
    >>>     ('user', 'follows', 'user') : edges1,
    >>>     ('user', 'plays', 'game') : edges2,
    >>>     ('store', 'sells', 'game')  : edges3})

This heterograph has three types of relations and nodes.

Create a HeteroGraphConv module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code::

    >>> import dgl.nn.pytorch as dglnn
    >>> conv = dglnn.HeteroGraphConv({
    >>>     'follows' : dglnn.GraphConv(...),
    >>>     'plays' : dglnn.GraphConv(...),
    >>>     'sells' : dglnn.SAGEConv(...)},
    >>>     aggregate='sum')

This module applies different convolution modules to different
relations. Note that the modules for ``'follows'`` and ``'plays'`` do
not share weights. The ``aggregate`` parameter indicates how results are
aggregated if multiple relations have the same destination node types.

Call forward with different inputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Case 1: Call forward with some ``'user'`` features. This computes new
features for both ``'user'`` and ``'game'`` nodes.

.. code::

    >>> import torch as th
    >>> h1 = {'user' : th.randn((g.number_of_nodes('user'), 5))}
    >>> h2 = conv(g, h1)
    >>> print(h2.keys())
    dict_keys(['user', 'game'])

Case 2: Call forward with both ``'user'`` and ``'store'`` features.

.. code::

    >>> f1 = {'user' : ..., 'store' : ...}
    >>> f2 = conv(g, f1)
    >>> print(f2.keys())
    dict_keys(['user', 'game'])

Because both the ``'plays'`` and ``'sells'`` relations will update the
``'game'`` features, their results are aggregated by the specified
method (i.e., summation here).

Case 3: Call forward with a pair of inputs.

.. code::

    >>> x_src = {'user' : ..., 'store' : ...}
    >>> x_dst = {'user' : ..., 'game' : ...}
    >>> y_dst = conv(g, (x_src, x_dst))
    >>> print(y_dst.keys())
    dict_keys(['user', 'game'])

Each submodule will also be invoked with a pair of inputs.
