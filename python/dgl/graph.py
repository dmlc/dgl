"""Base graph class specialized for neural networks on graphs.
"""

from collections import defaultdict
import networkx as nx
from networkx.classes.digraph import DiGraph

import dgl.backend as F
from dgl.backend import Tensor
import dgl.utils as utils

__MSG__ = "__msg__"
__E_REPR__ = "__e_repr__"
__N_REPR__ = "__n_repr__"
__MFUNC__ = "__mfunc__"
__EFUNC__ = "__efunc__"
__UFUNC__ = "__ufunc__"
__RFUNC__ = "__rfunc__"
__MBATCH__ = "__mbatch__"
__EBATCH__ = "__ebatch__"
__UBATCH__ = "__ubatch__"
__RBATCH__ = "__rbatch__"

class DGLGraph(DiGraph):
    """Base graph class specialized for neural networks on graphs.

    TODO(minjie): document of multi-node and multi-edge syntax.

    Parameters
    ----------
    data : graph data
        Data to initialize graph. Same as networkx's semantics.
    attr : keyword arguments, optional
        Attributes to add to graph as key=value pairs.
    """
    def __init__(self, graph_data=None, **attr):
        super(DGLGraph, self).__init__(graph_data, **attr)
        self.m_func = None
        self.e_func = None
        self.r_func = None
        self.u_func = None
        self.readout_func = None

        self.m_batch = None
        self.e_batch = None
        self.r_batch = None
        self.u_batch = None

    def init_node_repr(self, h_init=None, name=__N_REPR__, batch=True, expand_dims=False):
        """ Initialize node representations.

        Parameters
        ----------
        h_init : list, tensor or any object
            Initial node representations.
        name : str, optional
            Representation name.
        batch : bool, optional
            Whether `h_u` is a batch (list or tensor) of representations.
            If `h_u` is a list and `batch` is `True`, `len(h_u)` must be the same as
            the number of nodes in graph.
            If `h_u` is a tensor and `batch` is `True`, the shape of `h_u` must be
            `(N, D_1, ...)`, where `N` is the number of nodes in graph. The representation 
            for the `i`-th node is `h_u[i]` (a tensor with shape `(1, D_1, ...)`).
            If `h_u` is any other object, `batch` must be `False`, 
            in which case nodes in graph will share `h_u`.
        expand_dims : bool, optional
            Whether to prepend a dimension (the batch dimension) to `h_u`.
            Only valid when `h_u` is a tensor and `batch` is `False`.

        Examples
        --------

        Initialize every node's representation to `0`.

        >>> g.init_node_repr([0] * len(g.nodes))

        Initialize every node's representation to `torch.zeros(1, 1)`.

        >>> g.init_node_repr(torch.zeros(len(g.nodes), 1))
        """
        self.set_node_repr(list(self.nodes), h_init, name, batch, expand_dims)

    def init_edge_repr(self, h_init=None, name=__E_REPR__, batch=True, expand_dims=False):
        """ Initialize edge representations.

        Parameters
        ----------
        h_init : list, tensor or any object
            Initial edge representations.
        name : str, optional
            Representation name.
        batch : bool, optional
            Whether `h_u` is a batch (list or tensor) of representations.
            If `h_u` is a list and `batch` is `True`, `len(h_u)` must be the same as
            the number of edges in graph.
            If `h_u` is a tensor and `batch` is `True`, the shape of `h_u` must be
            `(N, D_1, ...)`, where `N` is the number of edges in graph. The representation 
            for the `i`-th edge is `h_u[i]` (a tensor with shape `(1, D_1, ...)`).
            If `h_u` is any other object, `batch` must be `False`, 
            in which case edges in graph will share `h_u`.
        expand_dims : bool, optional
            Whether to prepend a dimension (the batch dimension) to `h_u`.
            Only valid when `h_u` is a tensor and `batch` is `False`.

        Examples
        --------

        Initialize every edge's representation to `0`.

        >>> g.init_edge_repr([0] * len(g.edges))

        Initialize every edge's representation to `torch.zeros(1, 1)`.

        >>> g.init_edge_repr(torch.zeros(len(g.edges), 1))
        """
        u, v = map(list, zip(*self.edges))
        self.set_edge_repr(u, v, h_init, name, batch, expand_dims)

    def set_node_repr(self, u, h_u, name=__N_REPR__, batch=True, expand_dims=False):
        """ Set node representations.

        Parameters
        ----------
        u : node, list or tensor
            (Container of) node(s) to set representations to.
        h_u : list, tensor or any object
            Node representations.
        name : str, optional
            Representation name.
        batch : bool, optional
            Whether `h_u` is a batch (list or tensor) of representations.
            If `h_u` is a list and `batch` is `True`, `len(h_u)` must be the same as
            the number of nodes in `u`. The representation for the `i`th node is `h_u[i]`. 
            If `h_u` is a tensor and `batch` is `True`, the shape of `h_u` must be
            `(N, D_1, ...)`, where `N` is the number of nodes in `u`. The representation 
            for the `i`th node is `h_u[i]` (a tensor with shape `(1, D_1, ...)`).
            If `h_u` is any other object, `batch` must be `False`, 
            in which case nodes in `u` will share `h_u`.
        expand_dims : bool, optional
            Whether to prepend a dimension (the batch dimension) to `h_u`.
            Only valid when `h_u` is a tensor and `batch` is `False`.

        Examples
        --------

        Set every node's representation to `0`.

        >>> g.set_node_repr(list(g.nodes), [0] * len(g.nodes))

        Set every node's representation to `torch.zeros(1, 1)`.

        >>> g.set_node_repr(list(g.nodes), torch.zeros(len(g.nodes), 1))
        """

        if not isinstance(u, (list, Tensor)):
            print("[DEPRECATED]: please directly set node attrs "
                  "(e.g. g.nodes[node]['x'] = val).")
        node_iter = lambda: utils.node_iter(u)
        self.set_x_repr(self.nodes, node_iter, h_u, name, batch, expand_dims)

    def set_edge_repr(self, u, v, h_uv, name=__E_REPR__, batch=True, expand_dims=False):
        """ Set edge representations.

        Parameters
        ----------
        u, v : edge, list or tensor
            (Container of) edge(s) to set representations to.
        h_uv : list, tensor or any object
            Node representations.
        name : str, optional
            Representation name.
        batch : bool, optional
            Whether `h_uv` is a batch (list or tensor) of representations.
            If `h_uv` is a list and `batch` is `True`, `len(h_uv)` must be the same as
            the number of edges. The representation for the `i`th edge is `h_uv[i]`. 
            If `h_uv` is a tensor and `batch` is `True`, the shape of `h_uv` must be
            `(N, D_1, ...)`, where `N` is the number of edges. The representation 
            for the `i`th edge is `h_uv[i]` (a tensor with shape `(1, D_1, ...)`).
            If `h_uv` is any other object, `batch` must be `False`, 
            in which case edges will share `h_uv`.
        expand_dims : bool, optional
            Whether to prepend a dimension (the batch dimension) to `h_uv`.
            Only valid when `h_uv` is a tensor and `batch` is `False`.

        Examples
        --------

        Set every edge's representation to `0`.

        >>> g.set_edge_repr(list(g.edges), [0] * len(g.edges))

        Set every edge's representation to `torch.zeros(1, 1)`.

        >>> g.set_edge_repr(list(g.edges), torch.zeros(len(g.edges), 1))
        """
        if not isinstance(u, (list, Tensor)) and not isinstance(v, (list, Tensor)):
            print("[DEPRECATED]: please directly set edge attrs "
                  "(e.g. g.edges[u, v]['x'] = val).")
        edge_iter = lambda: utils.edge_iter(u, v)
        self.set_x_repr(self.edges, edge_iter, h_uv, name, batch, expand_dims)

    def set_x_repr(self, xs, x_iter, h_x, name, batch, expand_dims):
        x_str = 'node' if xs == self.nodes else 'edge'
        hx_str = 'h_u' if xs == self.nodes else 'h_uv'
        add_x = self.add_node if xs == self.nodes else \
                    lambda uv, **kwargs: self.add_edge(uv[0], uv[1], **kwargs)

        if batch:
            assert isinstance(h_x, (list, Tensor)), \
                "%s must be a list or tensor when batch is True" % hx_str

            n = len(list(x_iter()))
            if isinstance(h_x, list):
                assert len(h_x) == n, \
                    "len(%s) must be the same as the number of %ss." % (hx_str, x_str)
            elif isinstance(h_x, Tensor):
                assert F.shape(h_x)[0] == n, "The first dimension of %s " \
                    "must be the same as the number of %ss" % (hx_str, x_str)

            assert not expand_dims, "expand_dims may be True only when batch is False."

            for xx, h_xx in zip(x_iter(), h_x):
                assert xx in xs, "%s does not exist in graph." % (xx,)
                kwarg = {name: h_xx}
                add_x(xx, **kwarg)

        else:
            if expand_dims:
                assert isinstance(x, Tensor), \
                    "expand_dims may be True only when %s is a tensor." % hx_str
                x = F.expand_dims(x, 0)

            for xx in x_iter():
                assert xx in xs, "%s does not exist in graph." % (xx,)
                kwarg = {name: h_x}
                add_x(xx, **kwarg)

    def get_node_repr(self, u, name=__N_REPR__, batch=True):
        """ Get node representations.

        Parameters
        ----------
        u : node, list or tensor.
            (Container of) node(s) to get representations from.
        name : str, optional
            Representation name.
        batch : bool, optional
            Whether to batch representations.
            `batch` may be `True` only when representations are batchable.

        Returns
        -------
        repr : a list or tensor of representations.
        """
        if not isinstance(u, (list, Tensor)):
            print("[DEPRECATED]: please directly get node attrs "
                  "(e.g. g.nodes[node]['x']).")
        node_iter = lambda: utils.node_iter(u)
        return self.get_x_repr(self.nodes, node_iter, name, batch)

    def get_edge_repr(self, u, v, name=__E_REPR__, batch=True):
        """ Get edge representations.

        Parameters
        ----------
        u, v : edge, list or tensor.
            (Container of) edge(s) to get representations from.
        name : str, optional
            Representation name.
        batch : bool, optional
            Whether to batch representations.
            `batch` may be `True` only when representations are batchable.

        Returns
        -------
        repr : a list or tensor of representations.
        """
        if not isinstance(u, (list, Tensor)) and not isinstance(v, (list, Tensor)):
            print("[DEPRECATED]: please directly get edge attrs "
                  "(e.g. g.edges[u, v]['x']).")
        edge_iter = lambda: utils.edge_iter(u, v)
        return self.get_x_repr(self.edges, edge_iter, name, batch)

    def get_x_repr(self, xs, x_iter, name, batch):
        repr_list = []
        for xx in x_iter():
            assert xx in xs, "%s does not exist in graph." % (xx,)
            assert name in xs[xx], \
                "Representation %s does not exist for %s." % (name, xx)
            repr_list.append(xs[xx][name])

        if batch:
            assert F.isbatchable(repr_list, F.cat), "Representations are not batchable."
            return F.cat(repr_list)
        else:
            return repr_list

    def register_message_func(self, message_func, edges='all', batchable=False):
        """Register computation on edges.

        The message function should be compatible with following signature:

        (node_reprs, node_reprs, edge_reprs) -> edge_reprs

        It computes the representation of a message
        using the representations of the source node, target node and the edge
        itself. All node_reprs and edge_reprs are dictionaries.

        Parameters
        ----------
        message_func : callable
          Message function on the edge.
        edges : str, pair of nodes, pair of containers, pair of tensors
          The edges for which the message function is registered. Default is
          registering for all the edges. Registering for multiple edges is
          supported.
        batchable : bool
          Whether the provided message function allows batch computing.

        Examples
        --------

        Register for all edges.
        >>> g.register_message_func(mfunc)

        Register for a specific edge.
        >>> g.register_message_func(mfunc, (u, v))

        Register for multiple edges.
        >>> u = [u1, u2, u3, ...]
        >>> v = [v1, v2, v3, ...]
        >>> g.register_message_func(mfunc, (u, v))
        """
        if edges == 'all':
            self.m_func = message_func
            self.m_batch = batchable
        else:
            for e in edges:
                self.edges[e][__MFUNC__] = message_func
                self.edges[e][__MBATCH__] = batchable

    def register_edge_func(self, edge_func, edges='all', batchable=False):
        """Register computation on edges.

        The edge function should be compatible with following signature:

        (node_reprs, node_reprs, edge_reprs) -> edge_reprs

        It computes the new edge representations (the same concept as messages)
        using the representations of the source node, target node and the edge
        itself. All node_reprs and edge_reprs are dictionaries.

        Parameters
        ----------
        edge_func : callable
          Message function on the edge.
        edges : str, pair of nodes, pair of containers, pair of tensors
          The edges for which the message function is registered. Default is
          registering for all the edges. Registering for multiple edges is
          supported.
        batchable : bool
          Whether the provided message function allows batch computing.

        Examples
        --------

        Register for all edges.
        >>> g.register_edge_func(efunc)

        Register for a specific edge.
        >>> g.register_edge_func(efunc, (u, v))

        Register for multiple edges.
        >>> u = [u1, u2, u3, ...]
        >>> v = [v1, v2, v3, ...]
        >>> g.register_edge_func(mfunc, (u, v))
        """
        if edges == 'all':
            self.e_func = edge_func
            self.e_batch = batchable
        else:
            for e in edges:
                self.edges[e][__EFUNC__] = edge_func
                self.edges[e][__EBATCH__] = batchable

    def register_reduce_func(self, reduce_func, nodes='all', batchable=False):
        """Register message reduce function on incoming edges.

        The reduce function should be compatible with following signature:

        edge_reprs -> reduced_edge_repr

        It computes the reduced edge representations using the representations
        of the in-coming edges (the same concept as messages).

        The reduce function can be any of the pre-defined functions ('sum',
        'max'). If built-in function is used, computation will be performed
        efficiently (using generic-SPMV kernels).

        Parameters
        ----------
        reduce_func : str or callable
          Reduce function on incoming edges.
        nodes : str, node, container or tensor
          The nodes for which the reduce function is registered. Default is
          registering for all the nodes. Registering for multiple nodes is
          supported.
        batchable : bool
          Whether the provided reduce function allows batch computing.

        Examples
        --------

        Register for all nodes.
        >>> g.register_reduce_func(rfunc)

        Register for a specific node.
        >>> g.register_reduce_func(rfunc, u) # TODO Not implemented

        Register for multiple nodes.
        >>> u = [u1, u2, u3, ...]
        >>> g.register_reduce_func(rfunc, u)
        """
        if isinstance(reduce_func, str):
            # built-in reduce func
            if reduce_func == 'sum':
                reduce_func = F.reduce_sum
            elif reduce_func == 'max':
                reduce_func = F.reduce_max
            else:
                raise NotImplementedError(
                        "Built-in function %s not implemented" % reduce_func)
        if nodes == 'all':
            self.r_func = reduce_func
            self.r_batch = batchable
        else:
            for n in nodes:
                self.nodes[n][__RFUNC__] = reduce_func
                self.nodes[n][__RBATCH__] = batchable

    def register_update_func(self, update_func, nodes='all', batchable=False):
        """Register computation on nodes.

        The update function should be compatible with following signature:

        (node_reprs, reduced_edge_repr) -> node_reprs

        It computes the new node representations using the representations
        of the in-coming edges (the same concept as messages) and the node
        itself. All node_reprs and edge_reprs are dictionaries.

        Parameters
        ----------
        update_func : callable
          Update function on the node.
        nodes : str, node, container or tensor
          The nodes for which the update function is registered. Default is
          registering for all the nodes. Registering for multiple nodes is
          supported.
        batchable : bool
          Whether the provided update function allows batch computing.

        Examples
        --------

        Register for all nodes.
        >>> g.register_update_func(ufunc)

        Register for a specific node.
        >>> g.register_update_func(ufunc, u) # TODO Not implemented

        Register for multiple nodes.
        >>> u = [u1, u2, u3, ...]
        >>> g.register_update_func(ufunc, u)
        """
        if nodes == 'all':
            self.u_func = update_func
            self.u_batch = batchable
        else:
            for n in nodes:
                self.nodes[n][__UFUNC__] = update_func
                self.nodes[n][__UBATCH__] = batchable

    def register_readout_func(self, readout_func):
        """Register computation on the whole graph.

        The readout_func should be compatible with following signature:

        (node_reprs, edge_reprs) -> any

        It takes the representations of selected nodes and edges and
        returns readout values.

        NOTE: readout function can be implemented outside of DGLGraph.
        One can simple get the node/edge reprs of the graph and perform
        arbitrary computation.

        Parameters
        ----------
        readout_func : callable
          The readout function.

        See Also
        --------
        readout
        """
        self.readout_func = readout_func

    def readout(self, nodes='all', edges='all', **kwargs):
        """Trigger the readout function on the specified nodes/edges.

        Parameters
        ----------
        nodes : str, node, container or tensor
          The nodes to get reprs from.
        edges : str, pair of nodes, pair of containers or pair of tensors
          The edges to get reprs from.
        kwargs : keyword arguments, optional
            Arguments for the readout function.
        """
        nodes = self._nodes_or_all(nodes)
        edges = self._edges_or_all(edges)
        assert self.readout_func is not None, \
            "Readout function is not registered."
        # TODO(minjie): tensorize following loop.
        nstates = [self.nodes[n] for n in nodes]
        estates = [self.edges[e] for e in edges]
        return self.readout_func(nstates, estates, **kwargs)

    def sendto(self, u, v):
        """Trigger the message function on edge u->v

        Parameters
        ----------
        u : node, container or tensor
          The source node(s).
        v : node, container or tensor
          The destination node(s).
        """

        '''
        # TODO(minjie): tensorize the loop.
        for uu, vv in utils.edge_iter(u, v):
            f_msg = self.edges[uu, vv].get(__MFUNC__, self.m_func)
            assert f_msg is not None, \
                "message function not registered for edge (%s->%s)" % (uu, vv)
            m = f_msg(self.nodes[uu], self.nodes[vv], self.edges[uu, vv])
            self.edges[uu, vv][__MSG__] = m
        '''

        mfunc2args = defaultdict(list)
        mfunc2edges = defaultdict(list)
        for uu, vv in utils.edge_iter(u, v):
            m_func = self.edges[uu, vv].get(__MFUNC__, self.m_func)
            m_batch = self.edges[uu, vv].get(__MBATCH__, self.m_batch)
            assert m_func is not None, \
                "message function not registered for edge (%s->%s)" % (uu, vv)

            args = [self.nodes[uu], self.nodes[vv], self.edges[uu, vv]]
            if m_batch:
                mfunc2args[m_func].append(args)
                mfunc2edges[m_func].append([uu, vv])
            else:
                msg = m_func(*args)
                self.edges[uu, vv][__MSG__] = msg

        for m_func, args_list in mfunc2args.items():
            uu_tuple, vv_tuple, uv_tuple = zip(*args_tuple)
            uu_dict = utils.batch(uu_tuple)
            vv_dict = utils.batch(vv_tuple)
            uv_dict = utils.batch(uv_tuple)
            msg_dict = m_func(uu_dict, vv_dict, uv_dict)

            for i, [uu, vv] in enumerate(mfunc2edges[m_func]):
                msg = {k : v[i : i + 1] for k, v in msg_dict.items()}
                self.edges[uu, vv][G.__MSG__].update(msg)

    def update_edge(self, u, v):
        """Update representation on edge u->v

        Parameters
        ----------
        u : node, container or tensor
          The source node(s).
        v : node, container or tensor
          The destination node(s).
        """

        '''
        # TODO(minjie): tensorize the loop.
        for uu, vv in utils.edge_iter(u, v):
            f_edge = self.edges[uu, vv].get(__EFUNC__, self.m_func)
            assert f_edge is not None, \
                "edge function not registered for edge (%s->%s)" % (uu, vv)
            m = f_edge(self.nodes[uu], self.nodes[vv], self.edges[uu, vv])
            self.edges[uu, vv][__E_REPR__] = m
        '''

        efunc2args = defaultdict(list)
        efunc2edges = defaultdict(list)
        for uu, vv in utils.edge_iter(u, v):
            e_func = self.edges[uu, vv].get(__EFUNC__, self.e_func)
            e_batch = self.edges[uu, vv].get(__EBATCH__, self.e_batch)
            assert e_func is not None, \
                "edge function not registered for edge (%s->%s)" % (uu, vv)

            args = [self.nodes[uu], self.nodes[vv], self.edges[uu, vv]]
            if e_batch:
                efunc2args[e_func].append(args)
                efunc2edges[e_func].append([uu, vv])
            else:
                msg = e_func(*args)
                self.edges[uu, vv][__MSG__] = msg

        for e_func, args_list in efunc2args.items():
            uu_tuple, vv_tuple, uv_tuple = zip(*args_tuple)
            uu_dict = utils.batch(uu_tuple)
            vv_dict = utils.batch(vv_tuple)
            uv_dict = utils.batch(uv_tuple)
            ret_dict = e_func(uu_dict, vv_dict, uv_dict)

            for i, [uu, vv] in enumerate(efunc2edges[e_func]):
                ret = {k : v[i : i + 1] for k, v in ret_dict.items()}
                self.edges[uu, vv][G.__MSG__].update(ret)

    def recvfrom(self, u, preds=None):
        """Trigger the update function on node u.

        It computes the new node state using the messages and edge
        states from preds->u. If `u` is one node, `preds` is a list
        of predecessors. If `u` is a container or tensor of nodes,
        then `preds[i]` should be the predecessors of `u[i]`.

        Parameters
        ----------
        u : node, container or tensor
          The node to be updated.
        preds : container
          Nodes with pre-computed messages to u. Default is all
          the predecessors.
        """

        '''
        u_is_container = isinstance(u, list)
        u_is_tensor = isinstance(u, Tensor)
        # TODO(minjie): tensorize the loop.
        for i, uu in enumerate(utils.node_iter(u)):
            if preds is None:
                v = list(self.pred[uu])
            elif u_is_container or u_is_tensor:
                v = preds[i]
            else:
                v = preds
            # TODO(minjie): tensorize the message batching
            m = [self.edges[vv, uu][__MSG__] for vv in v]
            f_reduce = self.nodes[uu].get(__RFUNC__, self.r_func)
            assert f_reduce is not None, \
                "Reduce function not registered for node %s" % uu
            msgs_reduced_repr = f_reduce(m)
            f_update = self.nodes[uu].get(__UFUNC__, self.u_func)
            assert f_update is not None, \
                "Update function not registered for node %s" % uu
            self.node[uu].update(f_update(self.nodes[uu], msgs_reduced_repr))
        '''

        # Two loops for r_func and u_func for correctness.

        u_is_container = isinstance(u, list)
        u_is_tensor = isinstance(u, Tensor)
        rfunc2msgs = defaultdict(list)
        rfunc2nodes = defaultdict(list)
        rmsg_dict = {}
        for i, uu in enumerate(utils.node_iter(u)):
            if preds is None:
                v = list(self.pred[uu])
            elif u_is_container or u_is_tensor:
                v = preds[i]
            else:
                v = preds

            msgs = [self.edges[vv, uu][__MSG__] for vv in v]
            r_func = self.nodes[uu].get(__RFUNC__, self.r_func)
            r_batch = self.edges[uu, vv].get(__RBATCH__, self.r_batch)
            assert r_func is not None, \
                "Reduce function not registered for node %s" % uu
            if r_batch:
                rfunc2msgs[r_func].append(msgs)
                rfunc2nodes[r_func].append(uu)
            else:
                rmsg_dict[uu, vv] = r_func(msgs)

        def groupby(iterable, key):
            d = defaultdict(list)
            for x in iterable:
                d[key(x)].append(x)
            return [d[key] for key in sorted(d.keys())]

        def update(u_tuple, repr_dict):
            for i, uu in enumerate(u_tuple):
                self.nodes[uu].update({k : v[i : i + 1] for k, v in repr_dict.items()})

        for r_func, msgs_list in rfunc2msgs.items():
            deg_list = list(map(len, msgs_list))
            min_deg, max_deg = min(deg_list), max(deg_list)
            msgs = msgs_list if min_deg > 0 else filter(bool, msgs_list)
            msgs = groupby(msgs, len) # list of list of list of dict
            msgs = [list(map(utils.batch, x)) for x in msgs] # list of list of dict
            msgs = map(partial(utils.batch, method='stack'), msgs) # iter (map) of dict
            rmsgs = map(r_func, msgs) # iter (map) of dict

            uumsgs_zip = zip(rfunc2nodes[r_func], msgs_list)
            by_deg = groupby(uumsgs_zip, lambda x: x[1])
            for uu_list, rmsg_dict in zip(by_deg, rmsgs):
                pass # TODO(gaiyu): update

        ufunc2args = defaultdict(list)
        ufunc2nodes = defaultdict(list)
        for i, uu in enumerate(utils.node_iter(u)):
            if preds is None:
                v = list(self.pred[uu])
            elif u_is_container or u_is_tensor:
                v = preds[i]
            else:
                v = preds

            u_func = self.nodes[uu].get(__UFUNC__, self.u_func)
            u_batch = self.edges[uu, vv].get(__UBATCH__, self.u_batch)
            assert u_func is not None, \
                "Update function not registered for node %s" % uu

            args = [self.nodes[uu], rmsg_dict[uu]]
            if u_batch:
                ufunc2args[u_func].append(args)
                ufunc2nodes[u_func].append(uu)
            else:
                self.node[uu].update(u_func(*args))

        for u_func, args_list in ufunc2rmsgs.items():
            uu_tuple, rmsg_tuple = zip(*args_list)
            uu_dict = utils.batch(uu_dict)
            rmsg_dict = utils.batch(rmsg_dict)
            ret_dict = u_func(uu_dict, rmsg_dict)
            update(ufunc2nodes[u_func], ret_dict)

    def update_by_edge(self, u, v):
        """Trigger the message function on u->v and update v.

        Parameters
        ----------
        u : node, container or tensor
          The source node(s).
        v : node, container or tensor
          The destination node(s).
        """
        self.sendto(u, v)
        # TODO(minjie): tensorize the following loops.
        preds = defaultdict(list)
        for uu, vv in utils.edge_iter(u, v):
            preds[vv].append(uu)
        if len(preds) == 1:
            dst = list(preds.keys())[0]
            src = preds[dst]
            self.recvfrom(dst, src)
        elif len(preds) > 1:
            dst = list(preds.keys())
            src = [preds[d] for d in dst]
            self.recvfrom(dst, src)

    def update_to(self, u):
        """Pull messages from the node's predecessors and then update it.

        Parameters
        ----------
        u : node, container or tensor
          The node to be updated.
        """
        # TODO(minjie): tensorize the following code.
        for uu in utils.node_iter(u):
            assert uu in self.nodes
            preds = list(self.pred[uu])
            self.sendto(preds, uu)
            self.recvfrom(uu, preds)

    def update_from(self, u):
        """Send message from the node to its successors and update them.

        Parameters
        ----------
        u : node, container or tensor
          The node that sends out messages.
        """
        # TODO(minjie): tensorize the following code.
        for uu in utils.node_iter(u):
            assert uu in self.nodes
            for v in self.succ[uu]:
                self.update_by_edge(uu, v)

    def update_all(self):
        """Send messages through all the edges and update all nodes.
        """
        # TODO(minjie): tensorize the following code.
        u = [uu for uu, _ in self.edges]
        v = [vv for _, vv in self.edges]
        self.sendto(u, v)
        self.recvfrom(list(self.nodes()))
        # TODO(zz): this is a hack
        if self.e_func:
            self.update_edge(u, v)

    def propagate(self, iterator='bfs', **kwargs):
        """Propagate messages and update nodes using iterator.

        A convenient function for passing messages and updating
        nodes according to the iterator. The iterator can be
        any of the pre-defined iterators ('bfs', 'dfs', 'pre-order',
        'mid-order', 'post-order'). The computation will be unrolled
        in the backend efficiently. User can also provide custom
        iterator that generates the edges and nodes.

        Parameters
        ----------
        iterator : str or generator of steps.
          The iterator of the graph.
        kwargs : keyword arguments, optional
            Arguments for pre-defined iterators.
        """
        if isinstance(iterator, str):
            # TODO Call pre-defined routine to unroll the computation.
            raise RuntimeError('Not implemented.')
        else:
            # NOTE: the iteration can return multiple edges at each step.
            for u, v in iterator:
                self.update_by_edge(u, v)

    def draw(self):
        """Plot the graph using dot."""
        from networkx.drawing.nx_agraph import graphviz_layout

        pos = graphviz_layout(self, prog='dot')
        nx.draw(self, pos, with_labels=True)

    def _nodes_or_all(self, nodes='all'):
        return self.nodes() if nodes == 'all' else nodes

    def _edges_or_all(self, edges='all'):
        return self.edges() if edges == 'all' else edges
