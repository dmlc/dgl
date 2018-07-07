from collections import defaultdict
from itertools import starmap
from functools import partial

import dgl.graph as G
import dgl.backend as F
import dgl.utils as utils

# TODO(gaiyu): the case of one msg
# TODO(gaiyu): update_func not called for nodes with degree 0

class DGLBGraph(G.DGLGraph):
    def __init__(self, graph_data=None, **attr):
        super(DGLBGraph, self).__init__(graph_data, **attr)

    def sendto(self, u, v):
        """Trigger the message function on edge u->v

        Parameters
        ----------
        u : node, container or tensor
          The source node(s).
        v : node, container or tensor
          The destination node(s).
        """
        fmsg_set, u_list, v_list, edge_list = set(), [], [], []
        # TODO(gaiyu): Group f_msg.
        for uu, vv in utils.edge_iter(u, v):
            f_msg = self.edges[uu, vv].get(G.__MFUNC__, self.m_func)
            assert f_msg is not None, \
                "message function not registered for edge (%s->%s)" % (uu, vv)
            fmsg_set.add(f_msg)
            u_list.append(self.nodes[uu])
            v_list.append(self.nodes[vv])
            edge_list.append(self.edges[uu, vv])

        if len(fmsg_set) > 1:
            raise NotImplementedError()

        u_dict, v_dict = utils.batch(u_list), utils.batch(v_list)
        edge_dict = utils.batch(edge_list)
        msg = f_msg(u_dict, v_dict, edge_dict)

        for i, (uu, vv) in enumerate(utils.edge_iter(u, v)):
            self.edges[uu, vv][G.__MSG__] = {k : v[i : i + 1] for k, v in msg.items()}

    def update_edge(self, u, v):
        """Update representation on edge u->v

        Parameters
        ----------
        u : node, container or tensor
          The source node(s).
        v : node, container or tensor
          The destination node(s).
        """
        # TODO(minjie): tensorize the loop.
        fedge_set, u_list, v_list = set(), [], []
        for uu, vv in utils.edge_iter(u, v):
            f_edge = self.edges[uu, vv].get(G.__EFUNC__, self.m_func)
            assert f_edge is not None, \
                "edge function not registered for edge (%s->%s)" % (uu, vv)
            fedge_set.add(f_edge)
            u_list.append(self.nodes[uu])
            v_list.append(self.nodes[vv])

        u_dict, v_dict = utils.batch(u_list), utils.batch(v_list)
        new = f_edge(self.nodes[uu], self.nodes[vv], self.edges[uu, vv])

        for i, (uu, vv) in utils.edge_iter(u, v):
            self.edges[uu, vv][G.__E_REPR__] = {k : v[i : i + 1] for k, v in new.items()}

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
        u_is_container = isinstance(u, list)
        u_is_tensor = isinstance(u, F.Tensor)

        repr_list, msg_list = [], []
        freduce_set, fupdate_set = set(), set()
        for i, uu in enumerate(utils.node_iter(u)):
            if preds is None:
                v = list(self.pred[uu])
            elif u_is_container or u_is_tensor:
                v = preds[i]
            else:
                v = preds

            f_reduce = self.nodes[uu].get(G.__RFUNC__, self.r_func)
            assert f_reduce is not None, \
                "Reduce function not registered for node %s" % uu
            freduce_set.add(f_reduce)

            f_update = self.nodes[uu].get(G.__UFUNC__, self.u_func)
            assert f_update is not None, \
                "Update function not registered for node %s" % uu
            fupdate_set.add(f_update)

            repr_list.append(self.nodes[uu])
            msg_list.append([self.edges[vv, uu][G.__MSG__] for vv in v])

        def groupby(iterable, key):
            d = defaultdict(list)
            for x in iterable:
                d[key(x)].append(x)
            return [d[key] for key in sorted(d.keys())]

        if len(freduce_set) > 1:
            raise NotImplementedError() # TODO(gaiyu): group f_reduce
        f_reduce, = list(freduce_set)
        deg_list = list(map(len, msg_list))
        min_deg, max_deg = min(deg_list), max(deg_list)
        msgs = msg_list if min_deg > 0 else filter(bool, msg_list)
        msgs = groupby(msgs, len) # list of list of list of dict
        msgs = [list(map(utils.batch, x)) for x in msgs] # list of list of dict
        msgs = map(partial(utils.batch, method='stack'), msgs) # iter (map) of dict
        msgs = list(map(f_reduce, msgs)) # list of dict

        if len(fupdate_set) > 1:
            raise NotImplementedError() # TODO(gaiyu): group f_update
        f_update, = list(fupdate_set)

        ureprmsg_zip = zip(utils.node_iter(u), repr_list, deg_list)
        by_deg = groupby(ureprmsg_zip, lambda x: x[2]) # list of list of tuple

        def update(u_tuple, repr_dict):
            for i, uu in enumerate(u_tuple):
                self.nodes[uu].update({k : v[i : i + 1] for k, v in repr_dict.items()})

        # nodes without neighbors
        if min_deg == 0:
            without_msg = by_deg[0] # list of (u, repr)
            u_tuple, repr_tuple, _ = zip(*without_msg) # tuple of dict
            repr_dict = utils.batch(repr_tuple)
            repr_dict = f_update(repr_dict, None)
            update(u_tuple, repr_dict)

        # nodes with neighbors
        if max_deg > 0:
            by_deg = by_deg if min_deg > 0 else by_deg[1:]
            with_msg = sum(by_deg, []) # list of (u, repr)
            u_tuple, repr_tuple, _ = zip(*with_msg) # tuple of dict
            repr_dict = utils.batch(repr_tuple)
            msgs = utils.batch(msgs)
            repr_dict = f_update(repr_dict, msgs)
            update(u_tuple, repr_dict)
