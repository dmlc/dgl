import networkx as nx
#from networkx.classes.graph import Graph
from networkx.classes.digraph import DiGraph

import torch as th
#import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable as Var

# TODO: loss functions and training

class mx_Graph(DiGraph):
    def __init__(self, *args, **kargs):
        super(mx_Graph, self).__init__(*args, **kargs)
        self.set_msg_func()
        self.set_gather_func()
        self.set_reduction_func()
        self.set_update_func()
        self.set_readout_func()
        self.init_reprs()

    def init_reprs(self, h_init=None):
        for n in self.nodes:
            self.set_repr(n, h_init)

    def set_repr(self, u, h_u, name='h'):
        assert u in self.nodes
        kwarg = {name: h_u}
        self.add_node(u, **kwarg)

    def get_repr(self, u, name='h'):
        assert u in self.nodes
        return self.nodes[u][name]

    def set_reduction_func(self):
        def _default_reduction_func(x_s):
            out = th.stack(x_s)
            out = th.sum(out, dim=0)
            return out
        self._reduction_func = _default_reduction_func

    def set_gather_func(self, u=None):
        pass

    def set_msg_func(self, func=None, u=None):
        """Function that gathers messages from neighbors"""
        def _default_msg_func(u):
            assert u in self.nodes
            msg_gathered = []
            for v in self.pred[u]:
                x = self.get_repr(v)
                if x is not None:
                    msg_gathered.append(x)
            return self._reduction_func(msg_gathered)

        # TODO: per node message function
        # TODO: 'sum' should be a separate function
        if func == None:
            self._msg_func = _default_msg_func
        else:
            self._msg_func = func

    def set_update_func(self, func=None, u=None):
        """
        Update function upon receiving an aggregate
        message from a node's neighbor
        """
        def _default_update_func(x, m):
            return x + m

        # TODO: per node update function
        if func == None:
            self._update_func = _default_update_func
        else:
            self._update_func = func

    def set_readout_func(self, func=None):
        """Readout function of the whole graph"""
        def _default_readout_func():
            valid_hs = []
            for x in self.nodes:
                h = self.get_repr(x)
                if h is not None:
                    valid_hs.append(h)
            return self._reduction_func(valid_hs)
#
        if func == None:
            self.readout_func = _default_readout_func
        else:
            self.readout_func = func

    def readout(self):
        return self.readout_func()

    def update_to(self, u):
        """Pull messages from 1-step away neighbors of u"""
        assert u in self.nodes
        m = self._msg_func(u=u)
        x = self.get_repr(u)
        # TODO: ugly hack
        if x is None:
            y = self._update_func(m)
        else:
            y = self._update_func(x, m)
        self.set_repr(u, y)

    def update_from(self, u):
        """Update u's 1-step away neighbors"""
        assert u in self.nodes
        # TODO: this asks v to pull from nodes other than
        # TODO: u, is this a good thing?
        for v in self.succ[u]:
            self.update_to(v)

    def print_all(self):
        for n in self.nodes:
            print(n, self.nodes[n])
        print()

if __name__ == '__main__':
    th.random.manual_seed(0)

    ''': this makes a digraph with double edges
    tg = nx.path_graph(10)
    g = mx_Graph(tg)
    g.print_all()

    # this makes a uni-edge tree
    tr = nx.bfs_tree(nx.balanced_tree(2, 3), 0)
    m_tr = mx_Graph(tr)
    m_tr.print_all()
    '''
    print("testing GRU update")
    g = mx_Graph(nx.path_graph(3))
    g.set_update_func(nn.GRUCell(4, 4))
    for n in g:
        g.set_repr(n, Var(th.rand(2, 4)))

    print("\t**before:"); g.print_all()
    g.update_from(0)
    g.update_from(1)
    print("\t**after:"); g.print_all()

    print("\ntesting fwd update")
    g.clear()
    g.add_path([0, 1, 2])
    g.init_reprs()

    fwd_net = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    g.set_update_func(fwd_net)

    g.set_repr(0, Var(th.rand(2, 4)))
    print("\t**before:"); g.print_all()
    g.update_from(0)
    g.update_from(1)
    print("\t**after:"); g.print_all()
