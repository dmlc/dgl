import networkx as nx
from networkx.classes.graph import Graph

# TODO: make representation numpy/tensor from pytorch
# TODO: make message/update functions pytorch functions
# TODO: loss functions and training

class mx_Graph(Graph):
    def __init__(self, *args, **kargs):
        super(mx_Graph, self).__init__(*args, **kargs)
        self.set_msg_func()
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
        '''
        if name == None:
            self.add_node(u, h=h_u)
        else:
            self.add_node(u, name=h_u)
        '''

    def get_repr(self, u, name='h'):
        assert u in self.nodes
        return self.nodes[u][name]

    def set_msg_func(self, func=None, u=None):
        """Function that gathers messages from neighbors"""
        def _default_msg_func(u):
            assert u in self.nodes
            msg_gathered = 0
            for v in self.adj[u]:
                x = self.get_repr(v)
                if x is not None:
                    msg_gathered += x
            return msg_gathered

        # TODO: per node message function
        # TODO: 'sum' should be a separate function
        if func == None:
            self.msg_func = _default_msg_func
        else:
            self.msg_func = func

    def set_update_func(self, func=None, u=None):
        """
        Update function upon receiving an aggregate
        message from a node's neighbor
        """
        def _default_update_func(u, m):
            h_new = self.nodes[u]['h'] + m
            self.set_repr(u, h_new)

        # TODO: per node update function
        if func == None:
            self.update_func = _default_update_func
        else:
            self.update_func = func

    def set_readout_func(self, func=None):
        """Readout function of the whole graph"""
        def _default_readout_func():
            readout = 0
            for n in self.nodes:
                readout += self.nodes[n]['h']
            return readout

        if func == None:
            self.readout_func = _default_readout_func
        else:
            self.readout_func = func

    def readout(self):
        return self.readout_func()

    def update_to(self, u):
        """Pull messages from 1-step away neighbors of u"""
        assert u in self.nodes
        m = self.msg_func(u=u)
        self.update_func(u, m)

    def update_from(self, u):
        """Update u's 1-step away neighbors"""
        assert u in self.nodes
        # TODO: this asks v to pull from nodes other than
        # TODO: u, is this a good thing?
        for v in self.adj[u]:
            self.update_to(v)

    def print_all(self):
        for n in self.nodes:
            print(n, self.nodes[n])
        print()

if __name__ == '__main__':
    tg = nx.path_graph(10)
    g = mx_Graph(tg)
    g.print_all()

    tr = nx.balanced_tree(2, 3)
    m_tr = mx_Graph(tr)
    m_tr.print_all()

    g = mx_Graph(nx.path_graph(3))

    for n in g:
        g.set_repr(n, int(n) + 10)
    g.print_all()
    print(g.readout())

    print("before update:\t", g.nodes[0])
    g.update_to(0)
    print('after update:\t', g.nodes[0])
    g.print_all()

    print(g.readout())

