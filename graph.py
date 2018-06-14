import networkx as nx
from networkx.classes.digraph import DiGraph

class dgl_Graph(DiGraph):
    '''
    Functions:
        - m_func: per edge (u, v), default is u['state']
        - u_func: per node u, default is RNN(m, u['state'])
    '''
    def __init__(self, *args, **kargs):
        super(dgl_Graph, self).__init__(*args, **kargs)
        self.m_func = DefaultMessageModule()
        self.u_func = DefaultUpdateModule()
        self.readout_func = DefaultReadoutModule()
        self.init_reprs()

    def init_reprs(self, h_init=None):
        for n in self.nodes:
            self.set_repr(n, h_init)

    def set_repr(self, u, h_u, name='state'):
        assert u in self.nodes
        kwarg = {name: h_u}
        self.add_node(u, **kwarg)

    def get_repr(self, u, name='state'):
        assert u in self.nodes
        return self.nodes[u][name]

    def _nodes_or_all(self, nodes='all'):
        return self.nodes() if nodes == 'all' else nodes

    def _edges_or_all(self, edges='all'):
        return self.edges() if edges == 'all' else edges

    def register_message_func(self, message_func, edges='all', batched=False):
        if edges == 'all':
            self.m_func = message_func
        else:
            for e in self.edges:
                self.edges[e]['m_func'] = message_func

    def register_update_func(self, update_func, nodes='all', batched=False):
        if nodes == 'all':
            self.u_func = update_func
        else:
            for n in nodes:
                self.node[n]['u_func'] = update_func

    def register_readout_func(self, readout_func):
        self.readout_func = readout_func

    def readout(self, nodes='all', **kwargs):
        nodes_state = []
        nodes = self._nodes_or_all(nodes)
        for n in nodes:
            nodes_state.append(self.get_repr(n))
        return self.readout_func(nodes_state, **kwargs)

    def sendto(self, u, v):
        """Compute message on edge u->v
        Args:
            u: source node
            v: destination node
        """
        f_msg = self.edges[(u, v)].get('m_func', self.m_func)
        m = f_msg(self.get_repr(u))
        self.edges[(u, v)]['msg'] = m

    def sendto_ebunch(self, ebunch):
        """Compute message on edge u->v
        Args:
            ebunch: a bunch of edges
        """
        #TODO: simplify the logics
        for u, v in ebunch:
            f_msg = self.edges[(u, v)].get('m_func', self.m_func)
            m = f_msg(self.get_repr(u))
            self.edges[(u, v)]['msg'] = m

    def recvfrom(self, u, nodes):
        """Update u by nodes
        Args:
            u: node to be updated
            nodes: nodes with pre-computed messages to u
        """
        m = [self.edges[(v, u)]['msg'] for v in nodes]
        f_update = self.nodes[u].get('u_func', self.u_func)
        x_new = f_update(self.get_repr(u), m)
        self.set_repr(u, x_new)

    def update_by_edge(self, e):
        u, v = e
        self.sendto(u, v)
        self.recvfrom(v, [u])

    def update_to(self, u):
        """Pull messages from 1-step away neighbors of u"""
        assert u in self.nodes

        for v in self.pred[u]:
            self.sendto(v, u)
        self.recvfrom(u, list(self.pred[u]))

    def update_from(self, u):
        """Update u's 1-step away neighbors"""
        assert u in self.nodes
        for v in self.succ[u]:
            self.update_to(v)

    def update_all_step(self):
        self.sendto_ebunch(self.edges)
        for u in self.nodes:
            self.recvfrom(u, list(self.pred[u]))

    def draw(self):
        from networkx.drawing.nx_agraph import graphviz_layout

        pos = graphviz_layout(self, prog='dot')
        nx.draw(self, pos, with_labels=True)

    def print_all(self):
        for n in self.nodes:
            print(n, self.nodes[n])
        print()
