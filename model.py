import torch as T
import torch.nn as NN
import networkx as nx
from graph import DiGraph

class TreeGlimpsedClassifier(NN.Module):
    def __init__(self,
                 n_children=2,
                 n_depth=3,
                 h_dims=128,
                 node_tag_dims=128,
                 edge_tag_dims=128,
                 h_dims=128,
                 n_classes=10,
                 steps=5,
                 ):
        '''
        Basic idea:
        * We detect objects through an undirected graphical model.
        * The graphical model consists of a balanced tree of latent variables h
        * Each h is then connected to a bbox variable b and a class variable y
        * b of the root is fixed to cover the entire canvas
        * All other h, b and y are updated through message passing
        * The loss function should be either (not completed yet)
            * multiset loss, or
            * maximum bipartite matching (like Order Matters paper)
        '''
        NN.Module.__init__(self)
        self.n_children = n_children
        self.n_depth = n_depth
        self.h_dims = h_dims
        self.node_tag_dims = node_tag_dims
        self.edge_tag_dims = edge_tag_dims
        self.h_dims = h_dims
        self.n_classes = n_classes

        # Create graph of latent variables
        G = nx.balanced_tree(self.n_children, self.n_depth)
        nx.relabel_nodes(G,
                         {i: 'h%d' % i for i in range(self.G.nodes())},
                         False
                         )
        h_nodes_list = G.nodes()
        for h in h_nodes_list:
            G.node[h]['type'] = 'h'
        b_nodes_list = ['b%d' % i for i in range(len(h_nodes_list))]
        y_nodes_list = ['y%d' % i for i in range(len(h_nodes_list))]
        hy_edge_list = [(h, y) for h, y in zip(h_nodes_list, y_nodes_list)]
        hb_edge_list = [(h, b) for h, b in zip(h_nodes_list, y_nodes_list)]
        yh_edge_list = [(y, h) for y, h in zip(y_nodes_list, h_nodes_list)]
        bh_edge_list = [(b, h) for b, h in zip(b_nodes_list, h_nodes_list)]

        G.add_nodes_from(b_nodes_list, type='b')
        G.add_nodes_from(y_nodes_list, type='y')
        G.add_edges_from(hy_edge_list)
        G.add_edges_from(hb_edge_list)

        self.G = DiGraph(G)
        hh_edge_list = [(u, v)
                        for u, v in self.G.edges()
                        if self.G.node[u]['type'] == self.G.node[v]['type'] == 'h']

        self.G.init_node_tag_with(node_tag_dims, T.nn.init.uniform_, args=(-.01, .01))
        self.G.init_edge_tag_with(
                edge_tag_dims,
                T.nn.init.uniform_,
                args=(-.01, .01),
                edges=hy_edge_list + hb_edge_list + yh_edge_list, bh_edge_list
                )

        # y -> h.  An attention over embeddings dynamically generated through edge tags
        self.yh_emb = NN.Sequential(
                NN.Linear(edge_tag_dims, h_dims),
                NN.ReLU(),
                NN.Linear(h_dims, n_classes * h_dims),
                )
        self.G.register_message_func(self._y_to_h, edges=yh_edge_list, batched=True)

        # b -> h.  Projects b and edge tag to the same dimension, then concatenates and projects to h
        self.bh_1 = NN.Linear(self.glimpse.att_params, h_dims)
        self.bh_2 = NN.Linear(edge_tag_dims, h_dims)
        self.bh_all = NN.Linear(3 * h_dims, h_dims)
        self.G.register_message_func(self._b_to_h, edges=bh_edge_list, batched=True)

        # h -> h.  Just passes h itself
        self.G.register_message_func(self._h_to_h, edges=hh_edge_list, batched=True)

        # h -> b.  Concatenates h with edge tag and go through MLP.
        # Produces Δb
        self.hb = NN.Linear(hidden_layers + edge_tag_dims, self.glimpse.att_params)
        self.G.register_message_func(self._h_to_b, edges=hb_edge_list, batched=True)

        # h -> y.  Concatenates h with edge tag and go through MLP.
        # Produces Δy
        self.hy = NN.Linear(hidden_layers + edge_tag_dims, self.n_classes)
        self.G.register_message_func(self._h_to_y, edges=hy_edge_list, batched=True)

        # b update: just adds the original b by Δb
        self.G.register_update_func(self._update_b, nodes=b_nodes_list, batched=False)

        # y update: also adds y by Δy
        self.G.register_update_func(self._update_y, nodes=y_nodes_list, batched=False)

        # h update: simply adds h by the average messages and then passes it through ReLU
        self.G.register_update_func(self._update_h, nodes=h_nodes_list, batched=False)

    def _y_to_h(self, source, edge_tag):
        '''
        source: (n_yh_edges, batch_size, 10) logits
        edge_tag: (n_yh_edges, edge_tag_dims)
        '''
        n_yh_edges, batch_size, _ = source.shape

        if not self._yh_emb_cached:
            self._yh_emb_cached = True
            self._yh_emb_w = self.yh_emb(edge_tag)
            self._yh_emb_w = self._yh_emb_w.reshape(
                    n_yh_edges, self.n_classes, self.h_dims)

        w = self._yh_emb_w[:, None]
        w = w.expand(n_yh_edges, batch_size, self.n_classes, self.h_dims)
        source = source[:, :, None, :]
        return (F.softmax(source) @ w).reshape(n_yh_edges, batch_size, self.h_dims)

    def _b_to_h(self, source, edge_tag):
        '''
        source: (n_bh_edges, batch_size, 6) bboxes
        edge_tag: (n_bh_edges, edge_tag_dims)
        '''
        n_bh_edges, batch_size, _ = source.shape
        _, nchan, nrows, ncols = self.x.size()
        source = source.reshape(-1, self.glimpse.att_params)

        m_b = T.relu(self.bh_1(source))
        m_t = T.relu(self.bh_2(edge_tag))
        m_t = m_t[:, None, :].expand(n_bh_edges, batch_size, self.h_dims)
        m_t = m_t.reshape(-1, self.h_dims)

        # glimpse takes batch dimension first, glimpse dimension second.
        # here, the dimension of @source is n_bh_edges (# of glimpses), then
        # batch size, so we transpose them
        g = self.glimpse(self.x, source.transpose(0, 1)).transpose(0, 1)
        g = g.reshape(n_bh_edges * batch_size, nchan, nrows, ncols)
        phi = self.cnn(g).reshape(n_bh_edges * batch_size, -1)

        m = self.bh_all(T.cat([m_b, m_t, phi], 1))
        m = m.reshape(n_bh_edges, batch_size, self.h_dims)

        return m

    def _h_to_h(self, source, edge_tag):
        return source

    def _h_to_b(self, source, edge_tag):
        n_hb_edges, batch_size, _ = source.shape
        edge_tag = edge_tag[:, None]
        edge_tag = edge_tag.expand(n_hb_edges, batch_size, self.edge_tag_dims)
        I = T.cat([source, edge_tag], -1).reshape(n_hb_edges * batch_size, -1)
        b = self.hb(I)
        return db

    def _h_to_y(self, source, edge_tag):
        n_hy_edges, batch_size, _ = source.shape
        edge_tag = edge_tag[:, None]
        edge_tag = edge_tag.expand(n_hb_edges, batch_size, self.edge_tag_dims)
        I = T.cat([source, edge_tag], -1).reshape(n_hb_edges * batch_size, -1)
        y = self.hy(I)
        return dy

    def _update_b(self, b, b_n):
        return b['state'] + list(b_n.values())[0]['state']

    def _update_y(self, y, y_n):
        return y['state'] + list(y_n.values())[0]['state']

    def _update_h(self, h, h_n):
        m = T.stack([e['state'] for e in h_n]).mean(0)
        return T.relu(h + m)

    def forward(self, x):
        batch_size = x.shape[0]

        self.G.zero_node_state(self.h_dims, batch_size, nodes=h_nodes_list)
        full = self.glimpse.full().unsqueeze(0).expand(batch_size, self.glimpse.att_params)
        for v in self.G.nodes():
            if G.node[v]['type'] == 'b':
                # Initialize bbox variables to cover the entire canvas
                self.G.node[v]['state'] = full

        self._yh_emb_cached = False

        for t in range(self.steps):
            self.G.step()
            # We don't change b of the root
            self.G.node['b0']['state'] = full
