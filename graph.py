import networkx as nx
import torch as T
import torch.nn as NN

class DiGraph(nx.DiGraph, NN.Module):
    '''
    Reserved attributes:
    * state: node state vectors during message passing iterations
        edges does not have "state vectors"; the "state" field is reserved for storing messages
    * tag: node-/edge-specific feature tensors or other data
    '''
    def __init__(self, data=None, **attr):
        NN.Module.__init__(self)
        nx.DiGraph.__init__(self, data=data, **attr)

        self.message_funcs = []
        self.update_funcs = []

    def add_node(self, n, state=None, tag=None, attr_dict=None, **attr):
        nx.DiGraph.add_node(self, n, state=state, tag=None, attr_dict=attr_dict, **attr)

    def add_nodes_from(self, nodes, state=None, tag=None, **attr):
        nx.DiGraph.add_nodes_from(self, nodes, state=state, tag=tag, **attr)

    def add_edge(self, u, v, tag=None, attr_dict=None, **attr):
        nx.DiGraph.add_edge(self, u, v, tag=tag, attr_dict=attr_dict, **attr)

    def add_edges_from(self, ebunch, tag=tag, attr_dict=None, **attr):
        nx.DiGraph.add_edges_from(self, ebunch, tag=tag, attr_dict=attr_dict, **attr)

    def _nodes_or_all(self, nodes='all'):
        return self.nodes() if nodes == 'all' else nodes

    def _edges_or_all(self, edges='all'):
        return self.edges() if edges == 'all' else edges

    def _node_tag_name(self, v):
        return '(%s)' % v

    def _edge_tag_name(self, u, v):
        return '(%s, %s)' % (min(u, v), max(u, v))

    def zero_node_state(self, state_dims, batch_size=None, nodes='all'):
        shape = (
                [batch_size] + list(state_dims)
                if batch_size is not None
                else state_dims
                )
        nodes = self._nodes_or_all(nodes)

        for v in nodes:
            self.node[v]['state'] = T.zeros(shape)

    def init_node_tag_with(self, shape, init_func, dtype=T.float32, nodes='all', args=()):
        nodes = self._nodes_or_all(nodes)

        for v in nodes:
            self.node[v]['tag'] = init_func(NN.Parameter(T.zeros(shape, dtype=dtype)), *args)
            self.register_parameter(self._node_tag_name(v), self.node[v]['tag'])

    def init_edge_tag_with(self, shape, init_func, dtype=T.float32, edges='all', args=()):
        edges = self._edges_or_all(edges)

        for u, v in edges:
            self[u][v]['tag'] = init_func(NN.Parameter(T.zeros(shape, dtype=dtype)), *args)
            self.register_parameter(self._edge_tag_name(u, v), self[u][v]['tag'])

    def remove_node_tag(self, nodes='all'):
        nodes = self._nodes_or_all(nodes)

        for v in nodes:
            delattr(self, self._node_tag_name(v))
            del self.node[v]['tag']

    def remove_edge_tag(self, edges='all'):
        edges = self._edges_or_all(edges)

        for u, v in edges:
            delattr(self, self._edge_tag_name(u, v))
            del self[u][v]['tag']

    def edge_tags(self):
        for u, v in self.edges():
            yield self[u][v]['tag']

    def node_tags(self):
        for v in self.nodes():
            yield self.node[v]['tag']

    def states(self):
        for v in self.nodes():
            yield self.node[v]['state']

    def named_edge_tags(self):
        for u, v in self.edges():
            yield ((u, v), self[u][v]['tag'])

    def named_node_tags(self):
        for v in self.nodes():
            yield (v, self.node[v]['tag'])

    def named_states(self):
        for v in self.nodes():
            yield (v, self.node[v]['state'])

    def register_message_func(self, message_func, edges='all', batched=False):
        '''
        batched: whether to do a single batched computation instead of iterating
        message function: accepts source state tensor and edge tag tensor, and
        returns a message tensor
        '''
        self.message_funcs.append((self._edges_or_all(edges), message_func, batched))

    def register_update_func(self, update_func, nodes='all', batched=False):
        '''
        batched: whether to do a single batched computation instead of iterating
        update function: accepts a node attribute dictionary (including state and tag),
        and a dictionary of edge attribute dictionaries
        '''
        self.update_funcs.append((self._nodes_or_all(nodes), update_func, batched))

    def step(self):
        # update message
        for ebunch, f, batched in self.message_funcs:
            if batched:
                # FIXME: need to optimize since we are repeatedly stacking and
                # unpacking
                source = T.stack([self.node[u]['state'] for u, _ in ebunch])
                edge_tag = T.stack([self[u][v]['tag'] for u, v in ebunch])
                message = f(source, edge_tag)
                for i, (u, v) in enumerate(ebunch):
                    self[u][v]['state'] = message[i]
            else:
                for u, v in ebunch:
                    self[u][v]['state'] = f(
                            self.node[u]['state'],
                            self[u][v]['tag']
                            )

        # update state
        # TODO: does it make sense to batch update the nodes?
        for v, f in self.update_funcs:
            self.node[v]['state'] = f(self.node[v], self[v])
