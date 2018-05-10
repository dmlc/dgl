import networkx as nx
import torch as T
import torch.nn as NN
from util import *

class DiGraph(NN.Module):
    '''
    Reserved attributes:
    * state: node state vectors during message passing iterations
        edges does not have "state vectors"; the "state" field is reserved for storing messages
    * tag: node-/edge-specific feature tensors or other data
    '''
    def __init__(self, graph):
        NN.Module.__init__(self)

        self.G = graph
        self.message_funcs = []
        self.update_funcs = []

    def _nodes_or_all(self, nodes='all'):
        return self.G.nodes() if nodes == 'all' else nodes

    def _edges_or_all(self, edges='all'):
        return self.G.edges() if edges == 'all' else edges

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
            self.G.node[v]['state'] = tovar(T.zeros(shape))

    def init_node_tag_with(self, shape, init_func, dtype=T.float32, nodes='all', args=()):
        nodes = self._nodes_or_all(nodes)

        for v in nodes:
            self.G.node[v]['tag'] = init_func(NN.Parameter(tovar(T.zeros(shape, dtype=dtype))), *args)
            self.register_parameter(self._node_tag_name(v), self.G.node[v]['tag'])

    def init_edge_tag_with(self, shape, init_func, dtype=T.float32, edges='all', args=()):
        edges = self._edges_or_all(edges)

        for u, v in edges:
            self.G[u][v]['tag'] = init_func(NN.Parameter(tovar(T.zeros(shape, dtype=dtype))), *args)
            self.register_parameter(self._edge_tag_name(u, v), self.G[u][v]['tag'])

    def remove_node_tag(self, nodes='all'):
        nodes = self._nodes_or_all(nodes)

        for v in nodes:
            delattr(self, self._node_tag_name(v))
            del self.G.node[v]['tag']

    def remove_edge_tag(self, edges='all'):
        edges = self._edges_or_all(edges)

        for u, v in edges:
            delattr(self, self._edge_tag_name(u, v))
            del self.G[u][v]['tag']

    @property
    def node(self):
        return self.G.node

    @property
    def edges(self):
        return self.G.edges

    def edge_tags(self):
        for u, v in self.G.edges():
            yield self.G[u][v]['tag']

    def node_tags(self):
        for v in self.G.nodes():
            yield self.G.node[v]['tag']

    def states(self):
        for v in self.G.nodes():
            yield self.G.node[v]['state']

    def named_edge_tags(self):
        for u, v in self.G.edges():
            yield ((u, v), self.G[u][v]['tag'])

    def named_node_tags(self):
        for v in self.G.nodes():
            yield (v, self.G.node[v]['tag'])

    def named_states(self):
        for v in self.G.nodes():
            yield (v, self.G.node[v]['state'])

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
        and a list of tuples (source node, target node, edge attribute dictionary)
        '''
        self.update_funcs.append((self._nodes_or_all(nodes), update_func, batched))

    def step(self):
        # update message
        for ebunch, f, batched in self.message_funcs:
            if batched:
                # FIXME: need to optimize since we are repeatedly stacking and
                # unpacking
                source = T.stack([self.G.node[u]['state'] for u, _ in ebunch])
                edge_tags = [self.G[u][v].get('tag', None) for u, v in ebunch]
                if all(t is None for t in edge_tags):
                    edge_tag = None
                else:
                    edge_tag = T.stack([self.G[u][v]['tag'] for u, v in ebunch])
                message = f(source, edge_tag)
                for i, (u, v) in enumerate(ebunch):
                    self.G[u][v]['state'] = message[i]
            else:
                for u, v in ebunch:
                    self.G[u][v]['state'] = f(
                            self.G.node[u]['state'],
                            self.G[u][v]['tag']
                            )

        # update state
        # TODO: does it make sense to batch update the nodes?
        for vbunch, f, batched in self.update_funcs:
            for v in vbunch:
                self.G.node[v]['state'] = f(self.G.node[v], list(self.G.in_edges(v, data=True)))
