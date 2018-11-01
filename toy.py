###############################################################################
# A toy example
# -------------
#
# Let’s begin with the simplest graph possible with two nodes, and set
# the node representations:

import torch as th
import dgl

g = dgl.DGLGraph()
g.add_nodes(2)
g.add_edge(1, 0)

x = th.tensor([[0.0, 0.0], [1.0, 2.0]])
g.nodes[:].data['x'] = x

###############################################################################
# What we want to do is simply to copy representation from node#1 to
# node#0, but with a message passing interface. We do this like what we
# will do over a pair of sockets, with a send and a recv interface. The
# two user defined function (UDF) specifies the actions: deposit the
# value into an internal key-value store with the key msg, and retrive
# it. Note that there may be multiple incoming edges to a node, and the
# receiving end aggregates them.

def send_source(edges):  # type is dgl.EdgeBatch
    return {'msg': edges.src['x']}

def simple_reduce(nodes):  # type is dgl.NodeBatch
    msgs = nodes.mailbox['msg']
    return {'x' : th.sum(msgs, dim=1)}

g.send((1, 0), message_func=send_source)
g.recv(0, reduce_func=simple_reduce)
print(g.nodes[:].data)

###############################################################################
# Some times the computation may involve representations on the edges.
# Let’s say we want to “amplify” the message:

w = th.tensor([2.0])
g.edges[:].data['w'] = w

def send_source_with_edge_weight(edges):
    return {'msg': edges.src['x'] * edges.data['w']}

g.send((1, 0), message_func=send_source_with_edge_weight)
g.recv(0, reduce_func=simple_reduce)
print(g.nodes[:].data)

###############################################################################
# Or we may need to involve the desination’s representation, and here
# is one version:

def simple_reduce_addup(nodes):
    msgs = nodes.mailbox['msg']
    return {'x' : nodes.data['x'] + th.sum(msgs, dim=1)}

g.send((1, 0), message_func=send_source_with_edge_weight)
g.recv(0, reduce_func=simple_reduce_addup)
print(g.nodes[:].data)
