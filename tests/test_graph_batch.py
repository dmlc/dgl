import networkx as nx
import dgl

def tree1():
    """Generate a tree
         0
        / \
       1   2
      / \
     3   4
    Edges are from leaves to root.
    """
    g = nx.DiGraph()
    g.add_node(0, x=0)
    g.add_node(1, x=1)
    g.add_node(2, x=2)
    g.add_node(3, x=3)
    g.add_node(4, x=4)
    g.add_edge(3, 1)
    g.add_edge(4, 1)
    g.add_edge(1, 0)
    g.add_edge(2, 0)
    return g

def tree2():
    """Generate a tree
         1
        / \
       4   3
      / \
     2   0
    Edges are from leaves to root.
    """
    g = nx.DiGraph()
    g.add_node(0, x=0)
    g.add_node(1, x=1)
    g.add_node(2, x=2)
    g.add_node(3, x=3)
    g.add_node(4, x=4)
    g.add_edge(2, 4)
    g.add_edge(0, 4)
    g.add_edge(4, 1)
    g.add_edge(3, 1)
    return g

def fmsg(u, v, e_uv):
    return u['x']

def fupdate(u, accum):
    return {'x' : accum}

def test_batch_sendrecv():
    t1 = dgl.DGLGraph(tree1())
    t2 = dgl.DGLGraph(tree2())
    bg = dgl.batch([t1, t2])
    bg.register_message_func(fmsg)
    bg.register_reduce_func('sum')
    bg.register_update_func(fupdate)
    u = [(0, 3), (1, 2)]
    v = [(0, 1), (1, 4)]
    bg.sendto(u, v)
    bg.recv(v)
    assert bg.nodes[(0, 1)]['x'] == 3
    assert bg.nodes[(1, 4)]['x'] == 2
    trees = dgl.unbatch(bg)
    assert trees[0].nodes[1]['x'] == 3
    assert trees[1].nodes[4]['x'] == 2

def test_batch_propagate():
    t1 = dgl.DGLGraph(tree1())
    t2 = dgl.DGLGraph(tree2())
    bg = dgl.batch([t1, t2])
    bg.register_message_func(fmsg)
    bg.register_reduce_func('sum')
    bg.register_update_func(fupdate)
    # get leaves.
    leaves = [u for u in bg.nodes if bg.in_degree(u) == 0]
    print(leaves)
    iterator = []
    frontier = [u for u in bg.nodes if bg.out_degree(u) == 0]
    while frontier:
        src = sum([list(bg.pred[x]) for x in frontier], [])
        trg = sum([[x] * len(bg.pred[x]) for x in frontier], [])
        iterator.append((src, trg))
        frontier = src
    bg.propagate(reversed(iterator))
    trees = dgl.unbatch(bg)
    assert trees[0].nodes[0]['x'] == 9
    assert trees[1].nodes[1]['x'] == 5

if __name__ == '__main__':
    test_batch_sendrecv()
    test_batch_propagate()
