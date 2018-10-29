import dgl
import torch


if __name__ == '__main__':
    N = 3  # number of nodes
    DN = 1  # size of node representations
    DE = 1  # size of edge representations

    g = dgl.DGLGraph()

    # Test node addition
    g.add_nodes(N)
    g.set_n_repr({'h1': torch.randn(N, DN),
                  'h2': torch.randn(N, DN)})
    print('Node representations before dynamic addition')
    print('--------------------------------------------')
    print(g.get_n_repr())
    g.add_nodes(3)
    print('Node representations after dynamic addition')
    print('--------------------------------------------')
    print(g.get_n_repr())

    # Test edge addition
    g.add_edge(0, 1)
    g.add_edge(1, 0)
    g.set_e_repr({'h1': torch.randn(2, DE),
                  'h2': torch.randn(2, DE)})
    print('Edge representations before dynamic addition')
    print('--------------------------------------------')
    print(g.get_e_repr())

    g.add_edges([0, 2], [2, 0])
    print('Edge representations after adding edge')
    print('--------------------------------------------')
    print(g.get_e_repr())

    g.add_edge(1, 2)
    print('Edge representations after adding edges')
    print('--------------------------------------------')
    print(g.get_e_repr())
