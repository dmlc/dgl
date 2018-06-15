import networkx as nx
from networkx.classes.digraph import DiGraph

if __name__ == '__main__':
    from torch.autograd import Variable as Var

    th.random.manual_seed(0)

    print("testing vanilla RNN update")
    g_path = mx_Graph(nx.path_graph(2))
    g_path.set_repr(0, th.rand(2, 128))
    g_path.sendto(0, 1)
    g_path.recvfrom(1, [0])
    g_path.readout()

    '''
    # this makes a uni-edge tree
    tr = nx.bfs_tree(nx.balanced_tree(2, 3), 0)
    m_tr = mx_Graph(tr)
    m_tr.print_all()
    '''
    print("testing GRU update")
    g = mx_Graph(nx.path_graph(3))
    update_net = DefaultUpdateModule(h_dims=4, net_type='gru')
    g.register_update_func(update_net)
    msg_net = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    g.register_message_func(msg_net)

    for n in g:
        g.set_repr(n, th.rand(2, 4))

    y_pre = g.readout()
    g.update_from(0)
    y_after = g.readout()

    upd_nets = DefaultUpdateModule(h_dims=4, net_type='gru', n_func=2)
    g.register_update_func(upd_nets)
    g.update_from(0)
    g.update_from(0)
