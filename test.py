import torch as th
import dgl
import dgl.backend as F
import dgl.function as fn
import torch.nn as nn

in_feat = 5
out_feat = 4

def edge_udf(edges):
	# return {'h': edges.data['h'] * th.transpose(edges.src['h'], 0, 1)}
	return {'h': edges.src['h']}

th.cuda.set_device("cuda:0")
hg = dgl.heterograph({
        ('user', 'follows', 'user'): ([0, 1], [1, 0]),
        ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 1, 1])
        }, device="cuda:0")

g = dgl.to_homogeneous(hg)
g.ndata['h'] = th.ones((g.num_nodes(), in_feat), dtype=th.float32).to("cuda:0")

num_rel = hg._graph.number_of_etypes()
weight = nn.Parameter(th.Tensor(num_rel, in_feat, out_feat)).to("cuda:0")
nn.init.xavier_uniform_(weight, gain=nn.init.calculate_gain('relu'))

g.gather_mm(edge_udf, weight)



