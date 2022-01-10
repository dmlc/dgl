import torch as th
import dgl
import dgl.backend as F
import dgl.function as fn
import torch.nn as nn


in_feat = 5
def udf_copy_src(edges):
    return {'m': edges.src['u']}

def edge_udf(edges):
	# print(edges.data['h'])
	# print(edges.src['h'])
	# return {'h': edges.data['h'] * th.transpose(edges.src['h'], 0, 1)}
	return {'h': edges.src['h']}

	# return {'h': edges.src['h']}
# g.update_all(udf_copy_src, udf_reduce[red])
th.cuda.set_device("cuda:0")
hg = dgl.heterograph({
        ('user', 'follows', 'user'): ([0, 1], [1, 2]),
        ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 1, 1])
        })
# g = dgl.heterograph({('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 2, 1])})
# g = g.to(args.gpu)
hg.nodes['user'].data['h'] = th.ones((3, in_feat))
hg.edges[('user', 'plays', 'game')].data['h'] = th.ones(4, in_feat)
hg.edges[('user', 'follows', 'user')].data['h'] = th.ones(2, in_feat)

g = dgl.to_homogeneous(hg)
g.ndata['h'] = th.ones((g.num_nodes(), in_feat))

num_rel = hg._graph.number_of_etypes()

out_feat = 4
weight = nn.Parameter(th.Tensor(num_rel, in_feat, out_feat))
nn.init.xavier_uniform_(weight, gain=nn.init.calculate_gain('relu'))

print(weight.shape)
# g.apply_edges(lambda edges: {'h': edges.data['h'] * 2})
# g.apply_edges(udf_copy_src * 2)
# hm_g = dgl.to_homogeneous(g)
# hm_g.edata['h'] = th.ones(4, 5)
# print("before", hm_g.edata['h'])

print("before", hg.edges[('user', 'plays', 'game')].data['h'])
with F.record_grad():
	g.gather_mm(edge_udf, weight)
	# g.apply_edges(fn.copy_u('h', 'm'))
	# g.apply_edges(edge_udf, etype=('user', 'plays', 'game'))


# print(g.edges[('user', 'plays', 'game')].data['h'])