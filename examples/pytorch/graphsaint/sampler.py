import os
import time
import torch as th
from torch.utils.data import DataLoader
import random
import numpy as np
import dgl.function as fn
import dgl
from dgl.sampling import random_walk, pack_traces


# The base class of sampler
class SAINTSampler:
    """
    Description
    -----------
    SAINTSampler implements the sampler described in GraphSAINT. This sampler implements offline sampling in
    pre-sampling phase as well as fully offline sampling, fully online sampling in training phase.
    Users can conveniently set param 'online' of the sampler to choose different modes.

    Parameters
    ----------
    dn : str
        name of dataset.
    g : DGLGraph
        the full graph.
    train_nid : list
        ids of training nodes.
    num_workers_sampler : int
        number of processes to sample subgraphs in pre-sampling procedure using torch.dataloader.
    num_subg_norm : int, optional
        the max number of subgraphs sampled in pre-sampling phase for computing normalization coefficients in the beginning.
        Actually this param is used as `__len__` of sampler in pre-sampling phase.
        Please make sure that num_subg_norm is greater than batch_size_norm so that we can sample enough subgraphs.
        Defaults: 10000
    batch_size_norm : int, optional
        the number of subgraphs sampled by each process concurrently in pre-sampling phase.
        Defaults: 200
    online : bool, optional
        If `True`, we employ online sampling in training phase. Otherwise employing offline sampling.
        Defaults: True
    num_subg : int, optional
        the expected number of sampled subgraphs in pre-sampling phase.
        It is actually the 'N' in the original paper. Note that this param is different from the num_subg_norm.
        This param is just used to control the number of pre-sampled subgraphs.
        Defaults: 50

    Notes
    -----
    Here our implementation is a little bit different from the codes of the author. For example, author has limited the
    number of iterations in training phase to `math.ceil(self.train_g.num_nodes() / node_budget)` ( please refer to
    annotations of `SAINTNodeSampler` to check the explanation of `node_budget`),
    which can not fully make use of all pre-sampled subgraphs. We've remove this restriction.

    For parallelism of pre-sampling, we utilize `torch.DataLoader` to concurrently speed up sampling.
    The `num_subg_norm` is the return value of `__len__` in pre-sampling phase. Moreover, the param `batch_size_norm`
    determines the batch_size of `torch.DataLoader` in internal pre-sampling part. But note that if we wanna pass the
    SAINTSampler to `torch.DataLoader` for concurrently sampling subgraphs in training phase, we need to specify
    `batch_size` of `DataLoader`, that is, `batch_size_norm` is not related to how sampler works in training procedure.
    """

    def __init__(self, dn, g, train_nid, num_workers_sampler, num_subg_norm=10000,
                 batch_size_norm=200, online=True, num_subg=50):
        self.g = g.cpu()
        self.train_g: dgl.graph = g.subgraph(train_nid)
        self.dn, self.num_subg = dn, num_subg
        self.node_counter = th.zeros((self.train_g.num_nodes(),))
        self.edge_counter = th.zeros((self.train_g.num_edges(),))
        self.prob = None
        self.num_subg_norm = num_subg_norm
        self.batch_size_norm = batch_size_norm
        self.num_workers_sampler = num_workers_sampler
        self.train = False
        self.online = online

        assert self.num_subg_norm >= self.batch_size_norm, "num_subg_norm should be greater than batch_size_norm"
        graph_fn, norm_fn = self.__generate_fn__()

        if os.path.exists(graph_fn):
            self.subgraphs = np.load(graph_fn, allow_pickle=True)
            aggr_norm, loss_norm = np.load(norm_fn, allow_pickle=True)
        else:
            os.makedirs('./subgraphs/', exist_ok=True)

            self.subgraphs = []
            self.N, sampled_nodes = 0, 0
            # N: the number of pre-sampled subgraphs

            # Employ parallelism to speed up the sampling procedure
            loader = DataLoader(self, batch_size=self.batch_size_norm, shuffle=True,
                                num_workers=self.num_workers_sampler, collate_fn=self.__collate_fn__, drop_last=False)

            t = time.perf_counter()
            for num_nodes, subgraphs_nids, subgraphs_eids in loader:

                self.subgraphs.extend(subgraphs_nids)
                sampled_nodes += num_nodes

                _subgraphs, _node_counts = np.unique(np.concatenate(subgraphs_nids), return_counts=True)
                sampled_nodes_idx = th.from_numpy(_subgraphs)
                _node_counts = th.from_numpy(_node_counts)
                self.node_counter[sampled_nodes_idx] += _node_counts

                _subgraphs_eids, _edge_counts = np.unique(np.concatenate(subgraphs_eids), return_counts=True)
                sampled_edges_idx = th.from_numpy(_subgraphs_eids)
                _edge_counts = th.from_numpy(_edge_counts)
                self.edge_counter[sampled_edges_idx] += _edge_counts

                self.N += len(subgraphs_nids)  # number of subgraphs
                if sampled_nodes > self.train_g.num_nodes() * num_subg:
                    break

            print(f'Sampling time: [{time.perf_counter() - t:.2f}s]')
            np.save(graph_fn, self.subgraphs)

            t = time.perf_counter()
            aggr_norm, loss_norm = self.__compute_norm__()
            print(f'Normalization time: [{time.perf_counter() - t:.2f}s]')
            np.save(norm_fn, (aggr_norm, loss_norm))

        self.train_g.ndata['l_n'] = th.Tensor(loss_norm)
        self.train_g.edata['w'] = th.Tensor(aggr_norm)
        self.__compute_degree_norm()  # basically normalizing adjacent matrix

        random.shuffle(self.subgraphs)
        self.__clear__()
        print("The number of subgraphs is: ", len(self.subgraphs))

        self.train = True

    def __len__(self):
        if self.train is False:
            return self.num_subg_norm
        else:
            return len(self.subgraphs)

    def __getitem__(self, idx):
        # Only when sampling subgraphs in training procedure and need to utilize sampled subgraphs and we still
        # have sampled subgraphs we can fetch a subgraph from sampled subgraphs
        if self.train:
            if self.online:
                subgraph = self.__sample__()
                return dgl.node_subgraph(self.train_g, subgraph)
            else:
                return dgl.node_subgraph(self.train_g, self.subgraphs[idx])
        else:
            subgraph_nids = self.__sample__()
            num_nodes = len(subgraph_nids)
            subgraph_eids = dgl.node_subgraph(self.train_g, subgraph_nids).edata[dgl.EID]
            return num_nodes, subgraph_nids, subgraph_eids

    def __collate_fn__(self, batch):
        if self.train:
            subgraphs = []
            for g in batch:
                subgraphs.append(g)
            if len(subgraphs) == 1:
                return subgraphs[0]
            return subgraphs
        else:
            sum_num_nodes = 0
            subgraphs_nids_list = []
            subgraphs_eids_list = []
            for num_nodes, subgraph_nids, subgraph_eids in batch:
                sum_num_nodes += num_nodes
                subgraphs_nids_list.append(subgraph_nids)
                subgraphs_eids_list.append(subgraph_eids)
            return sum_num_nodes, subgraphs_nids_list, subgraphs_eids_list

    def __clear__(self):
        self.prob = None
        self.node_counter = None
        self.edge_counter = None
        self.g = None

    def __generate_fn__(self):
        raise NotImplementedError

    def __compute_norm__(self):

        self.node_counter[self.node_counter == 0] = 1
        self.edge_counter[self.edge_counter == 0] = 1

        loss_norm = self.N / self.node_counter / self.train_g.num_nodes()

        self.train_g.ndata['n_c'] = self.node_counter
        self.train_g.edata['e_c'] = self.edge_counter
        self.train_g.apply_edges(fn.v_div_e('n_c', 'e_c', 'a_n'))
        aggr_norm = self.train_g.edata.pop('a_n')

        self.train_g.ndata.pop('n_c')
        self.train_g.edata.pop('e_c')

        return aggr_norm.numpy(), loss_norm.numpy()

    def __compute_degree_norm(self):

        self.train_g.ndata['train_D_norm'] = 1. / self.train_g.in_degrees().float().clamp(min=1).unsqueeze(1)
        self.g.ndata['full_D_norm'] = 1. / self.g.in_degrees().float().clamp(min=1).unsqueeze(1)

    def __sample__(self):
        raise NotImplementedError


class SAINTNodeSampler(SAINTSampler):
    """
    Description
    -----------
    GraphSAINT with node sampler.

    Parameters
    ----------
    node_budget : int
        the expected number of nodes in each subgraph, which is specifically explained in the paper.
    """

    def __init__(self, node_budget, **kwargs):
        self.node_budget = node_budget
        super(SAINTNodeSampler, self).__init__(**kwargs)

    def __generate_fn__(self):
        graph_fn = os.path.join('./subgraphs/{}_Node_{}_{}.npy'.format(self.dn, self.node_budget,
                                                                       self.num_subg))
        norm_fn = os.path.join('./subgraphs/{}_Node_{}_{}_norm.npy'.format(self.dn, self.node_budget,
                                                                           self.num_subg))
        return graph_fn, norm_fn

    def __sample__(self):
        if self.prob is None:
            self.prob = self.train_g.in_degrees().float().clamp(min=1)

        sampled_nodes = th.multinomial(self.prob, num_samples=self.node_budget, replacement=True).unique()
        return sampled_nodes.numpy()


class SAINTEdgeSampler(SAINTSampler):
    """
    Description
    -----------
    GraphSAINT with edge sampler.

    Parameters
    ----------
    edge_budget : int
        the expected number of edges in each subgraph, which is specifically explained in the paper.
    """

    def __init__(self, edge_budget, **kwargs):
        self.edge_budget = edge_budget
        self.rng = np.random.default_rng()
        super(SAINTEdgeSampler, self).__init__(**kwargs)

    def __generate_fn__(self):
        graph_fn = os.path.join('./subgraphs/{}_Edge_{}_{}.npy'.format(self.dn, self.edge_budget,
                                                                       self.num_subg))
        norm_fn = os.path.join('./subgraphs/{}_Edge_{}_{}_norm.npy'.format(self.dn, self.edge_budget,
                                                                           self.num_subg))
        return graph_fn, norm_fn

    def __sample__(self):
        if self.prob is None:
            src, dst = self.train_g.edges()
            src_degrees, dst_degrees = self.train_g.in_degrees(src).float().clamp(min=1), \
                                       self.train_g.in_degrees(dst).float().clamp(min=1)
            self.prob = 1. / src_degrees + 1. / dst_degrees

        if len(self.prob) > 2 ^ 24:
            prob = self.prob.numpy()
            prob /= prob.sum()
            sampled_edges = np.unique(
                self.rng.choice(len(prob), size=self.edge_budget, p=prob, replace=True)
            )
            del prob
        else:
            sampled_edges = th.multinomial(self.prob, num_samples=self.edge_budget, replacement=True).unique()

        sampled_src, sampled_dst = self.train_g.find_edges(sampled_edges)
        sampled_nodes = th.cat([sampled_src, sampled_dst]).unique()
        return sampled_nodes.numpy()


class SAINTRandomWalkSampler(SAINTSampler):
    """
    Description
    -----------
    GraphSAINT with random walk sampler

    Parameters
    ----------
    num_roots : int
        the number of roots to generate random walks.
    length : int
        the length of each random walk.

    """

    def __init__(self, num_roots, length, **kwargs):
        self.num_roots, self.length = num_roots, length
        super(SAINTRandomWalkSampler, self).__init__(**kwargs)

    def __generate_fn__(self):
        graph_fn = os.path.join('./subgraphs/{}_RW_{}_{}_{}.npy'.format(self.dn, self.num_roots,
                                                                        self.length, self.num_subg))
        norm_fn = os.path.join('./subgraphs/{}_RW_{}_{}_{}_norm.npy'.format(self.dn, self.num_roots,
                                                                            self.length, self.num_subg))
        return graph_fn, norm_fn

    def __sample__(self):
        sampled_roots = th.randint(0, self.train_g.num_nodes(), (self.num_roots,))
        traces, types = random_walk(self.train_g, nodes=sampled_roots, length=self.length)
        sampled_nodes, _, _, _ = pack_traces(traces, types)
        sampled_nodes = sampled_nodes.unique()
        return sampled_nodes.numpy()
