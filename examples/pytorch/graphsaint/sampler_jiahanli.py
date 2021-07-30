import math
import os
import time
import torch as th
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
import dgl.function as fn
import dgl
from dgl.sampling import random_walk, pack_traces


# The base class of sampler
# (TODO): online sampling
# NOTE: we have to do offline sampling for computing normalization coefficients
class SAINTSampler(Dataset):
    def __init__(self, dn, g, train_nid, node_budget, num_workers, train, num_subg_train=0, num_subg_norm=10000,
                 batch_size_norm=200, online=False, num_repeat=50):
        """
        :param num_subg_train: the number of subgraphs used in training per epoch, which is equal to the number of
                                iterations in per epoch. This param can be specified by user as any number. If it's set
                                as 0 (by default), num_subg_train = len(train_nid) / node_budget,
                                which is how author computes num_subg_train
        :param num_subg_norm: the number of subgraphs sampled in advance for computing normalization coefficients.
                                Please make sure that num_subg_norm is greater than batch_size_norm so that we can
                                sample enough subgraphs.
        :param dn: name of dataset
        :param g: full graph
        :param train_nid: ids of training nodes
        :param node_budget: expected number of sampled nodes. Actually node_budget is the times of sampling nodes
                            which doesn't mean we sample 6000 different nodes.
        :param num_workers: number of processes to sample subgraphs using torch.dataloader
        :param train: True if sampling subgraphs in training procedure
        :param online: True if resampling new subgraphs in training procedure; False if utilizing sampled subgraphs
                        (which are used when computing normalization coefficients) for training and sampling new
                        subgraphs if the model need more. If online == 'tot', employ totally offline sampling, which
                        means all subgraphs originates from sampled subgraphs
                        This param is valid only when 'train' is True.
        :param num_repeat: the expected number of sampled subgraphs in advance for computing normalization statistics,
                            which is actually the 'N' in the original paper. Note that this param is different from the
                            num_subg_norm.

        The number of sampled subgraphs in advance for computing normalization statistics is actually limited by
        len(train_nid) * num_repeat, that is, the total number of nodes in these subgraphs should be less than
        len(train_nid) * num_repeat. However, for parallelism by torch.dataloader to speed up sampling these subgraphs,
        we need to specify the property __len__, which actually is the maximum number of subgraphs we can sample
        (ideally without the limit of len(train_nid) * num_repeat). So __len__ is specified as num_sub_norm when sampling
        happens for computing normalization statistics while num_sub_train when training.

        Moreover, num_subg_norm is not specified in paper and author's codes. It's introduced here for torch.dataloader.

        """
        # NOTE: In author's codes, the number of sampled subgraphs in advance for computing normalization coefficients
        # is controlled by self.train_g.num_nodes() * num_repeat where num_repeat is the expected number of sampled
        # subgraphs.
        self.g = g.cpu()
        self.train_g: dgl.graph = g.subgraph(train_nid)
        self.dn, self.num_repeat = dn, num_repeat
        self.node_counter = th.zeros((self.train_g.num_nodes(),))
        self.edge_counter = th.zeros((self.train_g.num_edges(),))
        self.prob = None
        self.num_subg_train = num_subg_train
        self.num_subg_norm = num_subg_norm
        self.batch_size_norm = batch_size_norm
        self.num_workers = num_workers
        self.train = train
        self.online = online
        self.cnt = 0 # count the times sampled subgraphs have been fetched.
        test = True # TODO: TEST

        assert self.num_subg_norm >= self.batch_size_norm, "num_sub_norm should be greater than batch_size_norm"
        if self.num_subg_train == 0:
            self.num_subg_train = math.ceil(self.train_g.num_nodes() / node_budget)  # TODO: weird!!!
        graph_fn, norm_fn = self.__generate_fn__() # NOTE: the file to store sampled graphs and computed norms

        if os.path.exists(graph_fn) and test is False:
            self.subgraphs = np.load(graph_fn, allow_pickle=True)
            aggr_norm, loss_norm = np.load(norm_fn, allow_pickle=True)
        else:
            os.makedirs('./subgraphs/', exist_ok=True)

            self.subgraphs = []
            self.N, sampled_nodes = 0, 0
            # N: the number of subgraphs sampled in __init__

            # Employ parallelism to speed up the sampling procedure
            loader = DataLoader(self, batch_size=self.batch_size_norm, shuffle=True, num_workers=self.num_workers,
                                collate_fn=self.__collate_fn__, drop_last=False)

            t = time.perf_counter()
            for num_nodes, subgraphs_nids, subgraphs_eids in loader:
                # t0 = time.perf_counter()
                # print('Sampling time consumption: {}'.format(t0 - t))
                self.subgraphs.extend(subgraphs_nids)
                sampled_nodes += num_nodes

                _subgraphs, _node_counts = np.unique(np.concatenate(subgraphs_nids), return_counts=True)
                sampled_nodes_idx = th.from_numpy(_subgraphs)
                _node_counts = th.from_numpy(_node_counts)
                self.node_counter[sampled_nodes_idx] += _node_counts

                # t1 = time.perf_counter()
                # print('Node counter time consumption: {}'.format(t1 - t0))

                _subgraphs_eids, _edge_counts = np.unique(np.concatenate(subgraphs_eids), return_counts=True)
                sampled_edges_idx = th.from_numpy(_subgraphs_eids)
                _edge_counts = th.from_numpy(_edge_counts)
                self.edge_counter[sampled_edges_idx] += _edge_counts

                # t2 = time.perf_counter()
                # print('Edge counter time consumption: {}'.format(t2 - t1))
                self.N += len(subgraphs_nids) # NOTE: number of subgraphs
                if sampled_nodes > self.train_g.num_nodes() * num_repeat:
                    break

            print(f'Sampling time: [{time.perf_counter() - t:.2f}s]')
            np.save(graph_fn, self.subgraphs) # NOTE: graph_fn, dir storing sampled subgraphs

            t = time.perf_counter()
            aggr_norm, loss_norm = self.__compute_norm__()
            print(f'Normalization time: [{time.perf_counter() - t:.2f}s]')
            np.save(norm_fn, (aggr_norm, loss_norm)) # NOTE: aggr_norm is related to edges, while loss_norm is related to nodes.

        self.train_g.ndata['l_n'] = th.Tensor(loss_norm)
        self.train_g.edata['w'] = th.Tensor(aggr_norm)
        self.__compute_degree_norm() # NOTE: used for basically normalizing adjacent matrix

        random.shuffle(self.subgraphs)
        self.__clear__()
        # NOTE: statistics below are not accurate
        print("The number of subgraphs is: ", len(self.subgraphs))
        print("The size of subgraphs is about: ", len(self.subgraphs[-1]))

    def __len__(self):
        if self.train is False:
            return self.num_subg_norm
        else:
            return self.num_subg_train

    def __getitem__(self, idx):
        # Only when sampling subgraphs in training procedure and need to utilize sampled subgraphs and we still
        # have sampled subgraphs can we fetch a subgraph from sampled subgraphs
        if self.train:
            if self.online is True:
                subgraph = self.__sample__()
                return dgl.node_subgraph(self.train_g, subgraph)
            else:
                if self.cnt < len(self.subgraphs):
                    self.cnt += 1
                    return dgl.node_subgraph(self.train_g, self.subgraphs[self.cnt-1])
                elif self.online == 'tot':
                    random.shuffle(self.subgraphs)
                    self.cnt = 1
                    return dgl.node_subgraph(self.train_g, self.subgraphs[self.cnt-1])
                else:
                    subgraph = self.__sample__()
                    return dgl.node_subgraph(self.train_g, subgraph)
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
    def __init__(self, node_budget, **kwargs):
        self.node_budget = node_budget
        super(SAINTNodeSampler, self).__init__(node_budget=node_budget, **kwargs)

    def __generate_fn__(self):
        graph_fn = os.path.join('./subgraphs/{}_Node_{}_{}.npy'.format(self.dn, self.node_budget,
                                                                       self.num_repeat))
        norm_fn = os.path.join('./subgraphs/{}_Node_{}_{}_norm.npy'.format(self.dn, self.node_budget,
                                                                           self.num_repeat))
        return graph_fn, norm_fn

    def __sample__(self):
        if self.prob is None:
            self.prob = self.train_g.in_degrees().float().clamp(min=1) # NOTE: clamp, clip a segment of data -jiahanli

        sampled_nodes = th.multinomial(self.prob, num_samples=self.node_budget, replacement=True).unique() # TODO: weird? why replacement=True?
        return sampled_nodes.numpy()


class SAINTEdgeSampler(SAINTSampler):
    def __init__(self, edge_budget, **kwargs):
        self.edge_budget = edge_budget
        super(SAINTEdgeSampler, self).__init__(node_budget=edge_budget*2, **kwargs)

    def __generate_fn__(self):
        graph_fn = os.path.join('./subgraphs/{}_Edge_{}_{}.npy'.format(self.dn, self.edge_budget,
                                                                       self.num_repeat))
        norm_fn = os.path.join('./subgraphs/{}_Edge_{}_{}_norm.npy'.format(self.dn, self.edge_budget,
                                                                           self.num_repeat))
        return graph_fn, norm_fn

    def __sample__(self):
        if self.prob is None:
            src, dst = self.train_g.edges()
            src_degrees, dst_degrees = self.train_g.in_degrees(src).float().clamp(min=1),\
                                       self.train_g.in_degrees(dst).float().clamp(min=1)
            self.prob = 1. / src_degrees + 1. / dst_degrees

        sampled_edges = th.multinomial(self.prob, num_samples=self.edge_budget, replacement=True).unique()

        sampled_src, sampled_dst = self.train_g.find_edges(sampled_edges)
        sampled_nodes = th.cat([sampled_src, sampled_dst]).unique()
        return sampled_nodes.numpy()


class SAINTRandomWalkSampler(SAINTSampler):
    def __init__(self, num_roots, length, **kwargs):
        self.num_roots, self.length = num_roots, length
        super(SAINTRandomWalkSampler, self).__init__(node_budget=num_roots*length, **kwargs)

    def __generate_fn__(self):
        graph_fn = os.path.join('./subgraphs/{}_RW_{}_{}_{}.npy'.format(self.dn, self.num_roots,
                                                                        self.length, self.num_repeat))
        norm_fn = os.path.join('./subgraphs/{}_RW_{}_{}_{}_norm.npy'.format(self.dn, self.num_roots,
                                                                            self.length, self.num_repeat))
        return graph_fn, norm_fn

    def __sample__(self):
        sampled_roots = th.randint(0, self.train_g.num_nodes(), (self.num_roots, ))
        traces, types = random_walk(self.train_g, nodes=sampled_roots, length=self.length)
        sampled_nodes, _, _, _ = pack_traces(traces, types)
        sampled_nodes = sampled_nodes.unique()
        return sampled_nodes.numpy()


