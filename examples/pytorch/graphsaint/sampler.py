import math
import os
import random
import time

import dgl
import dgl.function as fn

import numpy as np
import scipy
import torch as th
from dgl.sampling import pack_traces, random_walk
from torch.utils.data import DataLoader


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
    node_budget : int
        the expected number of nodes in each subgraph, which is specifically explained in the paper. Actually this
        param specifies the times of sampling nodes from the original graph with replacement. The meaning of edge_budget
        is similar to the node_budget.
    dn : str
        name of dataset.
    g : DGLGraph
        the full graph.
    train_nid : list
        ids of training nodes.
    num_workers_sampler : int
        number of processes to sample subgraphs in pre-sampling procedure using torch.dataloader.
    num_subg_sampler : int, optional
        the max number of subgraphs sampled in pre-sampling phase for computing normalization coefficients in the beginning.
        Actually this param is used as ``__len__`` of sampler in pre-sampling phase.
        Please make sure that num_subg_sampler is greater than batch_size_sampler so that we can sample enough subgraphs.
        Defaults: 10000
    batch_size_sampler : int, optional
        the number of subgraphs sampled by each process concurrently in pre-sampling phase.
        Defaults: 200
    online : bool, optional
        If `True`, we employ online sampling in training phase. Otherwise employing offline sampling.
        Defaults: True
    num_subg : int, optional
        the expected number of sampled subgraphs in pre-sampling phase.
        It is actually the 'N' in the original paper. Note that this param is different from the num_subg_sampler.
        This param is just used to control the number of pre-sampled subgraphs.
        Defaults: 50
    full : bool, optional
        True if the number of subgraphs used in the training phase equals to that of pre-sampled subgraphs, or
        ``math.ceil(self.train_g.num_nodes() / self.node_budget)``. This formula takes the result of A divided by B as
        the number of subgraphs used in the training phase, where A is the number of training nodes in the original
        graph, B is the expected number of nodes in each pre-sampled subgraph. Please refer to the paper to check the
        details.
        Defaults: True

    Notes
    -----
    For parallelism of pre-sampling, we utilize `torch.DataLoader` to concurrently speed up sampling.
    The `num_subg_sampler` is the return value of `__len__` in pre-sampling phase. Moreover, the param `batch_size_sampler`
    determines the batch_size of `torch.DataLoader` in internal pre-sampling part. But note that if we wanna pass the
    SAINTSampler to `torch.DataLoader` for concurrently sampling subgraphs in training phase, we need to specify
    `batch_size` of `DataLoader`, that is, `batch_size_sampler` is not related to how sampler works in training procedure.
    """

    def __init__(
        self,
        node_budget,
        dn,
        g,
        train_nid,
        num_workers_sampler,
        num_subg_sampler=10000,
        batch_size_sampler=200,
        online=True,
        num_subg=50,
        full=True,
    ):
        self.g = g.cpu()
        self.node_budget = node_budget
        self.train_g: dgl.graph = g.subgraph(train_nid)
        self.dn, self.num_subg = dn, num_subg
        self.node_counter = th.zeros((self.train_g.num_nodes(),))
        self.edge_counter = th.zeros((self.train_g.num_edges(),))
        self.prob = None
        self.num_subg_sampler = num_subg_sampler
        self.batch_size_sampler = batch_size_sampler
        self.num_workers_sampler = num_workers_sampler
        self.train = False
        self.online = online
        self.full = full

        assert (
            self.num_subg_sampler >= self.batch_size_sampler
        ), "num_subg_sampler should be greater than batch_size_sampler"
        graph_fn, norm_fn = self.__generate_fn__()

        if os.path.exists(graph_fn):
            self.subgraphs = np.load(graph_fn, allow_pickle=True)
            aggr_norm, loss_norm = np.load(norm_fn, allow_pickle=True)
        else:
            os.makedirs("./subgraphs/", exist_ok=True)

            self.subgraphs = []
            self.N, sampled_nodes = 0, 0
            # N: the number of pre-sampled subgraphs

            # Employ parallelism to speed up the sampling procedure
            loader = DataLoader(
                self,
                batch_size=self.batch_size_sampler,
                shuffle=True,
                num_workers=self.num_workers_sampler,
                collate_fn=self.__collate_fn__,
                drop_last=False,
            )

            t = time.perf_counter()
            for num_nodes, subgraphs_nids, subgraphs_eids in loader:
                self.subgraphs.extend(subgraphs_nids)
                sampled_nodes += num_nodes

                _subgraphs, _node_counts = np.unique(
                    np.concatenate(subgraphs_nids), return_counts=True
                )
                sampled_nodes_idx = th.from_numpy(_subgraphs)
                _node_counts = th.from_numpy(_node_counts)
                self.node_counter[sampled_nodes_idx] += _node_counts

                _subgraphs_eids, _edge_counts = np.unique(
                    np.concatenate(subgraphs_eids), return_counts=True
                )
                sampled_edges_idx = th.from_numpy(_subgraphs_eids)
                _edge_counts = th.from_numpy(_edge_counts)
                self.edge_counter[sampled_edges_idx] += _edge_counts

                self.N += len(subgraphs_nids)  # number of subgraphs
                if sampled_nodes > self.train_g.num_nodes() * num_subg:
                    break

            print(f"Sampling time: [{time.perf_counter() - t:.2f}s]")
            np.save(graph_fn, self.subgraphs)

            t = time.perf_counter()
            aggr_norm, loss_norm = self.__compute_norm__()
            print(f"Normalization time: [{time.perf_counter() - t:.2f}s]")
            np.save(norm_fn, (aggr_norm, loss_norm))

        self.train_g.ndata["l_n"] = th.Tensor(loss_norm)
        self.train_g.edata["w"] = th.Tensor(aggr_norm)
        self.__compute_degree_norm()  # basically normalizing adjacent matrix

        random.shuffle(self.subgraphs)
        self.__clear__()
        print("The number of subgraphs is: ", len(self.subgraphs))

        self.train = True

    def __len__(self):
        if self.train is False:
            return self.num_subg_sampler
        else:
            if self.full:
                return len(self.subgraphs)
            else:
                return math.ceil(self.train_g.num_nodes() / self.node_budget)

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
            subgraph_eids = dgl.node_subgraph(
                self.train_g, subgraph_nids
            ).edata[dgl.EID]
            return num_nodes, subgraph_nids, subgraph_eids

    def __collate_fn__(self, batch):
        if (
            self.train
        ):  # sample only one graph each epoch, batch_size in training phase in 1
            return batch[0]
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

        self.train_g.ndata["n_c"] = self.node_counter
        self.train_g.edata["e_c"] = self.edge_counter
        self.train_g.apply_edges(fn.v_div_e("n_c", "e_c", "a_n"))
        aggr_norm = self.train_g.edata.pop("a_n")

        self.train_g.ndata.pop("n_c")
        self.train_g.edata.pop("e_c")

        return aggr_norm.numpy(), loss_norm.numpy()

    def __compute_degree_norm(self):
        self.train_g.ndata[
            "train_D_norm"
        ] = 1.0 / self.train_g.in_degrees().float().clamp(min=1).unsqueeze(1)
        self.g.ndata["full_D_norm"] = 1.0 / self.g.in_degrees().float().clamp(
            min=1
        ).unsqueeze(1)

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
        super(SAINTNodeSampler, self).__init__(
            node_budget=node_budget, **kwargs
        )

    def __generate_fn__(self):
        graph_fn = os.path.join(
            "./subgraphs/{}_Node_{}_{}.npy".format(
                self.dn, self.node_budget, self.num_subg
            )
        )
        norm_fn = os.path.join(
            "./subgraphs/{}_Node_{}_{}_norm.npy".format(
                self.dn, self.node_budget, self.num_subg
            )
        )
        return graph_fn, norm_fn

    def __sample__(self):
        if self.prob is None:
            self.prob = self.train_g.in_degrees().float().clamp(min=1)

        sampled_nodes = th.multinomial(
            self.prob, num_samples=self.node_budget, replacement=True
        ).unique()
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

        super(SAINTEdgeSampler, self).__init__(
            node_budget=edge_budget * 2, **kwargs
        )

    def __generate_fn__(self):
        graph_fn = os.path.join(
            "./subgraphs/{}_Edge_{}_{}.npy".format(
                self.dn, self.edge_budget, self.num_subg
            )
        )
        norm_fn = os.path.join(
            "./subgraphs/{}_Edge_{}_{}_norm.npy".format(
                self.dn, self.edge_budget, self.num_subg
            )
        )
        return graph_fn, norm_fn

    # TODO: only sample half edges, then add another half edges
    # TODO: use numpy to implement cython sampling method
    def __sample__(self):
        if self.prob is None:
            src, dst = self.train_g.edges()
            src_degrees, dst_degrees = self.train_g.in_degrees(
                src
            ).float().clamp(min=1), self.train_g.in_degrees(dst).float().clamp(
                min=1
            )
            prob_mat = 1.0 / src_degrees + 1.0 / dst_degrees
            prob_mat = scipy.sparse.csr_matrix(
                (prob_mat.numpy(), (src.numpy(), dst.numpy()))
            )
            # The edge probability here only contains that of edges in upper triangle adjacency matrix
            # Because we assume the graph is undirected, that is, the adjacency matrix is symmetric. We only need
            # to consider half of edges in the graph.
            self.prob = th.tensor(scipy.sparse.triu(prob_mat).data)
            self.prob /= self.prob.sum()
            self.adj_nodes = np.stack(prob_mat.nonzero(), axis=1)

        sampled_edges = np.unique(
            dgl.random.choice(
                len(self.prob),
                size=self.edge_budget,
                prob=self.prob,
                replace=False,
            )
        )
        sampled_nodes = np.unique(
            self.adj_nodes[sampled_edges].flatten()
        ).astype("long")
        return sampled_nodes


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
        super(SAINTRandomWalkSampler, self).__init__(
            node_budget=num_roots * length, **kwargs
        )

    def __generate_fn__(self):
        graph_fn = os.path.join(
            "./subgraphs/{}_RW_{}_{}_{}.npy".format(
                self.dn, self.num_roots, self.length, self.num_subg
            )
        )
        norm_fn = os.path.join(
            "./subgraphs/{}_RW_{}_{}_{}_norm.npy".format(
                self.dn, self.num_roots, self.length, self.num_subg
            )
        )
        return graph_fn, norm_fn

    def __sample__(self):
        sampled_roots = th.randint(
            0, self.train_g.num_nodes(), (self.num_roots,)
        )
        traces, types = random_walk(
            self.train_g, nodes=sampled_roots, length=self.length
        )
        sampled_nodes, _, _, _ = pack_traces(traces, types)
        sampled_nodes = sampled_nodes.unique()
        return sampled_nodes.numpy()
