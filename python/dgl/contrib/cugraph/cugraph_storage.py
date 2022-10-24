# Copyright (c) 2019-2022, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NOTE: Requires cuGraph nightly cugraph-22.06.00a220417 or later

import dgl
from ... import backend as F
from .utils.cugraph_utils import _assert_valid_canonical_etype
from .utils.cugraph_utils import convert_can_etype_s_to_tup


class CuGraphStorage:
    """
    Duck-typed version of the DGLHeteroGraph class made for cuGraph
    for storing graph structure and node/edge feature data.

    This object is wrapper around cugraph's PropertyGraph and returns samples
    that conform with `DGLHeteroGraph`
    See: TODO link after https://github.com/rapidsai/cugraph/pull/2826


    Read the user guide chapter :ref:`guide-`CuGraphStorage` for an in-depth
    explanation about its usage.

    """

    def __init__(
        self, single_gpu: bool = True, num_nodes_dict={}, idtype=F.int32
    ):
        """
        Constructor for creating a object of instance CuGraphStorage

        See also ``dgl.contrib.cugraph.cugraph_storage_from_heterograph``
        to convert from DGLHeteroGraph to CuGraphStorage

        Parameters
        ----------
         single_gpu: bool
            Wether to create the cugraph object
            on a single GPU or multiple GPUs
            single GPU = True creates PropertyGraph
            single GPU = False creates MGPropertyGraph
         num_nodes_dict: dict[str, int]
            The number of nodes for some node types, which is a
            dictionary mapping a node type T to the number of T-typed nodes.
         idtype: Framework-specific device object,
            The data type for storing the structure-related graph
            information this can be ``torch.int32`` or ``torch.int64``
            for PyTorch.
         Examples
         --------
         The following example uses `CuGraphStorage` :

            >>> from dgl.contrib.cugraph.cugraph_storage import CuGraphStorage
            >>> import cudf
            >>> gs = CuGraphStorage(num_nodes_dict={'drug':3,
                                                    'gene':2,
                                                    'disease':1})
            # add node data
            >>> drug_df = cudf.DataFrame({'node_ids':[0,1,2],
                                          'node_feat':[0.1,0.2,0.3]})
            >>> gs.add_node_data(drug_df, "node_ids", ntype='drug')

            # add edges and edge data
            >>> drug_interacts_drug_df = cudf.DataFrame({'src':[0,1],
                                                         'dst':[1,2],
                                                         'edge_feat':[0.2,0.4]})})
            >>> drug_interacts_gene = cudf.DataFrame({'src':[0,1],
                                                      'dst':[0,1]})
            >>> drug_treats_disease = cudf.DataFrame({'src':[1],
                                                      'dst':[0]})
            >>> gs.add_edge_data(drug_interacts_drug_df,
                                 node_col_names=['src','dst'],
                                 canonical_etype=('drug', 'interacts', 'drug'))
            >>> gs.add_edge_data(drug_interacts_gene,
                                 node_col_names=['src','dst'],
                                 canonical_etype=('drug', 'interacts', 'gene'))
            >>> gs.add_edge_data(drug_treats_disease,
                                 node_col_names=['src','dst'],
                                canonical_etype=('drug', 'treats', 'disease'))
            >>> gs.ntypes
            ['disease', 'drug', 'gene']
            >>> gs.etypes
            ['interacts', 'interacts', 'treats']
            >>> gs.canonical_etypes
            [('drug', 'interacts', 'drug'),
             ('drug', 'interacts', 'gene'),
             ('drug', 'treats', 'disease')]

            >>> gs.sample_neighbors({'disease':[0]},
                                    1)
            Graph(num_nodes={'disease': 1, 'drug': 3, 'gene': 2},
            num_edges={('drug', 'interacts', 'drug'): 0,
                       ('drug', 'interacts', 'gene'): 0,
                       ('drug', 'treats', 'disease'): 1},
            metagraph=[('drug', 'drug', 'interacts'),
                       ('drug', 'gene', 'interacts'),
                       ('drug', 'disease', 'treats')])

            >>> gs.get_node_storage(key='node_feat',
                                    ntype='drug').fetch([0,1,2])
            tensor([0.1000, 0.2000, 0.3000], device='cuda:0',
             dtype=torch.float64)

            >>> es = gs.get_edge_storage(key='edge_feat',
                                    etype=('drug', 'interacts', 'drug'))
            >>> es.fetch([0,1])
            tensor([0.2000, 0.4000], device='cuda:0', dtype=torch.float64)
        """
        # lazy import to prevent creating cuda context
        # till later to help in multiprocessing
        from cugraph.gnn import CuGraphStore
        from cugraph.experimental import PropertyGraph
        from cugraph.experimental import MGPropertyGraph

        if single_gpu:
            pg = PropertyGraph()
        else:
            pg = MGPropertyGraph()

        self.single_gpu = single_gpu

        self.graphstore = CuGraphStore(graph=pg)
        self.idtype = idtype
        self.num_nodes_dict = num_nodes_dict
        self._node_id_offset_d = self.__get_node_id_offset_d(num_nodes_dict)
        # TODO: Potentially expand to set below
        # directly
        self._edge_id_offset_d = None

    def add_node_data(
        self,
        df,
        node_col_name,
        ntype=None,
        feat_name=None,
        contains_vector_features=False,
    ):
        """
        Add a dataframe describing node data to the cugraph graphstore.

        Parameters
        ----------
        df : DataFrame-compatible instance
            A DataFrame instance with a compatible Pandas-like DataFrame
            interface.
        node_col_name : string
            The column name that contains the values to be used as vertex IDs.
        ntype : string
            The node type to be added.
            For example, if dataframe contains data about users, ntype
            might be "users".
            If not specified, the type of properties will be added as
            an empty string.
        feat_name : string or dict
            A map of feature names under which we should save the added
            properties # {"feat_1":[f1, f2], "feat_2":[f3, f4]}
            (ignored if contains_vector_features=False and the col names of
            the dataframe are treated as corresponding feature names)
        contains_vector_features : True
            Wether to treat the columns of the dataframe being added as
            as 2d features
        Returns
        -------
        None
        """
        if ntype:
            df[node_col_name] = df[node_col_name] + self.get_node_id_offset(
                ntype
            )
        self.graphstore.add_node_data(
            df=df,
            node_col_name=node_col_name,
            ntype=ntype,
            feat_name=feat_name,
            contains_vector_features=contains_vector_features,
        )

        # This will delete cached value
        if "total_number_of_nodes" in self.__dict__:
            del self.total_number_of_nodes

    def add_edge_data(
        self,
        df,
        node_col_names,
        canonical_etype=None,
        feat_name=None,
        contains_vector_features=False,
    ):
        """
        Add a dataframe describing node data to the cugraph graphstore.

        Parameters
        ----------
        df : DataFrame-compatible instance
            A DataFrame instance with a compatible Pandas-like DataFrame
            interface.
        node_col_names : [src_name, dst_name]
            The column names that corresponding to the src_name and dst_name
        canonical_etype : Tuple[(str, str, str)]
            The edge type to be added
        feat_name : string or dict
            A map of feature names under which we should save the added
            properties # {"feat_1":[f1, f2], "feat_2":[f3, f4]}
            (ignored if contains_vector_features=False and the col names of
            the dataframe are treated as corresponding feature names)
        contains_vector_features : True
            Wether to treat the columns of the dataframe being added as
            as 2d features
        Returns
        -------
        None
        """
        if canonical_etype:
            _assert_valid_canonical_etype(canonical_etype)
            src_n, dst_n = node_col_names
            src_type, dst_type = canonical_etype[0], canonical_etype[2]
            df[src_n] = df[src_n] + self.get_node_id_offset(src_type)
            df[dst_n] = df[dst_n] + self.get_node_id_offset(dst_type)
        # Convert to a string because cugraph PG does not support tuple objects
        canonical_etype = str(canonical_etype)
        self.graphstore.add_edge_data(
            df=df,
            node_col_names=node_col_names,
            canonical_etype=canonical_etype,
            feat_name=feat_name,
            contains_vector_features=contains_vector_features,
        )

    # Sampling Function
    def sample_neighbors(
        self,
        nodes,
        fanout,
        edge_dir="in",
        prob=None,
        exclude_edges=None,
        replace=False,
        output_device=None,
    ):
        """
        Return a DGLGraph which is a subgraph induced by sampling neighboring
        edges of the given nodes.
        See ``dgl.sampling.sample_neighbors`` for detailed semantics.
        Parameters
        ----------
        nodes : Tensor or dict[str, Tensor]
            Node IDs to sample neighbors from.
            This argument can take a single ID tensor or a dictionary of node
            types and ID tensors. If a single tensor is given, the graph must
            only have one type of nodes.
        fanout : int or dict[etype, int]
            The number of edges to be sampled for each node on each edge type.
            This argument can take a single int or a dictionary of edge types
            and ints. If a single int is given, DGL will sample this number of
            edges for each node for every edge type.
            If -1 is given for a single edge type, all the neighboring edges
            with that edge type will be selected.
        prob : str, optional
            Feature name used as the (unnormalized) probabilities associated
            with each neighboring edge of a node.  The feature must have only
            one element for each edge.
            The features must be non-negative floats, and the sum of the
            features of inbound/outbound edges for every node must be positive
            (though they don't have to sum up to one).  Otherwise, the result
            will be undefined. If :attr:`prob` is not None, GPU sampling is
            not supported.
        exclude_edges: tensor or dict
            Edge IDs to exclude during sampling neighbors for the seed nodes.
            This argument can take a single ID tensor or a dictionary of edge
            types and ID tensors. If a single tensor is given, the graph must
            only have one type of nodes.
        replace : bool, optional
            If True, sample with replacement.
        output_device : Framework-specific device context object, optional
            The output device.  Default is the same as the input graph.
        Returns
        -------
        DGLGraph
            A sampled subgraph with the same nodes as the original graph, but
            only the sampled neighboring edges.  The induced edge IDs will be
            in ``edata[dgl.EID]``.
        """

        if prob is not None:
            raise NotImplementedError(
                "prob is not currently supported",
                " for sample_neighbors in CuGraphStorage",
            )

        if exclude_edges is not None:
            raise NotImplementedError(
                "exclude_edges is not currently supported",
                " for sample_neighbors in CuGraphStorage",
            )

        if not isinstance(nodes, dict):
            if len(self.ntypes) > 1:
                raise dgl.DGLError(
                    "Must specify node type when the graph is not homogeneous."
                )
            nodes = F.tensor(nodes)
            nodes_cap = F.zerocopy_to_dlpack(nodes)
        else:
            nodes = {
                k: self.dgl_n_id_to_cugraph_id(F.tensor(n), k)
                for k, n in nodes.items()
            }
            nodes_cap = {
                k: F.zerocopy_to_dlpack(F.tensor(n)) for k, n in nodes.items()
            }

        sample_cap_obj = self.graphstore.sample_neighbors(
            nodes_cap,
            fanout,
            edge_dir=edge_dir,
            prob=prob,
            replace=replace,
        )
        # heterograph case
        if len(self.etypes) > 1:
            graph_data_d, graph_eid_d = self.__convert_pycap_to_dgl_tensor_d(
                sample_cap_obj, self.idtype
            )
            sampled_graph = dgl.heterograph(
                data_dict=graph_data_d,
                num_nodes_dict=self.num_nodes_dict,
                idtype=self.idtype,
            )
            sampled_graph.edata[dgl.EID] = graph_eid_d
        else:
            src_c, dst_c, edge_id_c = sample_cap_obj
            src_ids = F.zerocopy_from_dlpack(src_c)
            dst_ids = F.zerocopy_from_dlpack(dst_c)
            edge_id_t = F.zerocopy_from_dlpack(edge_id_c)
            sampled_graph = dgl.graph(
                (src_ids, dst_ids),
                num_nodes=self.total_number_of_nodes,
                idtype=self.idtype,
            )
            sampled_graph.edata[dgl.EID] = edge_id_t

        # to device function move the dgl graph to desired devices
        if output_device is not None:
            sampled_graph.to(output_device)
        return sampled_graph

    # Required in Cluster-GCN
    def subgraph(self, nodes, relabel_nodes=False, output_device=None):
        """Return a subgraph induced on given nodes.
        This has the same semantics as ``dgl.node_subgraph``.
        Parameters
        ----------
        nodes : nodes or dict[str, nodes]
            The nodes to form the subgraph. The allowed nodes formats are:
            * Int Tensor: Each element is a node ID. The tensor must have the
             same device type and ID data type as the graph's.
            * iterable[int]: Each element is a node ID.
            * Bool Tensor: Each :math:`i^{th}` element is a bool flag
             indicating whether node :math:`i` is in the subgraph.
             If the graph is homogeneous, directly pass the above formats.
             Otherwise, the argument must be a dictionary with keys being
             node types and values being the node IDs in the above formats.
        relabel_nodes : bool, optional
            If True, the extracted subgraph will only have the nodes in the
            specified node set and it will relabel the nodes in order.
        output_device : Framework-specific device context object, optional
            The output device.  Default is the same as the input graph.
        Returns
        -------
        DGLGraph
            The subgraph.
        """
        raise NotImplementedError("subgraph is not implemented yet")

    # Required in Link Prediction
    # relabel = F we use dgl functions,
    # relabel = T, we need to delete nodes and relabel
    def edge_subgraph(self, edges, relabel_nodes=False, output_device=None):
        """
        Return a subgraph induced on given edges.
        This has the same semantics as ``dgl.edge_subgraph``.
        Parameters
        ----------
        edges : edges or dict[(str, str, str), edges]
            The edges to form the subgraph. The allowed edges formats are:
            * Int Tensor: Each element is an edge ID. The tensor must have the
              same device type and ID data type as the graph's.
            * iterable[int]: Each element is an edge ID.
            * Bool Tensor: Each :math:`i^{th}` element is a bool flag
             indicating whether edge :math:`i` is in the subgraph.
            If the graph is homogeneous, one can directly pass the above
            formats. Otherwise, the argument must be a dictionary with keys
            being edge types and values being the edge IDs in the above formats
        relabel_nodes : bool, optional
            If True, the extracted subgraph will only have the nodes in the
            specified node set and it will relabel the nodes in order.
        output_device : Framework-specific device context object, optional
            The output device.  Default is the same as the input graph.
        Returns
        -------
        DGLGraph
            The subgraph.
        """
        raise NotImplementedError("edge_subgraph is not implemented yet")

    # Required in Link Prediction negative sampler
    def find_edges(self, eid, etype=None, output_device=None):
        """
        Return the source and destination node ID(s) given the edge ID(s).

        Parameters
        ----------
        eid : edge ID(s)
            The edge IDs. The allowed formats are:

            * ``int``: A single ID.
            * Int Tensor: Each element is an ID.
            The tensor must have the same device type
            and ID data type as the graph's.
            * iterable[int]: Each element is an ID.

        etype : str
            The type names of the edges.
            Can be omitted if the graph has only one type of edges.

        Returns
        -------
        Tensor
            The source node IDs of the edges.
            The i-th element is the source node ID of the i-th edge.
        Tensor
            The destination node IDs of the edges.
            The i-th element is the destination node ID of the i-th edge.
        """
        src_type, connection_type, dst_type = etype
        eid = self.dgl_e_id_to_cugraph_id(eid)
        eid_cap = F.zerocopy_to_dlpack(eid)
        # Because we converted to dlpack so variable eid no longer
        # Owns the tensor so we free it up
        del eid
        etype_s = str(etype)
        src_cap, dst_cap = self.graphstore.find_edges(eid_cap, etype_s)
        src_nodes_tensor = F.zerocopy_from_dlpack(src_cap).to(output_device)
        dst_nodes_tensor = F.zerocopy_from_dlpack(dst_cap).to(output_device)

        src_nodes_tensor = self.cugraph_n_id_to_dgl_id(
            src_nodes_tensor, src_type
        )
        dst_nodes_tensor = self.cugraph_n_id_to_dgl_id(
            dst_nodes_tensor, dst_type
        )

        return src_nodes_tensor, dst_nodes_tensor

    # Required in Link Prediction negative sampler
    def global_uniform_negative_sampling(
        self, num_samples, exclude_self_loops=True, replace=False, etype=None
    ):
        """
        Per source negative sampling as in ``dgl.dataloading.GlobalUniform``
        """
        raise NotImplementedError(
            "global_uniform_negative_sampling not implemented yet"
        )

    def get_node_storage(self, key, ntype=None):
        """
        Get storage object of node feature of
        type :attr:`ntype` and name :attr:`key`
        """
        if len(self.ntypes) > 1:
            indices_offset = self.get_node_id_offset(ntype)
        else:
            indices_offset = 0

        return self.graphstore.get_node_storage(key, ntype, indices_offset)

    def get_edge_storage(self, key, etype=None):
        """
        Get storage object of edge feature of
        type :attr:`ntype` and name :attr:`key`
        """
        if len(self.canonical_etypes) > 1:
            indices_offset = self.get_edge_id_offset(etype)
        else:
            indices_offset = 0

        return self.graphstore.get_edge_storage(key, etype, indices_offset)

    # Number of edges/nodes utils
    def num_nodes(self, ntype=None):
        """
        Return the number of nodes in the graph.
        Parameters
        ----------
        ntype : str, optional
            The node type name. If given, it returns the number of nodes of the
            type.
            If not given (default), it  returns the total number of nodes
            of all types.

        Returns
        -------
        int
            The number of nodes.
        """
        if ntype:
            return self.num_nodes_dict[ntype]
        else:
            return self.graphstore.num_nodes(ntype)

    def number_of_nodes(self, ntype=None):
        """
        Return the number of nodes in the graph.
        Alias of ``num_nodes``
        Parameters
        ----------
        ntype : str, optional
            The node type name. If given, it returns the number of nodes of the
            type.
            If not given (default), it  returns the total number of nodes
            of all types.

        Returns
        -------
        int
            The number of nodes.
        """
        return self.num_nodes(ntype)

    @property
    def ntypes(self):
        """
        Return all the node type names in the graph.

        Returns
        -------
        list[str]
            All the node type names in a list.
        """
        return self.graphstore.ntypes

    @property
    def etypes(self):
        """
        Return all the edge type names in the graph.

        Returns
        -------
        list[str]
            All the edge type names in a list.
        """

        return [can_etype[1] for can_etype in self.canonical_etypes]

    def num_edges(self, etype=None):
        """
        Return the number of edges in the graph.
        Parameters
        ----------
        etype:

        Returns
        -------
        int
            The number of edges
        """
        # use graphstore function
        if etype:
            if etype not in self.canonical_etypes:
                etype = self.get_corresponding_canonical_etype(etype)
            etype = str(etype)

        return self.graphstore.num_edges(etype)

    # Node Properties
    @property
    def total_number_of_nodes(self):
        if not hasattr(self, '_total_number_of_nodes'):
            self.__total_number_of_nodes = self.num_nodes()
        return self.__total_number_of_nodes

    @property
    def num_canonical_edges_dict(self):
        return self.graphstore.num_edges_dict

    @property
    def canonical_etypes(self):
        can_etypes = self.graphstore.etypes
        return [convert_can_etype_s_to_tup(s) for s in can_etypes]

    @property
    def device(self):
        """
        Get the device of the graph.
        Returns
        -------
        device context
            The device of the graph, which should be a
            framework-specific device object (e.g., ``torch.device``).
        """
        import torch

        return torch.cuda.current_device()

    # Index Conversion Utils
    def get_node_id_offset(self, ntype: str) -> int:
        """
        Return the integer offset for node id of type ntype
        """
        if self._node_id_offset_d is None:
            self._node_id_offset_d = self.__get_node_id_offset_d(
                self.num_nodes_dict
            )
        return self._node_id_offset_d[ntype]

    def get_edge_id_offset(self, canonical_etype: tuple[str, str, str]) -> int:
        """
        Return the integer offset for node id of type etype
        """
        _assert_valid_canonical_etype(canonical_etype)
        canonical_etype = str(canonical_etype)
        if self._edge_id_offset_d is None:
            self._edge_id_offset_d = self.__get_edge_id_offset_d(
                self.num_canonical_edges_dict
            )

        return self._edge_id_offset_d[canonical_etype]

    def dgl_n_id_to_cugraph_id(self, index_t, ntype):
        return index_t + self.get_node_id_offset(ntype)

    def cugraph_n_id_to_dgl_id(self, index_t, ntype):
        return index_t - self.get_node_id_offset(ntype)

    def dgl_e_id_to_cugraph_id(self, index_t, canonical_etype):
        return index_t + self.get_edge_id_offset(canonical_etype)

    def cugraph_e_id_to_dgl_id(self, index_t, canonical_etype):
        return index_t - self.get_edge_id_offset(canonical_etype)

    # Methods for getting the offsets per type
    @staticmethod
    def __get_edge_id_offset_d(num_canonical_edges_dict):
        # dict for edge_id_offset_start
        last_st = 0
        edge_ind_st_d = {}
        for etype in sorted(num_canonical_edges_dict.keys()):
            edge_ind_st_d[etype] = last_st
            last_st = last_st + num_canonical_edges_dict[etype]
        return edge_ind_st_d

    @staticmethod
    def __get_node_id_offset_d(num_nodes_dict):
        # dict for node_id_offset_start
        last_st = 0
        node_ind_st_d = {}
        for ntype in sorted(num_nodes_dict.keys()):
            node_ind_st_d[ntype] = last_st
            last_st = last_st + num_nodes_dict[ntype]
        return node_ind_st_d

    def get_corresponding_canonical_etype(self, etype: str) -> str:
        can_etypes = [
            can_etype
            for can_etype in self.canonical_etypes
            if can_etype[1] == etype
        ]
        if len(can_etypes) > 1:
            raise dgl.DGLError(
                f'Edge type "{etype}" is ambiguous. Please use canonical'
                + "edge type in the form of (srctype, etype, dsttype)"
            )
        return can_etypes[0]

    def __convert_pycap_to_dgl_tensor_d(
        self, graph_data_cap_d, o_dtype=F.int64
    ):
        graph_data_d = {}
        graph_eid_d = {}
        for canonical_etype_s, (
            src_cap,
            dst_cap,
            edge_id_cap,
        ) in graph_data_cap_d.items():

            canonical_etype = convert_can_etype_s_to_tup(canonical_etype_s)
            src_type = canonical_etype[0]
            dst_type = canonical_etype[2]
            if src_cap is None:
                src_t = F.tensor(data=[])
                dst_t = F.tensor(data=[])
                edge_id_t = F.tensor(data=[])
            else:
                src_t = F.zerocopy_from_dlpack(src_cap)
                dst_t = F.zerocopy_from_dlpack(dst_cap)
                edge_id_t = F.zerocopy_from_dlpack(edge_id_cap)

                src_t = self.cugraph_n_id_to_dgl_id(src_t, src_type)
                dst_t = self.cugraph_n_id_to_dgl_id(dst_t, dst_type)
                edge_id_t = self.cugraph_e_id_to_dgl_id(
                    edge_id_t, canonical_etype
                )

            graph_data_d[canonical_etype] = (
                src_t.to(o_dtype).to("cuda"),
                dst_t.to(o_dtype).to("cuda"),
            )
            graph_eid_d[canonical_etype] = edge_id_t.to(o_dtype).to("cuda")

        return graph_data_d, graph_eid_d
