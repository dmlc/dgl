# Copyright (c) 2022, NVIDIA CORPORATION.
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

# Utils to convert b/w dgl heterograph to cugraph GraphStore
# TODO: Add upstream
from typing import Optional
from collections import defaultdict

import cudf
import cupy as cp
import dgl
import torch
from dgl.backend import zerocopy_to_dlpack


# Feature Tensor to DataFrame Utils
def convert_to_column_major(t: torch.Tensor):
    return t.t().contiguous().t()


def create_feature_frame(feat_t_d: dict[str, torch.Tensor]) -> cudf.DataFrame:
    """
    Convert a feature_tensor_d to a dataframe
    """
    df_ls = []
    feat_name_map = {}
    for feat_key, feat_t in feat_t_d.items():
        feat_t = feat_t.to("cuda")
        feat_t = convert_to_column_major(feat_t)
        ar = cp.from_dlpack(zerocopy_to_dlpack(feat_t))
        del feat_t
        df = cudf.DataFrame(ar)
        feat_columns = [f"{feat_key}_{i}" for i in range(len(df.columns))]
        df.columns = feat_columns
        feat_name_map[feat_key] = feat_columns
        df_ls.append(df)

    df = cudf.concat(df_ls, axis=1)
    return df, feat_name_map


# Add ndata utils
def add_ndata_of_single_type(
    gs: dgl.contrib.cugraph.CuGraphStorage,
    feat_t_d: Optional[dict[torch.Tensor]],
    ntype: str,
    n_rows: int,
    node_offset: int,
    idtype=torch.int64,
):
    node_ids = dgl.backend.arange(0, n_rows, idtype, ctx="cuda") + node_offset
    node_ids = cp.from_dlpack(zerocopy_to_dlpack(node_ids))

    if feat_t_d:
        df, feat_name_map = create_feature_frame(feat_t_d)
        df["node_id"] = node_ids
        gs.add_node_data(
            df,
            "node_id",
            feat_name_map,
            ntype=ntype,
            contains_vector_features=True,
        )
    else:
        df = cudf.DataFrame()
        df["node_id"] = node_ids
        gs.add_node_data(
            df, "node_id", ntype=ntype, contains_vector_features=False
        )
    return gs


def add_nodes_from_dgl_heteroGraph(
    gs: dgl.contrib.cugraph.CuGraphStorage,
    graph: dgl.DGLHeteroGraph,
    num_nodes_dict: Optional[dict[str, int]] = None,
):
    if len(graph.ntypes) > 1:
        if num_nodes_dict is None:
            raise ValueError(
                "num_nodes_dict must be provided for adding ndata"
                "from Heterogeneous Graphs"
            )
        node_id_offset_d = gs._CuGraphStorage__get_node_id_offset_d(
            num_nodes_dict
        )
        ntype_feat_d = dict()
        for feat_name in graph.ndata.keys():
            for ntype in graph.ndata[feat_name]:
                if ntype not in ntype_feat_d:
                    ntype_feat_d[ntype] = {}
                ntype_feat_d[ntype][feat_name] = graph.ndata[feat_name][ntype]

        for ntype in num_nodes_dict.keys():
            node_offset = node_id_offset_d[ntype]
            feat_t_d = ntype_feat_d.get(ntype, None)
            gs = add_ndata_of_single_type(
                gs=gs,
                feat_t_d=feat_t_d,
                ntype=ntype,
                n_rows=num_nodes_dict[ntype],
                node_offset=node_offset,
                idtype=graph.idtype,
            )
    else:
        ntype = graph.ntypes[0]
        ntype_feat_d = dict()
        gs = add_ndata_of_single_type(
            gs,
            feat_t_d=graph.ndata,
            ntype=ntype,
            n_rows=graph.number_of_nodes(),
            node_offset=0,
            idtype=graph.idtype,
        )
    return gs


# Add edata utils
def add_edata_of_single_type(
    gs: dgl.contrib.cugraph.CuGraphStorage,
    feat_t_d: Optional[dict[torch.Tensor]],
    src_t: torch.Tensor,
    dst_t: torch.Tensor,
    can_etype: tuple([str, str, str]),
    src_offset: int,
    dst_offset: int,
):

    src_t = src_t.to("cuda")
    dst_t = dst_t.to("cuda")

    src_t = src_t + src_offset
    dst_t = dst_t + dst_offset

    df = cudf.DataFrame(
        {
            "src": cudf.from_dlpack(zerocopy_to_dlpack(src_t)),
            "dst": cudf.from_dlpack(zerocopy_to_dlpack(dst_t)),
        }
    )
    if feat_t_d:
        feat_df, feat_name_map = create_feature_frame(feat_t_d)
        df = cudf.concat([df, feat_df], axis=1)
        gs.add_edge_data(
            df,
            ["src", "dst"],
            feat_name_map,
            canonical_etype=can_etype,
            contains_vector_features=True,
        )
    else:
        gs.add_edge_data(
            df,
            ["src", "dst"],
            canonical_etype=can_etype,
            contains_vector_features=False,
        )
    return gs


def add_edges_from_dgl_heteroGraph(
    gs: dgl.contrib.cugraph.CuGraphStorage,
    graph: dgl.DGLHeteroGraph,
    num_nodes_dict: Optional[dict[str, int]] = None,
):
    etype_feat_d = dict()
    for feat_name in graph.edata.keys():
        for etype in graph.edata[feat_name].keys():
            if etype not in etype_feat_d:
                etype_feat_d[etype] = {}
            etype_feat_d[etype][feat_name] = graph.edata[feat_name][etype]

    if len(graph.ntypes) > 1:
        if num_nodes_dict is None:
            raise ValueError(
                "num_nodes_dict must be provided for adding edges from HeteroGraphs"
            )
        node_id_offset_d = gs._CuGraphStorage__get_node_id_offset_d(
            num_nodes_dict
        )
    else:
        node_id_offset_d = None

    for can_etype in graph.canonical_etypes:
        src_t, dst_t = graph.edges(form="uv", etype=can_etype)
        src_type, _, dst_type = can_etype
        if node_id_offset_d:
            src_offset, dst_offset = (
                node_id_offset_d[src_type],
                node_id_offset_d[dst_type],
            )
        else:
            src_offset, dst_offset = 0, 0
        feat_t_d = etype_feat_d.get(can_etype, None)
        add_edata_of_single_type(
            gs, feat_t_d, src_t, dst_t, can_etype, src_offset, dst_offset
        )
