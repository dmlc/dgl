##
#   Copyright 2019-2021 Contributors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

import backend as F

import dgl
import dgl.partition
from utils import parametrize_idtype


@parametrize_idtype
def test_to_block(idtype):
    def check(g, bg, ntype, etype, dst_nodes, include_dst_in_src=True):
        if dst_nodes is not None:
            assert F.array_equal(bg.dstnodes[ntype].data[dgl.NID], dst_nodes)
        n_dst_nodes = bg.num_nodes("DST/" + ntype)
        if include_dst_in_src:
            assert F.array_equal(
                bg.srcnodes[ntype].data[dgl.NID][:n_dst_nodes],
                bg.dstnodes[ntype].data[dgl.NID],
            )

        g = g[etype]
        bg = bg[etype]
        induced_src = bg.srcdata[dgl.NID]
        induced_dst = bg.dstdata[dgl.NID]
        induced_eid = bg.edata[dgl.EID]

        bg_src, bg_dst = bg.all_edges(order="eid")
        src_ans, dst_ans = g.all_edges(order="eid")

        induced_src_bg = F.gather_row(induced_src, bg_src)
        induced_dst_bg = F.gather_row(induced_dst, bg_dst)
        induced_src_ans = F.gather_row(src_ans, induced_eid)
        induced_dst_ans = F.gather_row(dst_ans, induced_eid)

        assert F.array_equal(induced_src_bg, induced_src_ans)
        assert F.array_equal(induced_dst_bg, induced_dst_ans)

    def checkall(g, bg, dst_nodes, include_dst_in_src=True):
        for etype in g.etypes:
            ntype = g.to_canonical_etype(etype)[2]
            if dst_nodes is not None and ntype in dst_nodes:
                check(g, bg, ntype, etype, dst_nodes[ntype], include_dst_in_src)
            else:
                check(g, bg, ntype, etype, None, include_dst_in_src)

    # homogeneous graph
    g = dgl.graph(
        (F.tensor([1, 2], dtype=idtype), F.tensor([2, 3], dtype=idtype))
    )
    dst_nodes = F.tensor([3, 2], dtype=idtype)
    bg = dgl.to_block(g, dst_nodes=dst_nodes)
    check(g, bg, "_N", "_E", dst_nodes)

    src_nodes = bg.srcnodes["_N"].data[dgl.NID]
    bg = dgl.to_block(g, dst_nodes=dst_nodes, src_nodes=src_nodes)
    check(g, bg, "_N", "_E", dst_nodes)

    # heterogeneous graph
    g = dgl.heterograph(
        {
            ("A", "AA", "A"): ([0, 2, 1, 3], [1, 3, 2, 4]),
            ("A", "AB", "B"): ([0, 1, 3, 1], [1, 3, 5, 6]),
            ("B", "BA", "A"): ([2, 3], [3, 2]),
        },
        idtype=idtype,
        device=F.ctx(),
    )
    g.nodes["A"].data["x"] = F.randn((5, 10))
    g.nodes["B"].data["x"] = F.randn((7, 5))
    g.edges["AA"].data["x"] = F.randn((4, 3))
    g.edges["AB"].data["x"] = F.randn((4, 3))
    g.edges["BA"].data["x"] = F.randn((2, 3))
    g_a = g["AA"]

    def check_features(g, bg):
        for ntype in bg.srctypes:
            for key in g.nodes[ntype].data:
                assert F.array_equal(
                    bg.srcnodes[ntype].data[key],
                    F.gather_row(
                        g.nodes[ntype].data[key],
                        bg.srcnodes[ntype].data[dgl.NID],
                    ),
                )
        for ntype in bg.dsttypes:
            for key in g.nodes[ntype].data:
                assert F.array_equal(
                    bg.dstnodes[ntype].data[key],
                    F.gather_row(
                        g.nodes[ntype].data[key],
                        bg.dstnodes[ntype].data[dgl.NID],
                    ),
                )
        for etype in bg.canonical_etypes:
            for key in g.edges[etype].data:
                assert F.array_equal(
                    bg.edges[etype].data[key],
                    F.gather_row(
                        g.edges[etype].data[key], bg.edges[etype].data[dgl.EID]
                    ),
                )

    bg = dgl.to_block(g_a)
    check(g_a, bg, "A", "AA", None)
    check_features(g_a, bg)
    assert bg.number_of_src_nodes() == 5
    assert bg.number_of_dst_nodes() == 4

    bg = dgl.to_block(g_a, include_dst_in_src=False)
    check(g_a, bg, "A", "AA", None, False)
    check_features(g_a, bg)
    assert bg.number_of_src_nodes() == 4
    assert bg.number_of_dst_nodes() == 4

    dst_nodes = F.tensor([4, 3, 2, 1], dtype=idtype)
    bg = dgl.to_block(g_a, dst_nodes)
    check(g_a, bg, "A", "AA", dst_nodes)
    check_features(g_a, bg)

    g_ab = g["AB"]

    bg = dgl.to_block(g_ab)
    assert bg.idtype == idtype
    assert bg.num_nodes("SRC/B") == 4
    assert F.array_equal(
        bg.srcnodes["B"].data[dgl.NID], bg.dstnodes["B"].data[dgl.NID]
    )
    assert bg.num_nodes("DST/A") == 0
    checkall(g_ab, bg, None)
    check_features(g_ab, bg)

    dst_nodes = {"B": F.tensor([5, 6, 3, 1], dtype=idtype)}
    bg = dgl.to_block(g, dst_nodes)
    assert bg.num_nodes("SRC/B") == 4
    assert F.array_equal(
        bg.srcnodes["B"].data[dgl.NID], bg.dstnodes["B"].data[dgl.NID]
    )
    assert bg.num_nodes("DST/A") == 0
    checkall(g, bg, dst_nodes)
    check_features(g, bg)

    dst_nodes = {
        "A": F.tensor([4, 3, 2, 1], dtype=idtype),
        "B": F.tensor([3, 5, 6, 1], dtype=idtype),
    }
    bg = dgl.to_block(g, dst_nodes=dst_nodes)
    checkall(g, bg, dst_nodes)
    check_features(g, bg)

    # test specifying lhs_nodes with include_dst_in_src
    src_nodes = {}
    for ntype in dst_nodes.keys():
        # use the previous run to get the list of source nodes
        src_nodes[ntype] = bg.srcnodes[ntype].data[dgl.NID]
    bg = dgl.to_block(g, dst_nodes=dst_nodes, src_nodes=src_nodes)
    checkall(g, bg, dst_nodes)
    check_features(g, bg)

    # test without include_dst_in_src
    dst_nodes = {
        "A": F.tensor([4, 3, 2, 1], dtype=idtype),
        "B": F.tensor([3, 5, 6, 1], dtype=idtype),
    }
    bg = dgl.to_block(g, dst_nodes=dst_nodes, include_dst_in_src=False)
    checkall(g, bg, dst_nodes, False)
    check_features(g, bg)

    # test specifying lhs_nodes without include_dst_in_src
    src_nodes = {}
    for ntype in dst_nodes.keys():
        # use the previous run to get the list of source nodes
        src_nodes[ntype] = bg.srcnodes[ntype].data[dgl.NID]
    bg = dgl.to_block(
        g, dst_nodes=dst_nodes, include_dst_in_src=False, src_nodes=src_nodes
    )
    checkall(g, bg, dst_nodes, False)
    check_features(g, bg)
