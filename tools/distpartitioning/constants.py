GLOBAL_NID = "global_node_id"
GLOBAL_EID = "global_edge_id"

SHUFFLE_GLOBAL_NID = "shuffle_global_node_id"
SHUFFLE_GLOBAL_EID = "shuffle_global_edge_id"

NTYPE_ID = "node_type_id"
ETYPE_ID = "edge_type_id"

GLOBAL_TYPE_NID = "global_type_node_id"
GLOBAL_TYPE_EID = "global_type_edge_id"

GLOBAL_SRC_ID = "global_src_id"
GLOBAL_DST_ID = "global_dst_id"
SHUFFLE_GLOBAL_SRC_ID = "shuffle_global_src_id"
SHUFFLE_GLOBAL_DST_ID = "shuffle_global_dst_id"

OWNER_PROCESS = "owner_proc_id"

PART_LOCAL_NID = "part_local_nid"

#CI-Test related definitions below this line
CI_GRAPH_NUM_NODES = 40

'''
Following graph is used for CI-test 
NodeTypes: ntype-1, ntype-2
EdgeTypes: etype-1, etype-2, etype-3

ntype-1
-------
    Total Nodes : 20
    Rank-0 generates 0-9 (inclusive, both ends)
    Rank-1 generates 10-19 (inclusive, both ends)

ntype-2
-------
    Total Nodes: 20
    Rank-0 generates 0-9 (inclusive, both ends)
    Rank-1 generates 10-19 (inclusive, both ends)

etype-1
-------
    Edges among `ntype-1` nodes, where the source and destination end points
    are of type `ntype-1`

    Total edges : 20
    Rank-0 generates 0-9 (inclusive, both ends)
    Rank-1 generates 10-19 (inclusive, both ends)

    Edges are randomly generated on each rank. 

etype-2
-------
    Edges among `ntype-2` nodes, where the source and destination end points
    are of type `ntype-2`

    Total edges : 20
    Rank-0 generates 0-9 (inclusive, both ends)
    Rank-1 generates 10-19 (inclusive, both ends)

    Edges are randomly generated on each rank. 

etype-3
-------
    Edges between `ntype-1` and `ntype-2` nodes. Here source end point belongs to 
    `ntype-1` and destination end point belongs to `ntype-2`

    Total edges : 20
    Rank-0 generates 0-9 (inclusive, both ends)
    Rank-1 generates 10-19 (inclusive, both ends)

    Edges are randomly generated on each rank. 

Node Features: 
--------------
    Nodes of type `ntype-1` has two node features, which are nodes-feat-01 and nodes-feat-02. 
    These are of dimensions (10, 20) on each rank. 

    Rank-0 generates its data for nodes-feat-01 as np.ones((10, 20)) * (rank + 1)
    Rank-0 generates its data for nodes-feat-02 as np.ones((10, 20)) * (rank + 1)

    Rank-1 generates its data for nodes-feat-01 as np.ones((10, 20)) * (rank + 1)*10
    Rank-1 generates its data for nodes-feat-02 as np.ones((10, 20)) * (rank + 1)*10
'''

CI_JSON_STRING = '{ "nid": { "ntype-1": { "format": "csv", "data": [ ["", "0", "10" ], ["", "10", "20" ] ] }, "ntype-2": { "format": "csv", "data": [ ["", "0", "10" ], ["", "10", "20" ] ] } }, "eid": { "etype-1": { "format": "csv", "data": [ ["", "0", "10" ], [ "", "10", "20" ] ] }, "etype-2": { "format": "csv", "data": [ ["", "0", "10" ], ["", "10", "20" ] ] }, "etype-3": { "format": "csv", "data": [ ["", "0", "10" ], ["", "10", "20" ] ] } }, "node_data": { "ntype-1": { "nodes-feat-01": [ ["", "0", "10" ], ["", "10", "20" ] ], "nodes-feat-02": [ ["", "0", "10" ], ["", "10", "20" ] ] } } }'
