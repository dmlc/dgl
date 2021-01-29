import os
import json
import argparse
import numpy as np
import dgl
import torch as th

parser = argparse.ArgumentParser(description='Construct graph partitions')
parser.add_argument('--input-dir', required=True, type=str,
                    help='The directory path that contains the partition results.')
parser.add_argument('--graph-name', required=True, type=str,
                    help='The graph name')
parser.add_argument('--schema', required=True, type=str,
                    help='The schema of the graph')
parser.add_argument('--num-parts', required=True, type=int,
                    help='The number of partitions')
parser.add_argument('--num-ntypes', type=int, required=True,
                    help='The number of node types in the graph.')
parser.add_argument('--num-node-weights', required=True, type=int,
                    help='The number of node weights used by METIS.')
parser.add_argument('--workspace', type=str, default='/tmp',
                    help='The directory to store the intermediate results')
parser.add_argument('--output', required=True, type=str,
                    help='The output directory of the partitioned results')
args = parser.parse_args()

input_dir = args.input_dir
graph_name = args.graph_name
num_ntypes = args.num_ntypes
num_parts = args.num_parts
num_node_weights = args.num_node_weights
node_attr_dtype = np.float32
edge_attr_dtype = np.int64
workspace_dir = args.workspace
output_dir = args.output

with open(args.schema) as json_file:
    schema = json.load(json_file)
nid_ranges = schema['nid']
eid_ranges = schema['eid']
nid_ranges = {key: np.array(nid_ranges[key]).reshape(1, 2) for key in nid_ranges}
eid_ranges = {key: np.array(eid_ranges[key]).reshape(1, 2) for key in eid_ranges}
id_map = dgl.distributed.id_map.IdMap(nid_ranges)

ntypes = [(key, nid_ranges[key][0,0]) for key in nid_ranges]
ntypes.sort(key=lambda e: e[1])
ntypes = [e[0] for e in ntypes]
ntypes_map = {e:i for i, e in enumerate(ntypes)}
etypes = [(key, eid_ranges[key][0,0]) for key in eid_ranges]
etypes.sort(key=lambda e: e[1])
etypes = [e[0] for e in etypes]
etypes_map = {e:i for i, e in enumerate(etypes)}

num_edges = 0
num_nodes = 0
node_map_val = {ntype:[] for ntype in ntypes}
edge_map_val = {etype:[] for etype in etypes}
for part_id in range(num_parts):
    node_file = 'p{:03}-{}_nodes.txt'.format(part_id, graph_name)
    # The format of each line in the node file:
    # <node_id> <ndoe_type> <weight1> ... <orig_node_id> <attributes>
    # The node file contains nodes that belong to a partition. It doesn't include HALO nodes.
    orig_nid_col = 3 + num_node_weights
    first_attr_col = 4 + num_node_weights

    # Get the first two columns which is the node ID and node type.
    tmp_output = workspace_dir + '/' + node_file + '.tmp'
    os.system('awk \'{print $1, $2, $' + str(orig_nid_col) + '}\''
              + ' {} > {}'.format(input_dir + '/' + node_file, tmp_output))
    nodes = np.loadtxt(tmp_output, dtype=np.int64)
    nids, ntype_ids, orig_nid = nodes[:,0], nodes[:,1], nodes[:,2]
    assert np.all(nids[1:] - nids[:-1] == 1)
    nid_range = (nids[0], nids[-1])
    num_nodes += len(nodes)

    # Get node attributes
    # Here we just assume all nodes have the same attributes.
    # In practice, this is not the same, in which we need more complex solution to
    # encode and decode node attributes.
    os.system('cut -d\' \' -f {}- {} > {}'.format(first_attr_col,
                                                  input_dir + '/' + node_file,
                                                  tmp_output))
    node_attrs = np.loadtxt(tmp_output, dtype=node_attr_dtype)
    node_feats = {}
    # nodes in a partition has been sorted based on node types.
    for ntype_name in nid_ranges:
        ntype_id = ntypes_map[ntype_name]
        type_nids = nids[ntype_ids == ntype_id]
        assert np.all(type_nids == np.arange(type_nids[0], type_nids[-1] + 1))
        node_map_val[ntype_name].append([int(type_nids[0]), int(type_nids[-1]) + 1])
        node_feats[ntype_name + '/feat'] = th.as_tensor(node_attrs[ntype_ids == ntype_id])

    edge_file = 'p{:03}-{}_edges.txt'.format(part_id, graph_name)
    # The format of each line in the edge file:
    # <src_id> <dst_id> <orig_src_id> <orig_dst_id> <orig_edge_id> <edge_type> <attributes>

    tmp_output = workspace_dir + '/' + edge_file + '.tmp'
    os.system('awk \'{print $1, $2, $3, $4, $5, $6}\'' + ' {} > {}'.format(input_dir + '/' + edge_file,
                                                                           tmp_output))
    edges = np.loadtxt(tmp_output, dtype=np.int64)
    src_id, dst_id, orig_src_id, orig_dst_id, orig_edge_id, etype_ids = edges[:,0], edges[:,1], \
            edges[:,2], edges[:,3], edges[:,4], edges[:,5]
    # It's not guaranteed that the edges are sorted based on edge type.
    # Let's sort edges and all attributes on the edges.
    sort_idx = np.argsort(etype_ids)
    src_id, dst_id, orig_src_id, orig_dst_id, orig_edge_id, etype_ids = src_id[sort_idx], dst_id[sort_idx], \
            orig_src_id[sort_idx], orig_dst_id[sort_idx], orig_edge_id[sort_idx], etype_ids[sort_idx]
    assert np.all(np.diff(etype_ids) >= 0)

    # Get edge attributes
    # Here we just assume all edges have the same attributes.
    # In practice, this is not the same, in which we need more complex solution to
    # encode and decode edge attributes.
    os.system('cut -d\' \' -f 7- {} > {}'.format(input_dir + '/' + edge_file, tmp_output))
    edge_attrs = th.as_tensor(np.loadtxt(tmp_output, dtype=edge_attr_dtype))[sort_idx]
    edge_feats = {}
    edge_id_start = num_edges
    for etype_name in eid_ranges:
        etype_id = etypes_map[etype_name]
        edge_map_val[etype_name].append([int(edge_id_start),
                                         int(edge_id_start + np.sum(etype_ids == etype_id))])
        edge_id_start += np.sum(etype_ids == etype_id)
        edge_feats[etype_name + '/feat'] = th.as_tensor(edge_attrs[etype_ids == etype_id])

    ids = np.concatenate([src_id, dst_id])
    uniq_ids, idx, inverse_idx = np.unique(ids, return_index=True, return_inverse=True)
    assert len(uniq_ids) == len(idx)
    local_src_id, local_dst_id = np.split(inverse_idx, 2)
    # TODO: Do we need to make sure all non-HALO nodes are assigned with lower local IDs?
    compact_g = dgl.graph((local_src_id, local_dst_id))
    compact_g.edata['orig_id'] = th.as_tensor(orig_edge_id)
    compact_g.edata[dgl.ETYPE] = th.as_tensor(etype_ids)
    compact_g.edata['inner_edge'] = th.ones(compact_g.number_of_edges(), dtype=th.bool)
    compact_g.edata[dgl.EID] = th.arange(num_edges, num_edges + compact_g.number_of_edges())
    num_edges += compact_g.number_of_edges()
    assert num_edges == edge_id_start

    orig_ids = np.concatenate([orig_src_id, orig_dst_id])
    orig_homo_ids = orig_ids[idx]
    ntype, per_type_ids = id_map(orig_homo_ids)
    compact_g.ndata['orig_id'] = th.as_tensor(per_type_ids)
    compact_g.ndata[dgl.NTYPE] = th.as_tensor(ntype)
    compact_g.ndata[dgl.NID] = th.as_tensor(uniq_ids)
    compact_g.ndata['inner_node'] = th.as_tensor(np.logical_and(uniq_ids >= nid_range[0], uniq_ids <= nid_range[1]))
    print('|V|={}'.format(compact_g.number_of_nodes()))
    print('|E|={}'.format(compact_g.number_of_edges()))

    part_dir = output_dir + '/part' + str(part_id)
    os.makedirs(part_dir, exist_ok=True)
    dgl.save_graphs(part_dir + '/graph.dgl', [compact_g])
    dgl.data.utils.save_tensors(os.path.join(part_dir, "node_feat.dgl"), node_feats)
    dgl.data.utils.save_tensors(os.path.join(part_dir, "edge_feat.dgl"), edge_feats)

part_metadata = {'graph_name': graph_name,
                 'num_nodes': num_nodes,
                 'num_edges': num_edges,
                 'part_method': 'metis',
                 'num_parts': num_parts,
                 'halo_hops': 1,
                 'node_map': node_map_val,
                 'edge_map': edge_map_val,
                 'ntypes': ntypes,
                 'etypes': etypes}
with open('{}/{}.json'.format(output_dir, graph_name), 'w') as outfile:
    json.dump(part_metadata, outfile, sort_keys=True, indent=4)
