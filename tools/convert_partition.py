import os
import argparse
import numpy as np
import dgl
import torch as th

parser = argparse.ArgumentParser(description='Construct graph partitions')
parser.add_argument('--input-dir', required=True, type=str,
                    help='The directory path that contains the partition results.')
parser.add_argument('--graph-name', required=True, type=str,
                    help='The graph name')
parser.add_argument('--num-parts', required=True, type=int,
                    help='The number of partitions')
parser.add_argument('--num-ntypes', type=int, required=True,
                    help='The number of node types in the graph.')
parser.add_argument('--num-node-weights', required=True, type=int,
                    help='The number of node weights used by METIS.')
parser.add_argument('--workspace', type=str, default='/tmp',
                    help='The directory to store the intermediate results')
parser.add_argument('-o', '--output', required=True, type=str,
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

id_ranges = [[0, 163820],
             [163820, 262111]]
id_ranges = np.stack([np.array(row) for row in id_ranges], 1)
id_map = dgl.distributed.id_map.IdMap({'_N': id_ranges})

num_edges = 0
for part_id in range(num_parts):
    node_file = 'p{:03}-{}_nodes.txt'.format(part_id, graph_name)
    # The format of each line in the node file:
    # <node_id> <ndoe_type> <weight1> ... <orig_node_id> <attributes>
    orig_nid_col = 3 + num_node_weights
    first_attr_col = 4 + num_node_weights

    # Get the first two columns which is the node ID and node type.
    tmp_output = workspace_dir + '/' + node_file + '.tmp'
    os.system('awk \'{print $1, $2, $' + str(orig_nid_col) + '}\''
              + ' {} > {}'.format(input_dir + '/' + node_file, tmp_output))
    nodes = np.loadtxt(tmp_output, dtype=np.int64)
    nids, ntypes, orig_nid = nodes[:,0], nodes[:,1], nodes[:,2]
    assert np.all(nids[1:] - nids[:-1] == 1)
    nid_range = (nids[0], nids[-1])

    # Get node attributes
    # Here we just assume all nodes have the same attributes.
    # In practice, this is not the same, in which we need more complex solution to
    # encode and decode node attributes.
    #os.system('cut -d\' \' -f {}- {} > tmp_output'.format(first_attr_col,
    #                                                      input_dir + '/' + node_file,
    #                                                      tmp_output))
    #node_attrs = np.loadtxt(tmp_output, dtype=node_attr_dtype)
    #print(node_attrs)

    edge_file = 'p{:03}-{}_edges.txt'.format(part_id, graph_name)
    # The format of each line in the edge file:
    # <src_id> <dst_id> <orig_src_id> <orig_dst_id> <orig_edge_id> <edge_type> <attributes>

    tmp_output = workspace_dir + '/' + edge_file + '.tmp'
    os.system('awk \'{print $1, $2, $3, $4, $5, $6}\'' + ' {} > {}'.format(input_dir + '/' + edge_file,
                                                                           tmp_output))
    edges = np.loadtxt(tmp_output, dtype=np.int64)
    src_id, dst_id, orig_src_id, orig_dst_id, orig_edge_id, etype = edges[:,0], edges[:,1], \
            edges[:,2], edges[:,3], edges[:,4], edges[:,5]

    # Get edge attributes
    # Here we just assume all edges have the same attributes.
    # In practice, this is not the same, in which we need more complex solution to
    # encode and decode edge attributes.
    #os.system('cut -d\' \' -f 7- {} > tmp_output'.format(input_dir + '/' + edge_file, tmp_output))
    #edge_attrs = th.as_tensor(np.loadtxt(tmp_output, dtype=edge_attr_dtype))

    ids = np.concatenate([src_id, dst_id])
    uniq_ids, idx, inverse_idx = np.unique(ids, return_index=True, return_inverse=True)
    assert len(uniq_ids) == len(idx)
    local_src_id, local_dst_id = np.split(inverse_idx, 2)
    # TODO: Do we need to make sure all non-HALO nodes are assigned with lower local IDs?
    compact_g = dgl.graph((local_src_id, local_dst_id))
    compact_g.edata['orig_id'] = th.as_tensor(orig_edge_id)
    compact_g.edata[dgl.ETYPE] = th.as_tensor(etype)
    compact_g.edata['inner_edge'] = th.ones(compact_g.number_of_edges(), dtype=th.bool)
    compact_g.edata[dgl.EID] = th.arange(num_edges, num_edges + compact_g.number_of_edges())
    num_edges += compact_g.number_of_edges()

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
