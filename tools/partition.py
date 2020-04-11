import numpy as np
import argparse
import signal
import dgl
from dgl import backend as F
from dgl.data.utils import load_graphs, save_graphs
import pickle

def main():
    parser = argparse.ArgumentParser(description='Partition a graph')
    parser.add_argument('--data', required=True, type=str,
                        help='The file path of the input graph in the DGL format.')
    parser.add_argument('--graph-name', required=True, type=str,
                        help='The graph name')
    parser.add_argument('-k', '--num-parts', required=True, type=int,
                        help='The number of partitions')
    parser.add_argument('--num-hops', type=int, default=1,
                        help='The number of hops of HALO nodes we include in a partition')
    parser.add_argument('-m', '--method', required=True, type=str,
                        help='The partitioning method: random, metis')
    parser.add_argument('-o', '--output', required=True, type=str,
                        help='The output directory of the partitioned results')
    args = parser.parse_args()
    data_path = args.data
    num_parts = args.num_parts
    num_hops = args.num_hops
    method = args.method
    output = args.output
    graph_name = args.graph_name

    glist, _ = load_graphs(data_path)
    g = glist[0]

    if num_parts == 1:
        server_parts = {0: g}
        client_parts = {0: g}
        g.ndata['part_id'] = F.zeros((g.number_of_nodes()), F.int64, F.cpu())
        g.ndata[dgl.NID] = F.arange(0, g.number_of_nodes())
        g.edata[dgl.EID] = F.arange(0, g.number_of_edges())
    elif args.method == 'metis':
        node_parts = dgl.transform.metis_partition_assignment(g, num_parts)
        server_parts = dgl.transform.partition_graph_with_halo(g, node_parts, 0)
        client_parts = dgl.transform.partition_graph_with_halo(g, node_parts, num_hops)
        for part_id in client_parts:
            part = client_parts[part_id]
            part.ndata['part_id'] = F.gather_row(node_parts, part.ndata[dgl.NID])
    elif args.method == 'random':
        node_parts = np.random.choice(num_parts, g.number_of_nodes())
        server_parts = dgl.transform.partition_graph_with_halo(g, node_parts, 0)
        client_parts = dgl.transform.partition_graph_with_halo(g, node_parts, num_hops)
        for part_id in client_parts:
            part = client_parts[part_id]
            part.ndata['part_id'] = F.gather_row(node_parts, part.ndata[dgl.NID])
    else:
        raise Exception('unknown partitioning method: ' + args.method)

    tot_num_inner_edges = 0
    for part_id in range(num_parts):
        serv_part = server_parts[part_id]
        part = client_parts[part_id]

        if num_parts > 1:
            num_inner_nodes = len(np.nonzero(F.asnumpy(part.ndata['inner_node']))[0])
            num_inner_edges = len(np.nonzero(F.asnumpy(part.edata['inner_edge']))[0])
            print('part {} has {} nodes and {} edges. {} nodes and {} edges are inside the partition'.format(
                part_id, part.number_of_nodes(), part.number_of_edges(),
                num_inner_nodes, num_inner_edges))
            tot_num_inner_edges += num_inner_edges
            serv_part.copy_from_parent()

        save_graphs('{}/{}-server-{}.dgl'.format(output, graph_name, part_id), [serv_part])
        save_graphs('{}/{}-client-{}.dgl'.format(output, graph_name, part_id), [part])
    meta = np.array([g.number_of_nodes(), g.number_of_edges()])
    pickle.dump(meta, open('{}/{}-meta.pkl'.format(output, graph_name), 'wb'))
    num_cuts = g.number_of_edges() - tot_num_inner_edges
    if num_parts == 1:
        num_cuts = 0
    print('there are {} edges in the graph and {} edge cuts for {} partitions.'.format(
        g.number_of_edges(), num_cuts, num_parts))

if __name__ == '__main__':
    main()
