import numpy as np
import argparse
import signal
import dgl
from dgl import backend as F
from dgl.data.utils import load_graphs, save_graphs

def main():
    parser = argparse.ArgumentParser(description='Partition a graph')
    parser.add_argument('--data', required=True, type=str,
                        help='The file path of the input graph in the DGL format.')
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

    glist, _ = load_graphs(data_path)
    g = glist[0]

    if args.method == 'metis':
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

        num_inner_nodes = len(np.nonzero(F.asnumpy(part.ndata['inner_node']))[0])
        num_inner_edges = len(np.nonzero(F.asnumpy(part.edata['inner_edge']))[0])
        print('part {} has {} nodes and {} edges. {} nodes and {} edges are inside the partition'.format(
              part_id, part.number_of_nodes(), part.number_of_edges(),
              num_inner_nodes, num_inner_edges))
        tot_num_inner_edges += num_inner_edges

        serv_part.copy_from_parent()
        save_graphs(output + '/server-' + str(part_id) + '.dgl', [serv_part])
        save_graphs(output + '/client-' + str(part_id) + '.dgl', [part])
    print('there are {} edges in the graph and {} edge cuts for {} partitions.'.format(
        g.number_of_edges(), g.number_of_edges() - tot_num_inner_edges, num_parts))

if __name__ == '__main__':
    main()
