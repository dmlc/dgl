from dataloader import get_dataset
import scipy as sp
import numpy as np
import argparse
import signal
import dgl
from dgl import backend as F
from dgl.data.utils import load_graphs, save_graphs

def write_graph_txt(path, graph):
    edge_ids = F.arange(0, graph.number_of_edges())
    src, dst = graph.find_edges(edge_ids)
    edges = graph.all_edges(form='all', order='srcdst')
    print(graph.ndata['part_id'])
    print(graph.parent_nid)





def main():
    parser = argparse.ArgumentParser(description='Partition a knowledge graph')
    parser.add_argument('--data_path', type=str, default='data',
                        help='root path of all dataset')
    parser.add_argument('--data_files', type=str, default=None, nargs='+',
                        help='a list of data files, e.g. entity relation train valid test')
    parser.add_argument('--format', type=str, required=True,
                        help='the format of the dataset, it can be raw_udd_{htr} and udd_{htr}')
    parser.add_argument('-k', '--num-parts', required=True, type=int,
                        help='The number of partitions')
    args = parser.parse_args()
    num_parts = args.num_parts

    # load dataset and samplers
    dataset = get_dataset(args.data_path, None, args.format, args.data_files)

    src, etype_id, dst = dataset.train
    coo = sp.sparse.coo_matrix((np.ones(len(src)), (src, dst)),
            shape=[dataset.n_entities, dataset.n_entities])
    g = dgl.DGLGraph(coo, readonly=True, sort_csr=True)
    g.edata['tid'] = F.tensor(etype_id, F.int64)

    part_dict = dgl.transform.metis_partition(g, num_parts, 1)

    tot_num_inner_edges = 0
    for part_id in part_dict:
        part = part_dict[part_id]

        num_inner_nodes = len(np.nonzero(F.asnumpy(part.ndata['inner_node']))[0])
        num_inner_edges = len(np.nonzero(F.asnumpy(part.edata['inner_edge']))[0])
        print('part {} has {} nodes and {} edges. {} nodes and {} edges are inside the partition'.format(
              part_id, part.number_of_nodes(), part.number_of_edges(),
              num_inner_nodes, num_inner_edges))
        tot_num_inner_edges += num_inner_edges

        part.copy_from_parent()
        #save_graphs(args.data_path + '/part_' + str(part_id) + '.dgl', [part])
        write_graph_txt(args.data_path + '/partition_' + str(part_id), part)

    print('there are {} edges in the graph and {} edge cuts for {} partitions.'.format(
        g.number_of_edges(), g.number_of_edges() - tot_num_inner_edges, len(part_dict)))

if __name__ == '__main__':
    main()