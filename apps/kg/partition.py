from dataloader import get_dataset
import scipy as sp
import numpy as np
import argparse
import signal
import os
import dgl
from dgl import backend as F
from dgl.data.utils import load_graphs, save_graphs

def write_graph_txt(path, file_name, part_dict, total_nodes):
    partition_book = [0] * total_nodes
    for part_id in part_dict:
        # Get (h,r,t) triples
        new_path = path + str(part_id)
        if not os.path.exists(new_path):
            os.mkdir(new_path)
        new_file_name = os.path.join(new_path, file_name)
        f = open(new_file_name, 'w')
        graph = part_dict[part_id]
        src, dst = graph.all_edges(form='uv', order='eid')
        rel = graph.edata['tid']
        assert len(src) == len(rel)
        src = F.asnumpy(src)
        dst = F.asnumpy(dst)
        rel = F.asnumpy(rel)
        for i in range(len(src)):
            f.write(str(src[i])+'\t'+str(rel[i])+'\t'+str(dst[i])+'\n')
        f.close()
        # Get local2global
        new_file_name = os.path.join(new_path, 'local_to_global.txt')
        f = open(new_file_name, 'w')
        pid = F.asnumpy(graph.parent_nid)
        for i in range(len(pid)):
            f.write(str(pid[i])+'\n')
        f.close()
        # Update partition_book
        part_id = F.asnumpy(graph.ndata['part_id'])
        for i in range(len(pid)):
            partition_book[pid[i]] = part_id[i]
    # Write partition_book.txt
    for part_id in part_dict:
        new_path = path + str(part_id)
        new_file_name = os.path.join(new_path, 'partition_book.txt')
        f = open(new_file_name, 'w')
        for i in range(len(partition_book)):
            f.write(str(partition_book[i])+'\n')
        f.close()

def main():
    parser = argparse.ArgumentParser(description='Partition a knowledge graph')
    parser.add_argument('--data_path', type=str, default='data',
                        help='root path of all dataset')
    parser.add_argument('--dataset', type=str, default='FB15k',
                        help='dataset name, under data_path')
    parser.add_argument('--data_files', type=str, default=None, nargs='+',
                        help='a list of data files, e.g. entity relation train valid test')
    parser.add_argument('--format', type=str, default='built_in',
                        help='the format of the dataset, it can be built_in,'\
                                'raw_udd_{htr} and udd_{htr}')
    parser.add_argument('-k', '--num-parts', required=True, type=int,
                        help='The number of partitions')
    args = parser.parse_args()
    num_parts = args.num_parts

    # load dataset and samplers
    dataset = get_dataset(args.data_path, args.dataset, args.format, args.data_files)

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
    write_graph_txt(args.data_path+'/FB15k/partition_', 'train.txt', part_dict, g.number_of_nodes())
    print('there are {} edges in the graph and {} edge cuts for {} partitions.'.format(
        g.number_of_edges(), g.number_of_edges() - tot_num_inner_edges, len(part_dict)))

if __name__ == '__main__':
    main()