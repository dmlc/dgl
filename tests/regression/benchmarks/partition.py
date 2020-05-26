import dgl
from dgl import distributed as dgl_distributed
import argparse, time
from ogb.nodeproppred import DglNodePropPredDataset

parser = argparse.ArgumentParser(description='partition')
parser.add_argument("--dataset", type=float, default=0.5,
                    help="dropout probability")
args = parser.parse_args()

if args.dataset == 'amazon':
    data = DglNodePropPredDataset(name='ogbn-products')

graph, labels = data[0]

start = time.time()
dgl_distributed.partition_graph(graph, 'ogbn-prod', 16, '/tmp', num_hops=1, part_method="metis")
print('Time: {} seconds'.format(time.time() - start))
