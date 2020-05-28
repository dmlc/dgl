import dgl
from dgl import distributed as dgl_distributed
import argparse, time
from utils import get_graph

parser = argparse.ArgumentParser(description='partition')
parser.add_argument("--dataset", type=str, default='livejournal',
                    help="specify the graph for partitioning")
parser.add_argument("--num_parts", type=int, default=16,
                    help="the number of partitions")
args = parser.parse_args()

g = get_graph(args.dataset)
print('{}: |V|={}, |E|={}'.format(args.dataset, g.number_of_nodes(), g.number_of_edges()))
start = time.time()
dgl_distributed.partition_graph(g, args.dataset, args.num_parts, '/tmp', num_hops=1, part_method="metis")
print('Time: {} seconds'.format(time.time() - start))
