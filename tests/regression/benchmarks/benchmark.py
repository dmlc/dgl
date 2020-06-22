import dgl
from dgl import distributed as dgl_distributed
import argparse, time
from utils import get_graph

parser = argparse.ArgumentParser(description='partition')
parser.add_argument("--dataset", type=str, default='livejournal',
                    help="specify the graph for partitioning")
args = parser.parse_args()

g = get_graph(args.dataset)
print('{}: |V|={}, |E|={}'.format(args.dataset, g.number_of_nodes(), g.number_of_edges()))

def run_partition(g):
    dgl_distributed.partition_graph(g, args.dataset, 8, '/tmp', num_hops=1, part_method="metis")

def run_reverse(g):
    dgl.reverse(g)

full_graph_tests = {
    'partition' : run_partition,
    'reverse' : run_reverse,
}

mini_batch_tests = {
}

for name in full_graph_tests:
    start = time.time()
    full_graph_tests[name](g)
    print('full_graph {} {} seconds'.format(name, time.time() - start))

for name in mini_batch_tests:
    start = time.time()
    mini_batch_tests[name](g)
    print('mini_batch {} {} seconds'.format(name, time.time() - start))
