import mxnet as mx
import numpy as np
import time
import argparse
from dgl.graph import create_graph_index
from dgl import utils
from dgl.data import register_data_args, load_data

def get_vertices(g, subgraph_size):
    seed = utils.toindex(np.random.randint(0, g.number_of_nodes(), subgraph_size))
    src, _, _ = g.in_edges(seed)
    vs = np.concatenate((src.tousertensor().asnumpy(), seed.tousertensor().asnumpy()), axis=0)
    vs = mx.nd.array(np.unique(vs), dtype=np.int64)
    return vs


def subgraph_gen1(g, subgraph_size):
    vs = get_vertices(g, subgraph_size)
    subg = g.node_subgraph(utils.toindex(vs))
    return subg

def subgraph_gen2(g, n, subgraph_size):
    if n == 1:
        vs = get_vertices(g, subgraph_size)
        subg = g.node_subgraph(utils.toindex(vs))
        return subg
    else:
        subg_vs = []
        for _ in range(n):
            vs = get_vertices(g, subgraph_size)
            subg_vs.append(utils.toindex(vs))
        subgs = g.node_subgraphs(subg_vs)
        return subgs

def test_subgraph_gen(args):
    # load and preprocess dataset
    t0 = time.time()
    data = load_data(args)
    graph = data.graph
    try:
        graph_data = graph.get_graph()
    except:
        graph_data = graph
    print("load data: " + str(time.time() - t0))

    t0 = time.time()
    g = create_graph_index(graph_data)
    print("create graph index: " + str(time.time() - t0))

    t0 = time.time()
    ig = create_graph_index(graph_data, immutable_graph=True)
    print("create immutable graph index: " + str(time.time() - t0))

    for _ in range(5):
        t0 = time.time()
        for i in range(int(graph.number_of_nodes() / args.subgraph_size)):
            subg = subgraph_gen1(g, args.subgraph_size)
        mx.nd.waitall()
        t1 = time.time()
        print("subgraph on a mutable graph: " + str(t1 - t0))

    for _ in range(5):
        t0 = time.time()
        for i in range(int(graph.number_of_nodes() / args.subgraph_size)):
            subg = subgraph_gen2(ig, 1, args.subgraph_size)
            mx.nd.waitall()
        t1 = time.time()
        print("subgraph on a immutable graph: " + str(t1 - t0))

    for _ in range(5):
        t0 = time.time()
        for i in range(int(graph.number_of_nodes() / args.subgraph_size / 4)):
            subg = subgraph_gen2(ig, 4, args.subgraph_size)
            mx.nd.waitall()
        t1 = time.time()
        print("subgraph on a immutable graph: " + str(t1 - t0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='subgraph')
    register_data_args(parser)
    parser.add_argument("--subgraph-size", type=int, default=1000,
            help="The number of seed vertices in a subgraph.")
    parser.add_argument("--n-parallel", type=int, default=1,
            help="the number of subgraph construction in parallel.")
    args = parser.parse_args()

    test_subgraph_gen(args)
