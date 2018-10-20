import mxnet as mx
import numpy as np
import time
import argparse
from dgl.graph import create_graph_index
from dgl import utils
from dgl.data import register_data_args, load_data

def get_in_edges(g, i, batch_size):
    src, _, _ = g.in_edges(utils.toindex(np.arange(i * batch_size, (i + 1) * batch_size)))
    return src

def get_immutable_in_edges(g, i, batch_size):
    rows = g._in_csr[(i * batch_size):((i + 1) * batch_size)]
    src = utils.toindex(rows.indices)
    return src

def subgraph_gen1(g, i, batch_size, get_in_edges):
    src = get_in_edges(g, i, batch_size)
    seed = np.arange(i * batch_size, (i + 1) * batch_size, dtype=np.int64)
    vs = np.concatenate((src.tousertensor().asnumpy(), seed), axis=0)
    vs = mx.nd.array(np.unique(vs), dtype=np.int64)
    subg = g.node_subgraph(utils.toindex(vs))
    return subg

def subgraph_gen2(g, i, batch_size, get_in_edges):
    src = get_in_edges(g, i, batch_size)
    seed = np.arange(i * batch_size, (i + 1) * batch_size, dtype=np.int64)
    vs = np.concatenate((src.tousertensor().asnumpy(), seed), axis=0)
    vs = mx.nd.array(np.unique(vs), dtype=np.int64)
    subg = g.node_subgraph(utils.toindex(vs))
    return subg

def subgraph_gen3(g, n, batch_idx, batch_size, get_in_edges):
    subg_vs = []
    for _ in range(n):
        src = get_in_edges(g, batch_idx, batch_size)
        seed = np.arange(batch_idx * batch_size, (batch_idx + 1) * batch_size, dtype=np.int64)
        vs = np.concatenate((src.tousertensor().asnumpy(), seed), axis=0)
        vs = mx.nd.array(np.unique(vs), dtype=np.int64)
        subg_vs.append(utils.toindex(vs))
        batch_idx = batch_idx + 1
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

    t0 = time.time()
    for _ in range(1):
        for i in range(graph.number_of_nodes() / args.subgraph_size):
            subg = subgraph_gen1(g, i, args.subgraph_size, get_in_edges)
    mx.nd.waitall()
    t1 = time.time()
    print("subgraph on a mutable graph: " + str(t1 - t0))

    t0 = time.time()
    for _ in range(1):
        for i in range(graph.number_of_nodes() / args.subgraph_size):
            subg = subgraph_gen2(ig, i, args.subgraph_size, get_immutable_in_edges)
    mx.nd.waitall()
    t1 = time.time()
    print("subgraph on a immutable graph: " + str(t1 - t0))

    t0 = time.time()
    for _ in range(1):
        for i in range(graph.number_of_nodes() / args.subgraph_size / 4):
            subg = subgraph_gen3(ig, 4, i * 4, args.subgraph_size, get_immutable_in_edges)
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
