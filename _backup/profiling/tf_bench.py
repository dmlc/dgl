"""
Adapted from code by Lingfan
"""

import argparse
import cProfile
import networkx as nx
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=-1)
args = parser.parse_args()

g = nx.read_graphml('pgp.xml')
n = g.number_of_nodes()
adj = nx.adj_matrix(g).tocoo()
indices = list(map(list, zip(adj.row, adj.col)))
values = adj.data.astype('float32').tolist()

device = '/cpu:0' if args.gpu < 0 else '/device:GPU:%d' %args.gpu
with tf.device(device):
    m = tf.SparseTensor(indices=indices, values=values, dense_shape=[n, n])
session = tf.Session()

def pagerank(alpha, tol, max_iter):
    with tf.device(device):
        p = tf.Variable(tf.constant(1.0 / n, shape=[n, 1]), trainable=False)
        new_p = alpha * tf.sparse_tensor_dense_matmul(m, p) + (1 - alpha) / n
        delta = tf.reduce_sum(tf.abs(new_p - p))
        with tf.control_dependencies([delta]):
            assignment = p.assign(new_p)

    session.run(p.initializer)
    for i in range(max_iter):
        delta, _ = session.run([delta, assignment])
        if delta < tol * n:
            break

print("Profiling PageRank")
print("==================")
print()

# cProfile.run("for i in range(10): pagerank(alpha=0.85, tol=1e-3, max_iter=10000000)", sort="cumulative")

import time
t0 = time.time()
for i in range(10):
    pagerank(alpha=0.85, tol=1e-3, max_iter=10000000)
print((time.time() - t0) / 10)
