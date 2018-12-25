# This is a demo code shows that how to 
# implement a trainer node
import mxnet as mx
import numpy as np

import recver
import time

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

_LOCAL_PORT = 50051

def do_work(sub_graph):
	""" User-defined work

	Args:
	  sub_graph: sampled sub-graph
	"""
	print("------------------------")
	print(sub_graph[2].asnumpy())
	print("------------------------")

# Connect to server node and sample sub-graphs in a loop
def start_recver():
	seed = mx.nd.array([0,1,2,3,4], dtype=np.int64)
	sample_recver = recver.SamplerRecver()
	sample_recver.InitArgs(
		_LOCAL_PORT,
		seeds=seed,
		num_args=2,
		num_hops=1,
		num_neighbor=2,
		max_num_vertices=5)

	server = sample_recver.StartService()

	for i in range(5):
		sub_graph = sample_recver.Sample()
		do_work(sub_graph)

	print("Change args!")

	sample_recver.SetArguments(
		seeds=seed,
		num_args=2,
		num_hops=1,
		num_neighbor=4,
		max_num_vertices=5)

	while True:
		sub_graph = sample_recver.Sample()
		do_work(sub_graph)

	return server

if __name__ == '__main__':
    server = start_recver()
    try:
    	while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)