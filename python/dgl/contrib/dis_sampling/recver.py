# This code implements a distributed dgl-subgraph recver, which 
# uses multi-thread to handle sampling and training.
from concurrent import futures

import mxnet as mx
import numpy as np

import time
from queue import Queue

import grpc
import sampler
import sampler_pb2
import sampler_pb2_grpc

_MAX_QUEUE_SIZE = 10

graph_queue = Queue(maxsize = _MAX_QUEUE_SIZE)
args_queue = Queue(maxsize = _MAX_QUEUE_SIZE)

class SamplerRecver(sampler_pb2_grpc.SamplerServicer):
	""" SamplerRecer is called by trainer and it uses back-end thread
	    to recv sub-graph from sampler node via gRPC and uses other 
	    threads to train model.
	"""

	def AskArgs(self, request, context):
	    """ Initialzie arguments for remote sampler

	    Args:
	      request: grpc request
	      context: grpc context
	    """
	    print("Remote sampler " + str(request.ip) + " connected! ")

	    if args_queue.empty() == False:
	    	args = args_queue.get()
	    	return sampler_pb2.SamplerReply(
	    		seeds = sampler.Serilize(args[0]),
	    		num_args = args[1],
	    		num_hops = args[2],
	    		num_neighbor = args[3],
	    		max_num_vertices = args[4],
	    		update = True)
	    else:
	    	return sampler_pb2.SamplerReply(update=False)

	def SendSubGraph(self, request, context):
	    """ Recv sub-graph, re-construct it, and 
	        put it into the queue.

	    Args:
	      request: grpc request
	      context: grpc context
	    """
	    ver_id = sampler.Deserilize(request.vertices_id, np.int64)
	    layer = sampler.Deserilize(request.layer, np.int64)
	    csr_data = sampler.Deserilize(request.csr_data, np.int64)
	    csr_indices = sampler.Deserilize(request.csr_indices, np.int64)
	    csr_indptr = sampler.Deserilize(request.csr_indptr, np.int64)
	    csr_shape_0 = request.csr_shape_0
	    csr_shape_1 = request.csr_shape_1

	    csr_graph = mx.nd.sparse.csr_matrix((
	    	csr_data, 
	    	csr_indices, 
	    	csr_indptr), 
	      shape=(csr_shape_0, csr_shape_1))

	    sub_graph = (ver_id, layer, csr_graph)
	    # Put sub_graph to the queue
	    # This action can be blocked if the queue if full.
	    graph_queue.put(sub_graph)

	    if args_queue.empty() == False:
	    	args = args_queue.get()
	    	return sampler_pb2.SamplerReply(
	    		seeds = sampler.Serilize(args[0]),
	    		num_args = args[1],
	    		num_hops = args[2],
	    		num_neighbor = args[3],
	    		max_num_vertices = args[4],
	    		update = True)
	    else:
	    	return sampler_pb2.SamplerReply(update=False)

	def InitArgs(self, 
		port,
		seeds, 
		num_args, 
		num_hops, 
		num_neighbor, 
		max_num_vertices):
	    """ Initialize sampler arguments

	    Args:
	      port: local port for listening requets
	      seeds: seed vertices for sampling
	      num_args: number of arguments
	      num_hops: number of hops
	      num_neighbor: number of neighbor
	      max_num_vertices: maximal number of vertices
	    """
	    self.Port = port
	    args = (seeds, num_args, num_hops, num_neighbor, max_num_vertices)
	    args_queue.put(args)

	def StartService(self):
		""" Start gRPC server in back-end thread
		"""
		server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
		sampler_pb2_grpc.add_SamplerServicer_to_server(SamplerRecver(), server)
		server.add_insecure_port('[::]:%d' % self.Port)
		server.start()
		print("Start recver ... ")

		return server

	def Sample(self):
		""" Get one of the sub_graph from the queue.
		    This action could be blocked if the queue is empty.
		""" 
		return graph_queue.get()

	def SetArguments(self, 
		seeds, 
		num_args, 
		num_hops, 
		num_neighbor, 
		max_num_vertices):
		""" Reset arguments during the training.

		Args:
	      seeds: seed vertices for sampling
	      num_args: number of arguments
	      num_hops: number of hops
	      num_neighbor: number of neighbor
	      max_num_vertices: maximal number of vertices
		"""
		args = (seeds, num_args, num_hops, num_neighbor, max_num_vertices)
		args_queue.put(args)
