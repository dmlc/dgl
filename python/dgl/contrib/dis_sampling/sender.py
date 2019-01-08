# This code implements a distributed dgl-subgraph sender, which 
# samples sub-graphs continuously from a huge graph and sends sub-graphs
# to a reciver node, which uses these sub-graphs for model training.
import mxnet as mx
import numpy as np

import time

import grpc
import sampler_pb2
import sampler_pb2_grpc

def serilize(mx_ndarray):
    """ This method serilizes a mxnet ndarray to memory.
    It will first convert mxnet ndarray to numpy ndarray,
    and then serilize it to binary string.

    Args:
      mx_ndarray: mxnet ndarray
    Returns:
      serialized binary string
    """
    np_ndarray = mx_ndarray.asnumpy()
    str = np_ndarray.tostring()
    return str

def deserilize(str_array, type):
    """ This method deserilizes a mxnet ndarray from a binary string.
    It first serlizes the binary string to a numpy array, and then
    converts it to a mxnet ndarray.

    Args:
      str_array: binary ndarray
      type: data type
    Returns:
      mxnet ndarray
    """
    np_ndarray = np.frombuffer(str_array, dtype=type)
    mx_ndarray = mx.nd.array(np_ndarray, dtype=type)
    return mx_ndarray

class Sender(object):
    """ Send sub-graphs to reciver
    """
    def __init__(self, addr):
        """ Constructor

        Args:
            addr: remote address string, e.g., '162.168.8.12:50051'
        """
        self.addr = addr

    def bind(self, fn, graph, seeds, num_args, num_hops, num_neighbor, max_num_vertices):
        """ Bind sampler function and hyper-parameters for Sender

        Args:
            fn: sampler function
            graph: original graph used for sampling
            seeds: seeds vertices
            num_args: number of arguments for sampler
            num_hops: number of hops
            num_neighbor: number of neighbor
            max_num_vertices: max number of sampled vertices
        """
        self.func = fn
        self.graph = graph
        self.seeds = seeds
        self.num_args = num_args
        self.num_hops = num_hops
        self.num_neighbor = num_neighbor
        self.max_num_vertices = max_num_vertices

    def stop(self):
        """ Stop sampling
        """
        print("Stop sender ... ")

    def start(self):
        """ Start sampling
        """
        print("Start sender ... ")
        try:
            while True:
                sub_graph = self.func(
                    self.graph, 
                    self.seeds, 
                    num_args=self.num_args,
                    num_hops=self.num_hops, 
                    num_neighbor=self.num_neighbor, 
                    max_num_vertices=self.max_num_vertices)
                # Serilize mxnet ndarray to binary string
                str_ver_id = serilize(sub_graph[0])
                str_layer = serilize(sub_graph[2])
                str_csr_data = serilize(sub_graph[1].data)
                str_csr_indices = serilize(sub_graph[1].indices)
                str_csr_indptr = serilize(sub_graph[1].indptr)
                # Send sub-graphs to reciver
                with grpc.insecure_channel(self.addr) as channel:
                    stub = sampler_pb2_grpc.SamplerStub(channel)
                    response = stub.SendSubGraph(sampler_pb2.SamplerRequest(
                        vertices_id = str_ver_id,
                        layer = str_layer,
                        csr_data = str_csr_data,
                        csr_indices = str_csr_indices,
                        csr_indptr = str_csr_indptr,
                        csr_shape_0 = self.max_num_vertices,
                        csr_shape_1 = self.graph.shape[1]))
                    print(response)

        except KeyboardInterrupt:
            self.stop()
