# This code implements a distributed dgl-subgraph sampler, which 
# samples sub-graphs continuously from a huge graph and sends sub-graphs
# to a Recver node, which uses these sub-graphs for ML training.
import mxnet as mx
import numpy as np

import time

import grpc
import sampler_pb2
import sampler_pb2_grpc

_LOCAL_IP = '127.0.0.1'

def Serilize(mx_ndarray):
    """ This method serilizes a mxnet ndarray to memory.
        It will first convert mxnet ndarray to numpy ndarray,
        and then serilize it to binary string.

    Args:
      mx_ndarray: mxnet ndarray
    """
    np_ndarray = mx_ndarray.asnumpy()
    str = np_ndarray.tostring()
    return str

def Deserilize(str_array, type):
    """ This method deserilizes a mxnet ndarray from a binary string.
        It first serlizes the binary string to a numpy array, and then
        converts it to a mxnet ndarray.

    Args:
      str_array: binary ndarray
      type: data type
    """
    np_ndarray = np.frombuffer(str_array, dtype=type)
    mx_ndarray = mx.nd.array(np_ndarray, dtype=type)
    return mx_ndarray

class Sampler(object):
    """ Sampler is used to sample sub-graph on remote machine, and
        send sub-graph to trainer via the gRPC protocol.
    """ 
    def __init__(self, graph, addr):
        """ Constructor

        Args:
          graph: csr graph
          addr: address of remote trainer
        """
        self.Graph = graph
        self.Addr = addr

    def AskArguments(self):
        """ Ask initial arguments from remote trainer vai gRPC
        """
        with grpc.insecure_channel(self.Addr) as channel:
            stub = sampler_pb2_grpc.SamplerStub(channel)
            response = stub.AskArgs(sampler_pb2.ArgsRequest(
                ip=_LOCAL_IP))

        self.Seeds = Deserilize(response.seeds, np.int64)
        self.NumArgs = response.num_args
        self.NumHops = response.num_hops
        self.NumNeighbor = response.num_neighbor
        self.MaxVertices = response.max_num_vertices

    def Sample(self):
        """ Sample sub-graph from original big graph
            This function can be over-write by users.
        """
        pass

    def Stop(self):
        """ Stop sampler's work
            This function can be over-write by users.
        """
        print("Stopping sampler...")

    def SendToTrainer(self, sub_graph, max_vertices):
        """ Send sub-graph to remote trainer via gRPC.

        Args:
          sub_graph: csr sub-graph, sampled from original graph
          max_vertices: maximal number of sampled vertices
        """
        str_ver_id = Serilize(sub_graph[0])
        str_layer = Serilize(sub_graph[2])
        str_csr_data = Serilize(sub_graph[1].data)
        str_csr_indices = Serilize(sub_graph[1].indices)
        str_csr_indptr = Serilize(sub_graph[1].indptr)
        
        with grpc.insecure_channel(self.Addr) as channel:
            stub = sampler_pb2_grpc.SamplerStub(channel)
            response = stub.SendSubGraph(sampler_pb2.SamplerRequest(
                vertices_id = str_ver_id,
                layer = str_layer,
                csr_data = str_csr_data,
                csr_indices = str_csr_indices,
                csr_indptr = str_csr_indptr,
                csr_shape_0 = max_vertices,
                csr_shape_1 = self.Graph.shape[1]))

        # Sampler can update its arguments by each grpc call
        if response.update == True:
            self.Seeds = Deserilize(response.seeds, np.int64)
            self.NumArgs = response.num_args
            self.NumHops = response.num_hops
            self.NumNeighbor = response.num_neighbor
            self.MaxVertices = response.max_num_vertices

    def Start(self):
        """ Sample sub-graph in a loop
        """
        try:
            self.AskArguments()
            while True:
                sub_graph, max_vertices = self.Sample()
                self.SendToTrainer(sub_graph, max_vertices)
        except KeyboardInterrupt:
            self.Stop()
