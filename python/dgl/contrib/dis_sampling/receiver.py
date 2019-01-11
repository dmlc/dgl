# This code implements a distributed dgl-subgraph receiver.
from concurrent import futures

import mxnet as mx
import numpy as np

import time
from queue import Queue

import grpc
import sender
import sampler_pb2
import sampler_pb2_grpc

_MAX_MESSAGE_LENGTH = 1024 ** 3  # Gigabyte

class Receiver(sampler_pb2_grpc.SamplerServicer):
    """ Reciever is called by trainer and it uses back-end thread
        to receive sub-graph samples from remote sampler.
    """
    def __init__(self, port, buffer_size):
    	self.port = port;
    	self.queue = Queue(maxsize = buffer_size)

    def SendSubGraph(self, request, context):
        """ Recv sub-graph, re-construct it, and 
            put it into the queue.

        Args:
          request: grpc request
          context: grpc context
        """
        ver_id = sender.deserilize(request.vertices_id, np.int64)
        layer = sender.deserilize(request.layer, np.int64)
        csr_data = sender.deserilize(request.csr_data, np.int64)
        csr_indices = sender.deserilize(request.csr_indices, np.int64)
        csr_indptr = sender.deserilize(request.csr_indptr, np.int64)
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
        self.queue.put(sub_graph)

        return sampler_pb2.SamplerReply(response='Get sub-graph!')

    def start(self):
        """ Start gRPC server in back-end thread

        Args:
            port: local network port
            buffer_size: buffer size for message queue
        """
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
            ('grpc.max_send_message_length', _MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', _MAX_MESSAGE_LENGTH)])
        sampler_pb2_grpc.add_SamplerServicer_to_server(self, server)
        server.add_insecure_port('[::]:%d' % self.port)
        server.start()
        print("Start recver ... ")

        return server

    def recv(self):
        """ Return sub-graph at the top of the message queue
        """
        return self.queue.get()