import socket
import dgl
import io
import pickle
import numpy as np
import time
from .utils import recvall

class NodeFlowSender(object):
    def __init__(self, host, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        for _ in range(30):
            try:
                s.connect((host, port))
                break
            except Exception as e:
                print(e)
                time.sleep(30)
                continue
        print('Connected to %s:%d' % (host, port))
        self.socket = s

    def send(self, nf, aux_data):
        nf_buffer = dgl.network.serialize_nodeflow(nf)
        with io.BytesIO() as bio:
            pickle.dump(aux_data, bio)
            aux_buffer = bio.getvalue()
        with io.BytesIO() as bio:
            bio.write(np.array([len(aux_buffer), len(nf_buffer)], dtype='int32').tobytes())
            bio.write(aux_buffer)
            bio.write(nf_buffer)
            self.socket.sendall(bio.getvalue())

    def recv(self):
        data_len = np.frombuffer(recvall(self.socket, 4, False), dtype='int32')[0]
        with recvall(self.socket, data_len, True) as bio:
            data = pickle.load(bio)
        return data

    def complete(self):
        self.socket.sendall(np.array([0, 0], dtype='int32').tobytes())
