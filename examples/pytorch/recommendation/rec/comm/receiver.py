import socket
import dgl
import io
import pickle
import numpy as np
import selectors
import errno
import time
from .utils import recvall, sendall

class NodeFlowReceiver(object):
    def __init__(self, port):
        sel = selectors.DefaultSelector()

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        for _ in range(30):
            try:
                s.bind(('0.0.0.0', port))
                break
            except:
                print('Retrying...')
                time.sleep(5)
        s.listen()
        print('Created listener at %d' % port)
        s.setblocking(False)

        sel.register(s, selectors.EVENT_READ, self._accept)
        self.sel = sel
        self.listener = s
        self.senders = []

        self.parent_graph = None

    def set_parent_graph(self, g):
        self.parent_graph = g

    def _accept(self, s, mask):
        conn, addr = s.accept()
        print('Accepted connection', conn, 'from', addr)
        conn.setblocking(False)
        self.senders.append(conn)
        self.sel.register(conn, selectors.EVENT_READ, self._read)
        return None, None

    def _read(self, s, mask):
        aux_buffer_len, nf_buffer_len = np.frombuffer(recvall(s, 8, False), dtype='int32')
        if aux_buffer_len == 0 and nf_buffer_len == 0:
            print('Socket %s finished' % s)
            return None, None

        with recvall(s, aux_buffer_len, True) as bio:
            aux_data = pickle.load(bio)
        nf = dgl.network.deserialize_nodeflow(
                bytearray(recvall(s, nf_buffer_len, False)), self.parent_graph)
        return nf, aux_data

    def waitfor(self, n):
        for i in range(n):
            print('Waiting for connection %d/%d' % (i + 1, n))
            while True:
                try:
                    self._accept(self.listener, None)
                    break
                except socket.error as e:
                    err = e.args[0]
                    if err == errno.EAGAIN or err == errno.EWOULDBLOCK:
                        time.sleep(5)
                        continue
                    else:
                        raise e

    def distribute(self, data_list):
        data_segments = np.array_split(data_list, len(self.senders))
        data_segments = [seg.tolist() for seg in data_segments]
        for seg, s in zip(data_segments, self.senders):
            with io.BytesIO() as bio:
                pickle.dump(seg, bio)
                buf = bio.getvalue()
            with io.BytesIO() as bio:
                bio.write(np.array([len(buf)], dtype='int32').tobytes())
                bio.write(buf)
                sendall(s, bio.getvalue())

    def __iter__(self):
        try:
            completed = 0
            print('Number of senders:', len(self.senders))
            while completed < len(self.senders):
                events = self.sel.select()
                for key, mask in events:
                    callback = key.data
                    nf, aux_data = callback(key.fileobj, mask)
                    if nf is not None:
                        yield nf, aux_data
                    else:
                        completed += 1
            print('Iter finished')
        except Exception as e:
            raise e
            print('closing selector')
            self.sel.close()
