import numpy as np
import time

N = 1000
D = 16

class PerfMsgModule:
    def __init__(self):
        self.weight = np.random.randn(D, D)

    def __call__(self, node_states, edge_states):
        msg = 2 * np.dot(node_states['h'], self.weight) + 1
        return msg

def perf_batched_msgs():
    msg_mod = PerfMsgModule()
    # send from N nodes, each node has a feature vector of size D 

    # Non-batched version:
    ns = [{'h': np.zeros((1, D))} for i in range(N)]
    t0 = time.time()
    for k in range(10):
        msgs = []
        for i in range(N):
            msgs.append(msg_mod(ns[i], None))
    print('non-batched average time:', (time.time() - t0) / 10)

    # Semi-batched version:
    ns = [{'h': np.zeros((1, D))} for i in range(N)]
    t0 = time.time()
    for k in range(10):
        nss = np.concatenate([n['h'] for n in ns], axis=0)
        msgs = msg_mod({'h': nss}, None)
    print('semi-batched average time:', (time.time() - t0) / 10)

    # Fully-Batched version:
    ns = {'h': np.zeros((N, D))}
    t0 = time.time()
    for k in range(10):
        msg = msg_mod(ns, None)
    print('fully-batched average time:', (time.time() - t0) / 10)

perf_batched_msgs()
