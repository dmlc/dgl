import os
import dgl
from dgl.distributed.rpc import RPCMessage, send_rpc_message, recv_rpc_message
import pickle
import torch as th
import dgl.backend as F
import time
import cProfile
import argparse


def main():
    os.environ['DGL_DIST_MODE'] = 'distributed'
    ip_config = "rpc_ip_config.txt"
    dgl.distributed.connect_to_server(ip_config=ip_config, num_servers=1)
    data = bytearray(pickle.dumps(None))

    tensor_size_list = [4, 8, 32, 128, 512,
                        4096, 16384, 65536, 262144, 1048576, 4194304]

    # warm up

    tensors = [th.ones((100 // 4, ), dtype=th.float32)
               for _ in range(4)]
    msg = RPCMessage(-1, 0, 0, 0, data, tensors)
    for i in range(100):
        send_rpc_message(msg, 0)
        recv_rpc_message(0)

    start_time = time.time()
    records = {}
    for tensor_size in tensor_size_list:
        if tensor_size > 65536:
            iterations = 100
        else:
            iterations = 1000
        
        tensors = [th.ones((tensor_size // 4, ), dtype=th.float32)
                for _ in range(4)]
        msg = RPCMessage(-1, 0, 0, 0, data, tensors)
        start_time = time.time()
        for i in range(iterations):
            send_rpc_message(msg, 0)
            recv_rpc_message(0)
        time_elapsed = time.time() - start_time
        print(time_elapsed / iterations)
        per_iter_time = time_elapsed / iterations
        bandwidth = tensor_size*8/per_iter_time*2/1024/1024/1024*8
        records[tensor_size] = {"time": per_iter_time, "bandwidth": bandwidth}
        print(
            f"Finished {tensor_size=}, {time_elapsed=}, Bandwidth: {bandwidth: .5f} Gbps")
    # pr.disable()
    # pr.create_stats()
    # stats = pr.stats
    # time_elapsed = time.time() - start_time
    # print(time_elapsed / iterations)
    # print(stats)
    print(records)


parser = argparse.ArgumentParser()
parser.add_argument("-tensor-size", type=int, default=1)
parser.add_argument("-iters", type=int, default=100)
args = parser.parse_args()

main()
