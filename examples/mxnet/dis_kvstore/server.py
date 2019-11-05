# This is a simple MXNet server demo shows how to use DGL distributed kvstore.
# In this demo, we initialize two embeddings on server and push/pull data to/from it.
import dgl
import torch
import argparse
import mxnet as mx

server_namebook, client_namebook = dgl.contrib.ReadNetworkConfigure('config.txt')

def start_server(args):
    server = dgl.contrib.KVServer(
        server_id=args.id, 
        client_namebook=client_namebook, 
        server_addr=server_namebook[args.id])

    server.init_data(name='server_embed', data_tensor=mx.nd.array([0., 0., 0., 0., 0.]))

    server.start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='kvstore')
    parser.add_argument("--id", type=int, default=0, help="node ID")
    args = parser.parse_args()

    start_server(args)
