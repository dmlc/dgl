import dgl
import torch
import argparse

# In this example, we have 2 kvclient and 2 kvserver
client_namebook = { 0:'127.0.0.1:50051',
                    1:'127.0.0.1:50052' }

server_namebook = { 0:'127.0.0.1:50053',
                    1:'127.0.0.1:50054' }

def start_server(args):
    server = dgl.contrib.KVServer(
        server_id=args.id, 
        client_namebook=client_namebook, 
        server_addr=server_namebook[args.id])

    server.start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='kvstore')
    parser.add_argument("--id", type=int, default=0,
            help="node ID")
    args = parser.parse_args()

    start_server(args)
