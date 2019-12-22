# This is a simple MXNet server demo shows how to use DGL distributed kvstore.
# In this demo, we initialize two embeddings on server and push/pull data to/from it.
import dgl
import argparse
import mxnet as mx
import time

server_namebook = dgl.contrib.read_ip_config('ip_config.txt')

global2local = []
global2local.append(mx.nd.array([0,1,0,0,0,0,0,0], dtype='int64'))
global2local.append(mx.nd.array([0,0,0,1,0,0,0,0], dtype='int64'))
global2local.append(mx.nd.array([0,0,0,0,0,1,0,0], dtype='int64'))
global2local.append(mx.nd.array([0,0,0,0,0,0,0,1], dtype='int64'))

ID = []
ID.append(mx.nd.array([0,1], dtype='int64'))
ID.append(mx.nd.array([2,3], dtype='int64'))
ID.append(mx.nd.array([4,5], dtype='int64'))
ID.append(mx.nd.array([6,7], dtype='int64'))

partition_book = mx.nd.array([0,0,1,1,2,2,3,3], dtype='int64')

def start_client(args):
    client = dgl.contrib.KVClient(
        server_namebook=server_namebook, 
        client_id=args.id)

    client.set_partition_book(name='embed', partition_book=partition_book)

    client.connect()

    print("Client %d connected to kvstore ..." % args.id)
    
    client.push(name='embed', id_tensor=ID[args.id], data_tensor=mx.nd.array([[1.,1.,1.],[1.,1.,1.]]))

    client.barrier()

    tensor = client.pull(name='embed', id_tensor=mx.nd.array([0,1,2,3,4,5,6,7], dtype='int64'))

    print(tensor)

    client.barrier()

    if args.id == 0:
        client.shut_down()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='kvstore')
    parser.add_argument("--id", type=int, default=0, help="node ID")
    args = parser.parse_args()

    start_client(args)