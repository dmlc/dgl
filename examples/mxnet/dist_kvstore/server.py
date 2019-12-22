# This is a simple MXNet server demo shows how to use DGL distributed kvstore.
# In this demo, we initialize two embeddings on server and push/pull data to/from it.
import dgl
import argparse
import mxnet as mx

server_namebook = dgl.contrib.read_ip_config('ip_config.txt')

global2local = []
global2local.append(mx.nd.array([0,1,0,0,0,0,0,0], dtype='int64'))
global2local.append(mx.nd.array([0,0,0,1,0,0,0,0], dtype='int64'))
global2local.append(mx.nd.array([0,0,0,0,0,1,0,0], dtype='int64'))
global2local.append(mx.nd.array([0,0,0,0,0,0,0,1], dtype='int64'))

def start_server(args):
    server = dgl.contrib.KVServer(
        server_id=args.id, 
        server_addr=server_namebook[args.id],
        num_client=4)

    server.init_data(name='embed', data_tensor=mx.nd.array([[0.,0.,0.],[0.,0.,0.]]))

    server.set_global2local(name='embed', global2local=global2local[args.id])

    print("start server %d" % args.id)

    server.start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='kvstore')
    parser.add_argument("--id", type=int, default=0, help="node ID")
    args = parser.parse_args()

    start_server(args)