# This is a simple MXNet client demo shows how to use DGL distributed kvstore.
# In this demo, we initialize two embeddings on server and push/pull data to/from it.
import dgl
import mxnet as mx
import time
import argparse

server_namebook, client_namebook = dgl.contrib.ReadNetworkConfigure('config.txt')

def start_client(args):
    # Initialize client and connect to server
    client = dgl.contrib.KVClient(
        client_id=args.id, 
        server_namebook=server_namebook, 
        client_addr=client_namebook[args.id])

    client.connect()

    # Initialize data on server
    client.init_data(name='embed_0', shape=[10, 3], init_type='zero')
    client.init_data(name='embed_1', shape=[11, 3], init_type='uniform', low=0.0, high=0.0)
    client.init_data(name='embed_2', shape=[11], init_type='zero')

    tensor_id = mx.nd.array([0, 1, 2], dtype='int64')
    tensor_data = mx.nd.array([[0., 0., 0., ], [1., 1., 1.], [2., 2., 2.]])

    for i in range(5):
        client.push('embed_0', tensor_id, tensor_data)
        client.push('embed_1', tensor_id, tensor_data)
        client.push('embed_2', tensor_id, mx.nd.array([2., 2., 2.]))

    tensor_id = mx.nd.array([6, 7, 8], dtype='int64')
    for i in range(5):
        client.push('embed_0', tensor_id, tensor_data)
        client.push('embed_1', tensor_id, tensor_data)
        client.push('embed_2', tensor_id, mx.nd.array([3., 3., 3.]))

    client.barrier()

    if client.get_id() == 0:
        tensor_id = mx.nd.array([0,1,2,3,4,5,6,7,8,9], dtype='int64')
        new_tensor_0 = client.pull('embed_0', tensor_id)
        tensor_id = mx.nd.array([0,1,2,3,4,5,6,7,8,9,10], dtype='int64')
        new_tensor_1 = client.pull('embed_1', tensor_id)
        new_tensor_2 = client.pull('embed_2', tensor_id)

        client.push_all('embed_0', new_tensor_0)
        client.push_all('embed_1', new_tensor_1)
        client.push_all('embed_2', new_tensor_2)

        new_tensor_3 = client.pull_all('embed_0')
        new_tensor_4 = client.pull_all('embed_1')
        new_tensor_5 = client.pull_all('embed_2')
        print("embed_0: ")
        print(new_tensor_3)
        print("embed_1: ")
        print(new_tensor_4)
        print("embed_2: ")
        print(new_tensor_5)

    # Shut-down all the servers
    if client.get_id() == 0:
        client.shut_down()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='kvstore')
    parser.add_argument("--id", type=int, default=0, help="node ID")
    args = parser.parse_args()
    time.sleep(2)  # wait server start
    start_client(args)
