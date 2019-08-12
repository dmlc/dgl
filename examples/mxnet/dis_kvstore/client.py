import dgl
import mxnet as mx
import time
import argparse

# In this example, we have 2 kvclient and 2 kvserver
client_namebook = { 0:'127.0.0.1:50051',
                    1:'127.0.0.1:50052' }

server_namebook = { 0:'127.0.0.1:50053',
                    1:'127.0.0.1:50054' }

def start_client(args):
    client = dgl.contrib.KVClient(
        client_id=args.id, 
        server_namebook=server_namebook, 
        client_addr=client_namebook[args.id])

    client.connect()

    client.init_data('embed', [10, 3], 0.0, 0.0)

    tensor_id = mx.nd.array([0, 1, 2], dtype='int64')
    tensor_data = mx.nd.array([[0., 0., 0., ], [1., 1., 1.], [2., 2., 2.]])

    for i in range(5):
        client.push('embed', tensor_id, tensor_data)

    tensor_id = mx.nd.array([6, 7, 8], dtype='int64')
    for i in range(5):
        client.push('embed', tensor_id, tensor_data)

    # wait all push() done
    time.sleep(1)

    tensor_id = mx.nd.array([0, 1, 2, 6, 7, 8], dtype='int64')
    new_tensor = client.pull('embed', tensor_id)
    print(new_tensor)

    if client.get_id() == 0:
        client.shut_down()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='kvstore')
    parser.add_argument("--id", type=int, default=0,
            help="node ID")
    args = parser.parse_args()

    time.sleep(2)  # wait server start
    start_client(args)
