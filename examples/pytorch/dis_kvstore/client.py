# This is a simple pytorch client demo shows how to use DGL distributed kvstore.
# In this demo, we initialize two embeddings on server and push/pull data to/from it.
import dgl
import torch
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

    tensor_id = torch.tensor([0, 1, 2])
    tensor_data = torch.tensor([[0., 0., 0., ], [1., 1., 1.], [2., 2., 2.]])

    # Push data
    for i in range(5):
        client.push('embed_0', tensor_id, tensor_data)
        client.push('embed_1', tensor_id, tensor_data)

    tensor_id = torch.tensor([6, 7, 8])
    for i in range(5):
        client.push('embed_0', tensor_id, tensor_data)
        client.push('embed_1', tensor_id, tensor_data)

    time.sleep(1) # wait all Push done

    # Pull data
    if client.get_id() == 0:
        tensor_id = torch.tensor([0, 1, 2, 6, 7, 8])
        new_tensor_0 = client.pull('embed_0', tensor_id)
        new_tensor_1 = client.pull('embed_1', tensor_id)
        print("Tensor_0:")
        print(new_tensor_0)
        print("Tensor_1")
        print(new_tensor_1)

    # Shut-down all the servers
    if client.get_id() == 0:
        client.shut_down()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='kvstore')
    parser.add_argument("--id", type=int, default=0, help="node ID")
    args = parser.parse_args()
    time.sleep(2)  # wait server start
    start_client(args)
