# This is a simple pytorch client demo shows how to use DGL distributed kvstore.
# In this demo, we initialize two embeddings on server and push/pull data to/from it.
import dgl
import time
import argparse
import torch as th

server_namebook, client_namebook = dgl.contrib.ReadNetworkConfigure('config.txt')

def start_client(args):
    # Initialize client and connect to server
    client = dgl.contrib.KVClient(
        client_id=args.id, 
        server_namebook=server_namebook, 
        client_addr=client_namebook[args.id])

    client.connect()

    # Initialize data on server
    client.init_data(name='embed_0', server_id=0, shape=[5, 3], init_type='zero')
    client.init_data(name='embed_0', server_id=1, shape=[6, 3], init_type='zero')
    client.init_data(name='embed_1', server_id=0, shape=[5], init_type='uniform', low=0.0, high=0.0)
    client.init_data(name='embed_1', server_id=1, shape=[6], init_type='uniform', low=0.0, high=0.0)

    data_0 = th.tensor([[0., 0., 0., ], [1., 1., 1.], [2., 2., 2.]])
    data_1 = th.tensor([0., 1., 2.])

    for i in range(5):
        client.push(name='embed_0', server_id=0, id_tensor=th.tensor([0, 2, 4]), data_tensor=data_0)
        client.push(name='embed_0', server_id=1, id_tensor=th.tensor([1, 3, 5]), data_tensor=data_0)
        client.push(name='embed_1', server_id=0, id_tensor=th.tensor([0, 2, 4]), data_tensor=data_1)
        client.push(name='embed_1', server_id=1, id_tensor=th.tensor([1, 3, 5]), data_tensor=data_1)
        client.push(name='server_embed', server_id=0, id_tensor=th.tensor([0, 2, 4]), data_tensor=data_1)
        client.push(name='server_embed', server_id=1, id_tensor=th.tensor([0, 2, 4]), data_tensor=data_1)

    client.barrier()

    if client.get_id() == 0:
        client.pull(name='embed_0', server_id=0, id_tensor=th.tensor([0, 1, 2, 3, 4]))
        msg_0 = client.pull_wait()
        assert msg_0.rank == 0
        client.pull(name='embed_0', server_id=1, id_tensor=th.tensor([0, 1, 2, 3, 4, 5]))
        msg_1 = client.pull_wait()
        assert msg_1.rank == 1
        print("embed_0:")
        print(th.cat([msg_0.data, msg_1.data]))

        client.pull(name='embed_1', server_id=0, id_tensor=th.tensor([0, 1, 2, 3, 4]))
        msg_0 = client.pull_wait()
        assert msg_0.rank == 0
        client.pull(name='embed_1', server_id=1, id_tensor=th.tensor([0, 1, 2, 3, 4, 5]))
        msg_1 = client.pull_wait()
        assert msg_1.rank == 1
        print("embed_1:")
        print(th.cat([msg_0.data, msg_1.data]))

        client.pull(name='server_embed', server_id=0, id_tensor=th.tensor([0, 1, 2, 3, 4]))
        msg_0 = client.pull_wait()
        assert msg_0.rank == 0
        client.pull(name='server_embed', server_id=1, id_tensor=th.tensor([0, 1, 2, 3, 4]))
        msg_1 = client.pull_wait()
        assert msg_1.rank == 1
        print("server_embed:")
        print(th.cat([msg_0.data, msg_1.data]))

    # Shut-down all the servers
    if client.get_id() == 0:
        client.shut_down()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='kvstore')
    parser.add_argument("--id", type=int, default=0, help="node ID")
    args = parser.parse_args()
    time.sleep(2)  # wait server start
    start_client(args)
