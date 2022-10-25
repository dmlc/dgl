import os

import numpy as np

import dgl
import dgl.backend as F
from dgl.distributed import load_partition_book

mode = os.environ.get("DIST_DGL_TEST_MODE", "")
graph_name = os.environ.get("DIST_DGL_TEST_GRAPH_NAME", "random_test_graph")
num_part = int(os.environ.get("DIST_DGL_TEST_NUM_PART"))
num_servers_per_machine = int(os.environ.get("DIST_DGL_TEST_NUM_SERVER"))
num_client_per_machine = int(os.environ.get("DIST_DGL_TEST_NUM_CLIENT"))
shared_workspace = os.environ.get("DIST_DGL_TEST_WORKSPACE")
graph_path = os.environ.get("DIST_DGL_TEST_GRAPH_PATH")
part_id = int(os.environ.get("DIST_DGL_TEST_PART_ID"))
net_type = os.environ.get("DIST_DGL_TEST_NET_TYPE")
ip_config = os.environ.get("DIST_DGL_TEST_IP_CONFIG", "ip_config.txt")

os.environ["DGL_DIST_MODE"] = "distributed"


def zeros_init(shape, dtype):
    return F.zeros(shape, dtype=dtype, ctx=F.cpu())


def run_server(
    graph_name,
    server_id,
    server_count,
    num_clients,
    shared_mem,
    keep_alive=False,
):
    # server_count = num_servers_per_machine
    g = dgl.distributed.DistGraphServer(
        server_id,
        ip_config,
        server_count,
        num_clients,
        graph_path + "/{}.json".format(graph_name),
        disable_shared_mem=not shared_mem,
        graph_format=["csc", "coo"],
        keep_alive=keep_alive,
        net_type=net_type,
    )
    print("start server", server_id)
    g.start()


##########################################
############### DistTensor ###############
##########################################


def dist_tensor_test_sanity(data_shape, name=None):
    local_rank = dgl.distributed.get_rank() % num_client_per_machine
    dist_ten = dgl.distributed.DistTensor(
        data_shape, F.int32, init_func=zeros_init, name=name
    )
    # arbitrary value
    stride = 3
    pos = (part_id // 2) * num_client_per_machine + local_rank
    if part_id % 2 == 0:
        dist_ten[pos * stride : (pos + 1) * stride] = F.ones(
            (stride, 2), dtype=F.int32, ctx=F.cpu()
        ) * (pos + 1)

    dgl.distributed.client_barrier()
    assert F.allclose(
        dist_ten[pos * stride : (pos + 1) * stride],
        F.ones((stride, 2), dtype=F.int32, ctx=F.cpu()) * (pos + 1),
    )


def dist_tensor_test_destroy_recreate(data_shape, name):
    dist_ten = dgl.distributed.DistTensor(
        data_shape, F.float32, name, init_func=zeros_init
    )
    del dist_ten

    dgl.distributed.client_barrier()

    new_shape = (data_shape[0], 4)
    dist_ten = dgl.distributed.DistTensor(
        new_shape, F.float32, name, init_func=zeros_init
    )


def dist_tensor_test_persistent(data_shape):
    dist_ten_name = "persistent_dist_tensor"
    dist_ten = dgl.distributed.DistTensor(
        data_shape,
        F.float32,
        dist_ten_name,
        init_func=zeros_init,
        persistent=True,
    )
    del dist_ten
    try:
        dist_ten = dgl.distributed.DistTensor(
            data_shape, F.float32, dist_ten_name
        )
        raise Exception("")
    except:
        pass


def test_dist_tensor(g):
    first_type = g.ntypes[0]
    data_shape = (g.number_of_nodes(first_type), 2)
    dist_tensor_test_sanity(data_shape)
    dist_tensor_test_sanity(data_shape, name="DistTensorSanity")
    dist_tensor_test_destroy_recreate(data_shape, name="DistTensorRecreate")
    dist_tensor_test_persistent(data_shape)


##########################################
############# DistEmbedding ##############
##########################################


def dist_embedding_check_sanity(num_nodes, optimizer, name=None):
    local_rank = dgl.distributed.get_rank() % num_client_per_machine

    emb = dgl.distributed.DistEmbedding(
        num_nodes, 1, name=name, init_func=zeros_init
    )
    lr = 0.001
    optim = optimizer(params=[emb], lr=lr)

    stride = 3

    pos = (part_id // 2) * num_client_per_machine + local_rank
    idx = F.arange(pos * stride, (pos + 1) * stride)

    if part_id % 2 == 0:
        with F.record_grad():
            value = emb(idx)
            optim.zero_grad()
            loss = F.sum(value + 1, 0)
        loss.backward()
        optim.step()

    dgl.distributed.client_barrier()
    value = emb(idx)
    F.allclose(value, F.ones((len(idx), 1), dtype=F.int32, ctx=F.cpu()) * -lr)

    not_update_idx = F.arange(
        ((num_part + 1) / 2) * num_client_per_machine * stride, num_nodes
    )
    value = emb(not_update_idx)
    assert np.all(F.asnumpy(value) == np.zeros((len(not_update_idx), 1)))


def dist_embedding_check_existing(num_nodes):
    dist_emb_name = "UniqueEmb"
    emb = dgl.distributed.DistEmbedding(
        num_nodes, 1, name=dist_emb_name, init_func=zeros_init
    )
    try:
        emb1 = dgl.distributed.DistEmbedding(
            num_nodes, 2, name=dist_emb_name, init_func=zeros_init
        )
        raise Exception("")
    except:
        pass


def test_dist_embedding(g):
    num_nodes = g.number_of_nodes(g.ntypes[0])
    dist_embedding_check_sanity(num_nodes, dgl.distributed.optim.SparseAdagrad)
    dist_embedding_check_sanity(
        num_nodes, dgl.distributed.optim.SparseAdagrad, name="SomeEmbedding"
    )
    dist_embedding_check_sanity(
        num_nodes, dgl.distributed.optim.SparseAdam, name="SomeEmbedding"
    )

    dist_embedding_check_existing(num_nodes)


def dist_optimizer_check_store(g, num_nodes):
    rank = dgl.distributed.get_rank()
    emb = dgl.distributed.DistEmbedding(
        num_nodes, 1, name='optimizer_test', init_func=zeros_init
    )
    emb_optimizer = dgl.distributed.optim.SparseAdam(
            [emb], lr=0.1
    )
    if rank == 0:
        name_to_state = {}
        for _, emb_states in emb_optimizer._state.items():
            for state in emb_states:
                name_to_state[state.name] = F.uniform(state.shape, F.float32, F.cpu(), 0, 1)
                state[F.arange(0, state.shape[0], F.int64, F.cpu())] = name_to_state[state.name]
        emb_optimizer.save_state_to('emb.pt')
        new_emb_optimizer = dgl.distributed.optim.SparseAdam(
            [emb], lr=000.1,  eps=2e-08, betas=(0.1, 0.222)
        )
        new_emb_optimizer.load_state_from('emb.pt')
        if rank == 0:
            for _, emb_states in new_emb_optimizer._state.items():
                for new_state in emb_states:
                    state = name_to_state[new_state.name]
                    new_state = new_state[F.arange(0, state.shape[0], F.int64, F.cpu())]
                    is_same = F.equal(state, new_state)
                    assert np.all(F.asnumpy(is_same))
            assert new_emb_optimizer._lr == emb_optimizer._lr
            assert new_emb_optimizer._eps == emb_optimizer._eps
            assert new_emb_optimizer._betas1 == emb_optimizer._betas1
            assert new_emb_optimizer._betas2 == emb_optimizer._betas2
        dgl.distributed.client_barrier()

def test_dist_optimizer(g):
    num_nodes = g.number_of_nodes(g.ntypes[0])
    dist_optimizer_check_store(g, num_nodes)

if mode == "server":
    shared_mem = bool(int(os.environ.get("DIST_DGL_TEST_SHARED_MEM")))
    server_id = int(os.environ.get("DIST_DGL_TEST_SERVER_ID"))
    run_server(
        graph_name,
        server_id,
        server_count=num_servers_per_machine,
        num_clients=num_part * num_client_per_machine,
        shared_mem=shared_mem,
        keep_alive=False,
    )
elif mode == "client":
    os.environ["DGL_NUM_SERVER"] = str(num_servers_per_machine)
    dgl.distributed.initialize(ip_config, net_type=net_type)

    gpb, graph_name, _, _ = load_partition_book(
        graph_path + "/{}.json".format(graph_name), part_id, None
    )
    g = dgl.distributed.DistGraph(graph_name, gpb=gpb)

    target_func_map = {
        "DistTensor": test_dist_tensor,
        "DistEmbedding": test_dist_embedding,
    }

    target = os.environ.get("DIST_DGL_TEST_OBJECT_TYPE", "")
    if target not in target_func_map:
        for test_func in target_func_map.values():
            test_func(g)
    else:
        target_func_map[target](g)

else:
    print("DIST_DGL_TEST_MODE has to be either server or client")
    exit(1)
