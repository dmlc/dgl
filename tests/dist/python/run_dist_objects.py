import os

import numpy as np

import dgl
import dgl.backend as F
import json
from dgl.distributed import load_partition_book, node_split, edge_split

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

def batched_assert_zero(tensor, size):
    BATCH_SIZE=2**16
    curr_pos = 0
    while curr_pos < size:
        end = min(curr_pos + BATCH_SIZE, size)
        assert F.sum(tensor[F.arange(curr_pos, end)], 0) == 0
        curr_pos = end

def zeros_init(shape, dtype):
    return F.zeros(shape, dtype=dtype, ctx=F.cpu())

def rand_init(shape, dtype):
    return F.tensor((np.random.randint(0, 100, size=shape) > 30), dtype=dtype)

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
############### DistGraph ###############
##########################################

def node_split_test(g, force_even, ntype='_N'):
    gpb = g.get_partition_book()

    selected_nodes_dist_tensor = dgl.distributed.DistTensor([g.number_of_nodes(ntype)], F.uint8, init_func=rand_init)

    nodes = node_split(selected_nodes_dist_tensor, gpb, ntype=ntype, force_even=force_even)
    g.barrier()

    selected_nodes_dist_tensor[nodes] = F.astype(F.zeros_like(nodes), selected_nodes_dist_tensor.dtype)
    g.barrier()

    if g.rank() == 0:
        batched_assert_zero(selected_nodes_dist_tensor, g.number_of_nodes(ntype))

    g.barrier()

def edge_split_test(g, force_even, etype='_E'):
    gpb = g.get_partition_book()

    selected_edges_dist_tensor = dgl.distributed.DistTensor([g.number_of_edges(etype)], F.uint8, init_func=rand_init)

    edges = edge_split(selected_edges_dist_tensor, gpb, etype=etype, force_even=force_even)
    g.barrier()

    selected_edges_dist_tensor[edges] = F.astype(F.zeros_like(edges), selected_edges_dist_tensor.dtype)
    g.barrier()

    if g.rank() == 0:
        batched_assert_zero(selected_edges_dist_tensor, g.number_of_edges(etype))

    g.barrier()

def test_dist_graph(g):
    gpb_path = graph_path + '/{}.json'.format(graph_name)
    with open(gpb_path) as conf_f:
        part_metadata = json.load(conf_f)
    assert 'num_nodes' in part_metadata
    assert 'num_edges' in part_metadata
    num_nodes = part_metadata['num_nodes']
    num_edges = part_metadata['num_edges']

    assert g.number_of_nodes() == num_nodes
    assert g.number_of_edges() == num_edges

    num_nodes = {ntype : g.num_nodes(ntype) for ntype in g.ntypes}
    num_edges = {etype : g.num_edges(etype) for etype in g.etypes}

    for key, n_nodes in num_nodes.items():
        assert g.number_of_nodes(key) == n_nodes
        node_split_test(g, force_even=False, ntype=key)
        node_split_test(g, force_even=True,  ntype=key)

    for key, n_edges in num_edges.items():
        assert g.number_of_edges(key) == n_edges
        edge_split_test(g, force_even=False, etype=key)
        edge_split_test(g, force_even=True,  etype=key)

##########################################
########### DistGraphServices ###########
##########################################
    nids = F.arange(0, 16)

    # Test in_degrees
    orig_in_degrees = g.ndata['in_degrees']
    local_in_degrees = g.in_degrees(nids)
    F.allclose(local_in_degrees, orig_in_degrees[nids])

    # Test out_degrees
    orig_out_degrees = g.ndata['out_degrees']
    local_out_degrees = g.out_degrees(nids)
    F.allclose(local_out_degrees, orig_out_degrees[nids])

    find_edges_test(g)
    edge_subgraph_test(g)
    sample_neighbors_test(g)

def find_edges_test(g, orig_nid_map):
    etypes = g.canonical_etypes

    etype_eids_uv_map = dict()
    for u_type, etype, v_type in etypes:
        orig_u =  g.edges[etype].data['edge_u']
        orig_v =  g.edges[etype].data['edge_v']
        eids = F.tensor(np.random.randint(g.number_of_edges(etype), size=100))
        u, v = g.find_edges(eids, etype=etype)
        assert F.allclose(orig_nid_map[u_type][u], orig_u[eids])
        assert F.allclose(orig_nid_map[v_type][v], orig_v[eids])
        etype_eids_uv_map[etype] = (eids, F.cat([u, v], dim=0))
    return etype_eids_uv_map

def edge_subgraph_test(g, etype_eids_uv_map):
    etypes = g.canonical_etypes
    all_eids = dict()
    for t in etypes:
        all_eids[t] = etype_eids_uv_map[t[1]][0]

    sg = g.edge_subgraph(all_eids)
    for t in etypes:
        assert sg.number_of_edges(t[1]) == len(all_eids[t])
        assert F.allclose(sg.edges[t].data[dgl.EID], all_eids[t])

    for u_type, etype, v_type in etypes:
        uv = etype_eids_uv_map[etype][1]
        sg_u_nids = sg.nodes[u_type].data[dgl.NID]
        sg_v_nids = sg.nodes[v_type].data[dgl.NID]
        sg_uv = F.cat([sg_u_nids, sg_v_nids], dim=0)
        for node_id in uv:
            assert node_id in sg_uv

def sample_neighbors_with_args(g, size, fanout):
    num_nodes = {ntype : g.num_nodes(ntype) for ntype in g.ntypes}
    etypes = g.canonical_etypes

    sampled_graph = g.sample_neighbors({ntype : np.random.randint(0, n, size=size) for ntype, n in num_nodes.items()}, fanout)

    for ntype, n in num_nodes.items():
        assert sampled_graph.number_of_nodes(ntype) == n
    for t in etypes:
        src, dst = sampled_graph.edges(etype=t)
        eids = sampled_graph.edges[t].data[dgl.EID]
        dist_u, dist_v = g.find_edges(eids, etype=t[1])
        assert F.allclose(dist_u, src)
        assert F.allclose(dist_v, dst)

def sample_neighbors_test(g):
    sample_neighbors_with_args(g, size=1024, fanout=3)
    sample_neighbors_with_args(g, size=1, fanout=10)
    sample_neighbors_with_args(g, size=1024, fanout=2)
    sample_neighbors_with_args(g, size=10, fanout=-1)
    sample_neighbors_with_args(g, size=2**10, fanout=1)
    sample_neighbors_with_args(g, size=2**12, fanout=1)

def test_dist_graph_services(g):
    num_nodes = {ntype : g.num_nodes(ntype) for ntype in g.ntypes}

    orig_nid_map = dict()
    dtype =  g.edges[g.etypes[0]].data['edge_u'].dtype
    for ntype, _ in num_nodes.items():
        orig_nid = F.tensor(np.load(graph_path + f'/orig_nid_array_{ntype}.npy'), dtype)
        orig_nid_map[ntype] = orig_nid

    etype_eids_uv_map = find_edges_test(g, orig_nid_map)
    edge_subgraph_test(g, etype_eids_uv_map)
    sample_neighbors_test(g)

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
        dist_ten[pos * stride: (pos + 1) * stride] = F.ones(
            (stride, 2), dtype=F.int32, ctx=F.cpu()
        ) * (pos + 1)

    dgl.distributed.client_barrier()
    assert F.allclose(
        dist_ten[pos * stride: (pos + 1) * stride],
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
    except BaseException:
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
    except BaseException:
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


##########################################
############# DistOptimizer ##############
##########################################


def dist_optimizer_check_store(g):
    num_nodes = g.number_of_nodes(g.ntypes[0])
    rank = g.rank()
    try:
        emb = dgl.distributed.DistEmbedding(
            num_nodes, 1, name="optimizer_test", init_func=zeros_init
        )
        emb2 = dgl.distributed.DistEmbedding(
            num_nodes, 5, name="optimizer_test2", init_func=zeros_init
        )
        emb_optimizer = dgl.distributed.optim.SparseAdam([emb, emb2], lr=0.1)
        if rank == 0:
            name_to_state = {}
            for _, emb_states in emb_optimizer._state.items():
                for state in emb_states:
                    name_to_state[state.name] = F.uniform(
                        state.shape, F.float32, F.cpu(), 0, 1
                    )
                    state[
                        F.arange(0, num_nodes, F.int64, F.cpu())
                    ] = name_to_state[state.name]
        emb_optimizer.save("emb.pt")
        new_emb_optimizer = dgl.distributed.optim.SparseAdam(
            [emb, emb2], lr=000.1, eps=2e-08, betas=(0.1, 0.222)
        )
        new_emb_optimizer.load("emb.pt")
        if rank == 0:
            for _, emb_states in new_emb_optimizer._state.items():
                for new_state in emb_states:
                    state = name_to_state[new_state.name]
                    new_state = new_state[
                        F.arange(0, num_nodes, F.int64, F.cpu())
                    ]
                    assert F.allclose (state, new_state, 0., 0.)
            assert new_emb_optimizer._lr == emb_optimizer._lr
            assert new_emb_optimizer._eps == emb_optimizer._eps
            assert new_emb_optimizer._beta1 == emb_optimizer._beta1
            assert new_emb_optimizer._beta2 == emb_optimizer._beta2
        g.barrier()
    finally:
        file = f'emb.pt_{rank}'
        if os.path.exists(file):
            os.remove(file)

def test_dist_optimizer(g):
    dist_optimizer_check_store(g)


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
        "DistGraph": test_dist_graph,
        "DistGraphServices": test_dist_graph_services,
        "DistTensor": test_dist_tensor,
        "DistEmbedding": test_dist_embedding,
        "DistOptimizer": test_dist_optimizer,
    }

    target = os.environ.get("DIST_DGL_TEST_OBJECT_TYPE", "")
    if target not in target_func_map:
        for test_func in target_func_map.values():
            test_func(g)
    else:
        target_func_map[target](g)

else:
    exit(1)
