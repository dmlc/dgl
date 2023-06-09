import os

import backend as F

import dgl
from numpy.testing import assert_array_equal

INTEGER = 2
STR = "hello world!"
HELLO_SERVICE_ID = 901231
TENSOR = F.zeros((1000, 1000), F.int64, F.cpu())


def tensor_func(tensor):
    return tensor * 2


class HelloResponse(dgl.distributed.Response):
    def __init__(self, hello_str, integer, tensor):
        self.hello_str = hello_str
        self.integer = integer
        self.tensor = tensor

    def __getstate__(self):
        return self.hello_str, self.integer, self.tensor

    def __setstate__(self, state):
        self.hello_str, self.integer, self.tensor = state


class HelloRequest(dgl.distributed.Request):
    def __init__(self, hello_str, integer, tensor, func):
        self.hello_str = hello_str
        self.integer = integer
        self.tensor = tensor
        self.func = func

    def __getstate__(self):
        return self.hello_str, self.integer, self.tensor, self.func

    def __setstate__(self, state):
        self.hello_str, self.integer, self.tensor, self.func = state

    def process_request(self, server_state):
        assert self.hello_str == STR
        assert self.integer == INTEGER
        new_tensor = self.func(self.tensor)
        res = HelloResponse(self.hello_str, self.integer, new_tensor)
        return res


def start_server(server_id, ip_config, num_servers, num_clients, keep_alive):
    server_state = dgl.distributed.ServerState(
        None, local_g=None, partition_book=None, keep_alive=keep_alive
    )
    dgl.distributed.register_service(
        HELLO_SERVICE_ID, HelloRequest, HelloResponse
    )
    print("Start server {}".format(server_id))
    dgl.distributed.start_server(
        server_id=server_id,
        ip_config=ip_config,
        num_servers=num_servers,
        num_clients=num_clients,
        server_state=server_state,
    )


def start_client(ip_config, num_servers, group_id):
    dgl.distributed.register_service(
        HELLO_SERVICE_ID, HelloRequest, HelloResponse
    )
    dgl.distributed.connect_to_server(
        ip_config=ip_config,
        num_servers=num_servers,
        group_id=group_id,
    )
    req = HelloRequest(STR, INTEGER, TENSOR, tensor_func)
    server_namebook = dgl.distributed.read_ip_config(ip_config, num_servers)
    for server_id in server_namebook.keys():
        # test send and recv
        dgl.distributed.send_request(server_id, req)
        res = dgl.distributed.recv_response()
        assert res.hello_str == STR
        assert res.integer == INTEGER
        assert_array_equal(F.asnumpy(res.tensor), F.asnumpy(TENSOR))
        # test remote_call
        target_and_requests = []
        for i in range(10):
            target_and_requests.append((server_id, req))
        res_list = dgl.distributed.remote_call(target_and_requests)
        for res in res_list:
            assert res.hello_str == STR
            assert res.integer == INTEGER
            assert_array_equal(F.asnumpy(res.tensor), F.asnumpy(TENSOR))
        # test send_request_to_machine
        dgl.distributed.send_request_to_machine(server_id, req)
        res = dgl.distributed.recv_response()
        assert res.hello_str == STR
        assert res.integer == INTEGER
        assert_array_equal(F.asnumpy(res.tensor), F.asnumpy(TENSOR))
        # test remote_call_to_machine
        target_and_requests = []
        for i in range(10):
            target_and_requests.append((server_id, req))
        res_list = dgl.distributed.remote_call_to_machine(target_and_requests)
        for res in res_list:
            assert res.hello_str == STR
            assert res.integer == INTEGER
            assert_array_equal(F.asnumpy(res.tensor), F.asnumpy(TENSOR))


def main():
    ip_config = os.environ.get("DIST_DGL_TEST_IP_CONFIG")
    num_servers = int(os.environ.get("DIST_DGL_TEST_NUM_SERVERS"))
    if os.environ.get("DIST_DGL_TEST_ROLE", "server") == "server":
        server_id = int(os.environ.get("DIST_DGL_TEST_SERVER_ID"))
        num_clients = int(os.environ.get("DIST_DGL_TEST_NUM_CLIENTS"))
        keep_alive = "DIST_DGL_TEST_KEEP_ALIVE" in os.environ
        start_server(server_id, ip_config, num_servers, num_clients, keep_alive)
    else:
        group_id = int(os.environ.get("DIST_DGL_TEST_GROUP_ID", "0"))
        start_client(ip_config, num_servers, group_id)


if __name__ == "__main__":
    main()
