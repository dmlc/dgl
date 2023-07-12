import multiprocessing as mp
import os
import socket
import time
import unittest

import backend as F

import dgl
import pytest
from numpy.testing import assert_array_equal
from utils import generate_ip_config, reset_envs

if os.name != "nt":
    import fcntl
    import struct

INTEGER = 2
STR = "hello world!"
HELLO_SERVICE_ID = 901231
TENSOR = F.zeros((1000, 1000), F.int64, F.cpu())


def foo(x, y):
    assert x == 123
    assert y == "abc"


class MyRequest(dgl.distributed.Request):
    def __init__(self):
        self.x = 123
        self.y = "abc"
        self.z = F.randn((3, 4))
        self.foo = foo

    def __getstate__(self):
        return self.x, self.y, self.z, self.foo

    def __setstate__(self, state):
        self.x, self.y, self.z, self.foo = state

    def process_request(self, server_state):
        pass


class MyResponse(dgl.distributed.Response):
    def __init__(self):
        self.x = 432

    def __getstate__(self):
        return self.x

    def __setstate__(self, state):
        self.x = state


def simple_func(tensor):
    return tensor


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


TIMEOUT_SERVICE_ID = 123456789
TIMEOUT_META = "timeout_test"


class TimeoutResponse(dgl.distributed.Response):
    def __init__(self, meta):
        self.meta = meta

    def __getstate__(self):
        return self.meta

    def __setstate__(self, state):
        self.meta = state


class TimeoutRequest(dgl.distributed.Request):
    def __init__(self, meta, timeout, response=True):
        self.meta = meta
        self.timeout = timeout
        self.response = response

    def __getstate__(self):
        return self.meta, self.timeout, self.response

    def __setstate__(self, state):
        self.meta, self.timeout, self.response = state

    def process_request(self, server_state):
        assert self.meta == TIMEOUT_META
        # convert from milliseconds to seconds
        time.sleep(self.timeout / 1000)
        if not self.response:
            return None
        res = TimeoutResponse(self.meta)
        return res


def start_server(
    num_clients,
    ip_config,
    server_id=0,
    num_servers=1,
):
    print("Sleep 1 seconds to test client re-connect.")
    time.sleep(1)
    server_state = dgl.distributed.ServerState(
        None, local_g=None, partition_book=None
    )
    dgl.distributed.register_service(
        HELLO_SERVICE_ID, HelloRequest, HelloResponse
    )
    dgl.distributed.register_service(
        TIMEOUT_SERVICE_ID, TimeoutRequest, TimeoutResponse
    )
    print("Start server {}".format(server_id))
    dgl.distributed.start_server(
        server_id=server_id,
        ip_config=ip_config,
        num_servers=num_servers,
        num_clients=num_clients,
        server_state=server_state,
    )


def start_client(ip_config, group_id=0, num_servers=1):
    dgl.distributed.register_service(
        HELLO_SERVICE_ID, HelloRequest, HelloResponse
    )
    dgl.distributed.connect_to_server(
        ip_config=ip_config,
        num_servers=num_servers,
        group_id=group_id,
    )
    req = HelloRequest(STR, INTEGER, TENSOR, simple_func)
    # test send and recv
    dgl.distributed.send_request(0, req)
    res = dgl.distributed.recv_response()
    assert res.hello_str == STR
    assert res.integer == INTEGER
    assert_array_equal(F.asnumpy(res.tensor), F.asnumpy(TENSOR))
    # test remote_call
    target_and_requests = []
    for i in range(10):
        target_and_requests.append((0, req))
    res_list = dgl.distributed.remote_call(target_and_requests)
    for res in res_list:
        assert res.hello_str == STR
        assert res.integer == INTEGER
        assert_array_equal(F.asnumpy(res.tensor), F.asnumpy(TENSOR))
    # test send_request_to_machine
    dgl.distributed.send_request_to_machine(0, req)
    res = dgl.distributed.recv_response()
    assert res.hello_str == STR
    assert res.integer == INTEGER
    assert_array_equal(F.asnumpy(res.tensor), F.asnumpy(TENSOR))
    # test remote_call_to_machine
    target_and_requests = []
    for i in range(10):
        target_and_requests.append((0, req))
    res_list = dgl.distributed.remote_call_to_machine(target_and_requests)
    for res in res_list:
        assert res.hello_str == STR
        assert res.integer == INTEGER
        assert_array_equal(F.asnumpy(res.tensor), F.asnumpy(TENSOR))


def start_client_timeout(ip_config, group_id=0, num_servers=1):
    dgl.distributed.register_service(
        TIMEOUT_SERVICE_ID, TimeoutRequest, TimeoutResponse
    )
    dgl.distributed.connect_to_server(
        ip_config=ip_config,
        num_servers=num_servers,
        group_id=group_id,
    )
    timeout = 1 * 1000  # milliseconds
    req = TimeoutRequest(TIMEOUT_META, timeout)
    # test send and recv
    dgl.distributed.send_request(0, req)
    res = dgl.distributed.recv_response(timeout=int(timeout / 2))
    assert res is None
    res = dgl.distributed.recv_response()
    assert res.meta == TIMEOUT_META
    # test remote_call
    req = TimeoutRequest(TIMEOUT_META, timeout, response=False)
    target_and_requests = []
    for i in range(3):
        target_and_requests.append((0, req))
    expect_except = False
    try:
        res_list = dgl.distributed.remote_call(
            target_and_requests, timeout=int(timeout / 2)
        )
    except dgl.DGLError:
        expect_except = True
    assert expect_except
    # test send_request_to_machine
    req = TimeoutRequest(TIMEOUT_META, timeout)
    dgl.distributed.send_request_to_machine(0, req)
    res = dgl.distributed.recv_response(timeout=int(timeout / 2))
    assert res is None
    res = dgl.distributed.recv_response()
    assert res.meta == TIMEOUT_META
    # test remote_call_to_machine
    req = TimeoutRequest(TIMEOUT_META, timeout, response=False)
    target_and_requests = []
    for i in range(3):
        target_and_requests.append((0, req))
    expect_except = False
    try:
        res_list = dgl.distributed.remote_call_to_machine(
            target_and_requests, timeout=int(timeout / 2)
        )
    except dgl.DGLError:
        expect_except = True
    assert expect_except


@unittest.skipIf(os.name == "nt", reason="Do not support windows yet")
def test_rpc_timeout():
    reset_envs()
    os.environ["DGL_DIST_MODE"] = "distributed"
    ip_config = "rpc_ip_config.txt"
    generate_ip_config(ip_config, 1, 1)
    ctx = mp.get_context("spawn")
    pserver = ctx.Process(target=start_server, args=(1, ip_config, 0, 1))
    pclient = ctx.Process(target=start_client_timeout, args=(ip_config, 0, 1))
    pserver.start()
    pclient.start()
    pserver.join()
    pclient.join()


def test_serialize():
    reset_envs()
    os.environ["DGL_DIST_MODE"] = "distributed"
    from dgl.distributed.rpc import (
        deserialize_from_payload,
        serialize_to_payload,
    )

    SERVICE_ID = 12345
    dgl.distributed.register_service(SERVICE_ID, MyRequest, MyResponse)
    req = MyRequest()
    data, tensors = serialize_to_payload(req)
    req1 = deserialize_from_payload(MyRequest, data, tensors)
    req1.foo(req1.x, req1.y)
    assert req.x == req1.x
    assert req.y == req1.y
    assert F.array_equal(req.z, req1.z)

    res = MyResponse()
    data, tensors = serialize_to_payload(res)
    res1 = deserialize_from_payload(MyResponse, data, tensors)
    assert res.x == res1.x


def test_rpc_msg():
    reset_envs()
    os.environ["DGL_DIST_MODE"] = "distributed"
    from dgl.distributed.rpc import (
        deserialize_from_payload,
        RPCMessage,
        serialize_to_payload,
    )

    SERVICE_ID = 32452
    dgl.distributed.register_service(SERVICE_ID, MyRequest, MyResponse)
    req = MyRequest()
    data, tensors = serialize_to_payload(req)
    rpcmsg = RPCMessage(SERVICE_ID, 23, 0, 1, data, tensors)
    assert rpcmsg.service_id == SERVICE_ID
    assert rpcmsg.msg_seq == 23
    assert rpcmsg.client_id == 0
    assert rpcmsg.server_id == 1
    assert len(rpcmsg.data) == len(data)
    assert len(rpcmsg.tensors) == 1
    assert F.array_equal(rpcmsg.tensors[0], req.z)


@unittest.skipIf(os.name == "nt", reason="Do not support windows yet")
def test_multi_client():
    reset_envs()
    os.environ["DGL_DIST_MODE"] = "distributed"
    ip_config = "rpc_ip_config_mul_client.txt"
    generate_ip_config(ip_config, 1, 1)
    ctx = mp.get_context("spawn")
    num_clients = 20
    pserver = ctx.Process(
        target=start_server,
        args=(num_clients, ip_config, 0, 1),
    )
    pclient_list = []
    for i in range(num_clients):
        pclient = ctx.Process(target=start_client, args=(ip_config, 0, 1))
        pclient_list.append(pclient)
    pserver.start()
    for i in range(num_clients):
        pclient_list[i].start()
    for i in range(num_clients):
        pclient_list[i].join()
    pserver.join()


@unittest.skipIf(os.name == "nt", reason="Do not support windows yet")
def test_multi_thread_rpc():
    reset_envs()
    os.environ["DGL_DIST_MODE"] = "distributed"
    num_servers = 2
    ip_config = "rpc_ip_config_multithread.txt"
    generate_ip_config(ip_config, num_servers, num_servers)
    ctx = mp.get_context("spawn")
    pserver_list = []
    for i in range(num_servers):
        pserver = ctx.Process(target=start_server, args=(1, ip_config, i, 1))
        pserver.start()
        pserver_list.append(pserver)

    def start_client_multithread(ip_config):
        import threading

        dgl.distributed.connect_to_server(
            ip_config=ip_config,
            num_servers=1,
        )
        dgl.distributed.register_service(
            HELLO_SERVICE_ID, HelloRequest, HelloResponse
        )

        req = HelloRequest(STR, INTEGER, TENSOR, simple_func)
        dgl.distributed.send_request(0, req)

        def subthread_call(server_id):
            req = HelloRequest(STR, INTEGER, TENSOR, simple_func)
            dgl.distributed.send_request(server_id, req)

        subthread = threading.Thread(target=subthread_call, args=(1,))
        subthread.start()
        subthread.join()

        res0 = dgl.distributed.recv_response()
        res1 = dgl.distributed.recv_response()
        # Order is not guaranteed
        assert_array_equal(F.asnumpy(res0.tensor), F.asnumpy(TENSOR))
        assert_array_equal(F.asnumpy(res1.tensor), F.asnumpy(TENSOR))
        dgl.distributed.exit_client()

    start_client_multithread(ip_config)
    pserver.join()


@unittest.skipIf(os.name == "nt", reason="Do not support windows yet")
def test_multi_client_connect():
    reset_envs()
    os.environ["DGL_DIST_MODE"] = "distributed"
    ip_config = "rpc_ip_config_mul_client.txt"
    generate_ip_config(ip_config, 1, 1)
    ctx = mp.get_context("spawn")
    num_clients = 1
    pserver = ctx.Process(
        target=start_server,
        args=(num_clients, ip_config, 0, 1),
    )

    # small max try times
    os.environ["DGL_DIST_MAX_TRY_TIMES"] = "1"
    expect_except = False
    try:
        start_client(ip_config, 0, 1)
    except dgl.distributed.DistConnectError as err:
        print("Expected error: {}".format(err))
        expect_except = True
    assert expect_except

    # large max try times
    os.environ["DGL_DIST_MAX_TRY_TIMES"] = "1024"
    pclient = ctx.Process(target=start_client, args=(ip_config, 0, 1))
    pclient.start()
    pserver.start()
    pclient.join()
    pserver.join()
    reset_envs()


if __name__ == "__main__":
    test_serialize()
    test_rpc_msg()
    test_multi_client("socket")
    test_multi_client("tesnsorpipe")
    test_multi_thread_rpc()
    test_multi_client_connect("socket")
