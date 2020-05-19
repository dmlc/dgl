import dgl
import backend as F
import unittest, pytest

from numpy.testing import assert_array_equal

INTEGER = 2
STR = 'hello world!'
HELLO_SERVICE_ID = 901231

def test_func(tensor, integer):
    return tensor * integer

def test_rank():
    dgl.distributed.set_rank(2)
    assert dgl.distributed.get_rank() == 2

def test_msg_seq():
    from dgl.distributed.rpc import get_msg_seq, incr_msg_seq
    assert get_msg_seq() == 0
    incr_msg_seq()
    incr_msg_seq()
    incr_msg_seq()
    assert get_msg_seq() == 3

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
        new_tensor = self.func(self.tensor, self.integer)
        res = HelloResponse(self.hello_str, self.integer, new_tensor)

def start_server():
    dgl.distributed.register_service(HELLO_SERVICE_ID, HelloRequest, HelloResponse)
    dgl.distributed.start_server(server_id=0, ip_config='ip_config.txt', num_clients=1)

def start_client():
    dgl.distributed.register_service(HELLO_SERVICE_ID, HelloRequest, HelloResponse)
    dgl.distributed.connect_to_server(ip_config='ip_config.txt')
    req = HelloRequest(STR, INTEGER, F.tensor([1,2,3]), test_func)
    # test send and recv
    dgl.distributed.send_request(0, req)
    res = dgl.distributed.recv_response()
    assert res.hello_str == STR
    assert res.integer == INTEGER
    assert_array_equal(res.tensor, F.tensor([2,4,6]))
    # test remote_call
    target_and_requests = []
    for i in range(10):
        target_and_requests.append((0, req))
    res_list = dgl.distributed.remote_call(target_and_requests)
    for res in res_list:
        assert res.hello_str == STR
        assert res.integer == INTEGER
        assert_array_equal(res.tensor, F.tensor([2,4,6]))
    # clean up
    dgl.distributed.shutdown_servers()
    dgl.distributed.finalize_client()

def test_serialize():
    from dgl.distributed.rpc import serialize_to_payload, deserialize_from_payload
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
    from dgl.distributed.rpc import serialize_to_payload, deserialize_from_payload, RPCMessage
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

def test_rpc():
    pid = os.fork()
    if pid == 0:
        start_server()
    else:
        time.sleep(1)
        start_client()
