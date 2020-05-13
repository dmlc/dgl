import dgl
import backend as F
import unittest, pytest

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

