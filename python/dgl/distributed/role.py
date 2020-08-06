"""Manage the roles in different clients.

Right now, the clients have different roles. Some clients work as samplers and
some work as trainers.
"""

import numpy as np

from . import rpc

REGISTER_ROLE = 700001
REG_ROLE_MSG = "Register_Role"

class RegisterRoleResponse(rpc.Response):
    """Send a confirmation signal (just a short string message)
    of RegisterRoleRequest to client.
    """
    def __init__(self, msg):
        self.msg = msg

    def __getstate__(self):
        return self.msg

    def __setstate__(self, state):
        self.msg = state

class RegisterRoleRequest(rpc.Request):
    """Send client id and role to server

    Parameters
    ----------
    client_id : int
        ID of client
    role : str
        role of client
    """
    def __init__(self, client_id, machine_id, role):
        self.client_id = client_id
        self.machine_id = machine_id
        self.role = role

    def __getstate__(self):
        return self.client_id, self.machine_id, self.role

    def __setstate__(self, state):
        self.client_id, self.machine_id, self.role = state

    def process_request(self, server_state):
        kv_store = server_state.kv_store
        role = kv_store.role
        if self.role not in role:
            role[self.role] = set()
            kv_store.barrier_count[self.role] = 0
        role[self.role].add((self.client_id, self.machine_id))
        total_count = 0
        for key in role:
            total_count += len(role[key])
        # Clients are blocked util all clients register their roles.
        if total_count == kv_store.num_clients:
            res_list = []
            for target_id in range(kv_store.num_clients):
                print('send', target_id)
                res_list.append((target_id, RegisterRoleResponse(REG_ROLE_MSG)))
            return res_list
        return None

GET_ROLE = 700002
GET_ROLE_MSG = "Get_Role"

class GetRoleResponse(rpc.Response):
    def __init__(self, role):
        self.role = role
        self.msg = GET_ROLE_MSG

    def __getstate__(self):
        return self.role, self.msg

    def __setstate__(self, state):
        self.role, self.msg = state

class GetRoleRequest(rpc.Request):
    def __init__(self):
        self.msg = GET_ROLE_MSG

    def __getstate__(self):
        return self.msg

    def __setstate__(self, state):
        self.msg = state

    def process_request(self, server_state):
        kv_store = server_state.kv_store
        return GetRoleResponse(kv_store.role)

# The key is role, the value is a dict of mapping RPC rank to a rank within the role.
PER_ROLE_RANK = {}

GLOBAL_RANK = {}

def init_role(role):
    # Register the current role. This blocks until all clients register themselves.
    client_id = rpc.get_rank()
    machine_id = rpc.get_machine_id()
    request = RegisterRoleRequest(client_id, machine_id, role)
    rpc.send_request(0, request)
    response = rpc.recv_response()
    assert response.msg == REG_ROLE_MSG

    # Get all clients on all machines.
    request = GetRoleRequest()
    rpc.send_request(0, request)
    response = rpc.recv_response()
    assert response.msg == GET_ROLE_MSG

    # Here we want to compute a new rank for each client.
    # We compute the per-role rank as well as global rank.
    # For per-role rank, we ensure that all ranks within a machine is contiguous.
    # For global rank, we also ensure that all ranks within a machine are contiguous,
    # and all ranks within a role are contiguous.
    global PER_ROLE_RANK
    global GLOBAL_RANK
    global_rank = 0

    # We want to ensure that the global rank of the trainer process starts from 0.
    role_names = ['default']
    for role_name in response.role:
        if role_name not in role_names:
            role_names.append(role_name)

    for role_name in role_names:
        # Let's collect the ranks of this role in all machines.
        machines = {}
        for client_id, machine_id in response.role[role_name]:
            if machine_id not in machines:
                machines[machine_id] = []
            machines[machine_id].append(client_id)

        num_machines = len(machines)
        PER_ROLE_RANK[role_name] = {}
        per_role_rank = 0
        for i in range(num_machines):
            clients = machines[i]
            clients = np.sort(clients)
            for client_id in clients:
                GLOBAL_RANK[client_id] = global_rank
                global_rank += 1
                PER_ROLE_RANK[role_name][client_id] = per_role_rank
                per_role_rank += 1

def get_global_rank():
    return GLOBAL_RANK[rpc.get_rank()]

def get_rank(role):
    return PER_ROLE_RANK[role][rpc.get_rank()]

def get_trainer_rank():
    return PER_ROLE_RANK['default'][rpc.get_rank()]

def get_num_trainers():
    return len(PER_ROLE_RANK['default'])

rpc.register_service(REGISTER_ROLE, RegisterRoleRequest, RegisterRoleResponse)
rpc.register_service(GET_ROLE, GetRoleRequest, GetRoleResponse)
