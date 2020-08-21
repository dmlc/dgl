"""Manage the roles in different clients.

Right now, the clients have different roles. Some clients work as samplers and
some work as trainers.
"""

import os
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
        role = server_state.roles
        if self.role not in role:
            role[self.role] = set()
            if kv_store is not None:
                kv_store.barrier_count[self.role] = 0
        role[self.role].add((self.client_id, self.machine_id))
        total_count = 0
        for key in role:
            total_count += len(role[key])
        # Clients are blocked util all clients register their roles.
        if total_count == rpc.get_num_client():
            res_list = []
            for target_id in range(rpc.get_num_client()):
                res_list.append((target_id, RegisterRoleResponse(REG_ROLE_MSG)))
            return res_list
        return None

GET_ROLE = 700002
GET_ROLE_MSG = "Get_Role"

class GetRoleResponse(rpc.Response):
    """Send the roles of all client processes"""
    def __init__(self, role):
        self.role = role
        self.msg = GET_ROLE_MSG

    def __getstate__(self):
        return self.role, self.msg

    def __setstate__(self, state):
        self.role, self.msg = state

class GetRoleRequest(rpc.Request):
    """Send a request to get the roles of all client processes."""
    def __init__(self):
        self.msg = GET_ROLE_MSG

    def __getstate__(self):
        return self.msg

    def __setstate__(self, state):
        self.msg = state

    def process_request(self, server_state):
        return GetRoleResponse(server_state.roles)

# The key is role, the value is a dict of mapping RPC rank to a rank within the role.
PER_ROLE_RANK = {}

# The global rank of a client process. The client processes of the same role have
# global ranks that fall in a contiguous range.
GLOBAL_RANK = {}

# The role of the current process
CUR_ROLE = None

IS_STANDALONE = False

def init_role(role):
    """Initialize the role of the current process.

    Each process is associated with a role so that we can determine what
    function can be invoked in a process. For example, we do not allow some
    functions in sampler processes.

    The initialization includes registeration the role of the current process and
    get the roles of all client processes. It also computes the rank of all client
    processes in a deterministic way so that all clients will have the same rank for
    the same client process.
    """
    global CUR_ROLE
    CUR_ROLE = role

    global PER_ROLE_RANK
    global GLOBAL_RANK
    global IS_STANDALONE

    if os.environ.get('DGL_DIST_MODE', 'standalone') == 'standalone':
        if role == 'default':
            GLOBAL_RANK[0] = 0
            PER_ROLE_RANK['default'] = {0:0}
        IS_STANDALONE = True
        return

    PER_ROLE_RANK = {}
    GLOBAL_RANK = {}

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
    """Get the global rank

    The rank can globally identify the client process. For the client processes
    of the same role, their ranks are in a contiguous range.
    """
    if IS_STANDALONE:
        return 0
    else:
        return GLOBAL_RANK[rpc.get_rank()]

def get_rank(role):
    """Get the role-specific rank"""
    if IS_STANDALONE:
        return 0
    else:
        return PER_ROLE_RANK[role][rpc.get_rank()]

def get_trainer_rank():
    """Get the rank of the current trainer process.

    This function can only be called in the trainer process. It will result in
    an error if it's called in the process of other roles.
    """
    assert CUR_ROLE == 'default'
    if IS_STANDALONE:
        return 0
    else:
        return PER_ROLE_RANK['default'][rpc.get_rank()]

def get_role():
    """Get the role of the current process"""
    return CUR_ROLE

def get_num_trainers():
    """Get the number of trainer processes"""
    return len(PER_ROLE_RANK['default'])

rpc.register_service(REGISTER_ROLE, RegisterRoleRequest, RegisterRoleResponse)
rpc.register_service(GET_ROLE, GetRoleRequest, GetRoleResponse)
