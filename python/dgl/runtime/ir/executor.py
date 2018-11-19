from __future__ import absolute_import

from abc import abstractmethod

from ... import backend as F

from .program import get_current_prog
from . import var
from .var import VarType
from .registry import IR_REGISTRY

class OpCode(object):
    # immutable op
    CALL = 0
    SPMV = 1
    SPMV_WITH_DATA = 2
    READ = 3
    READ_COL = 4
    READ_ROW = 5
    MERGE_ROW = 6
    UPDATE_DICT = 7
    # mutable op (no return)
    # remember the name is suffixed with "_"
    WRITE_ = 21
    WRITE_COL_ = 22
    WRITE_ROW_ = 23
    WRITE_DICT_ = 24
    APPEND_ROW_ = 25

class Executor(object):
    @abstractmethod
    def opcode(self):
        raise NotImplementedError

    @abstractmethod
    def arg_vars(self):
        raise NotImplementedError

    @abstractmethod
    def ret_var(self):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError

class CallExecutor(Executor):
    def __init__(self, fn, args, ret):
        self.fn = fn
        self.args = args
        self.ret = ret

    def opcode(self):
        return OpCode.CALL

    def arg_vars(self):
        return [self.fn] + self.args

    def ret_var(self):
        return self.ret

    def run(self):
        fn_data = self.fn.data
        args_data = [a.data for a in self.args]
        self.ret.data = fn_data(*args_data)

IR_REGISTRY[OpCode.CALL] = {
    'name' : 'CALL',
    'args_type' : [VarType.FUNC, VarType.FEAT_DICT, '*'],
    'ret_type' : VarType.FEAT_DICT,
    'executor_cls' : CallExecutor,
}
def CALL(func, args, ret=None):
    reg = IR_REGISTRY[OpCode.CALL]
    ret = var.new(reg['ret_type']) if ret is None else ret
    get_current_prog().issue(reg['executor_cls'](func, args, ret))
    return ret

class ReadExecutor(Executor):
    def __init__(self, fd, row, col, ret):
        self.fd = fd
        self.row = row
        self.col = col
        self.ret = ret

    def opcode(self):
        return OpCode.READ

    def arg_vars(self):
        return [self.fd, self.row, self.col]

    def ret_var(self):
        return self.ret

    def run(self):
        fd_data = self.fd.data  # feature dict
        row_data = self.row.data  # idx
        col_data = self.col.data  # key str
        self.ret.data = fd_data[col_data][row_data]

IR_REGISTRY[OpCode.READ] = {
    'name' : 'READ',
    'args_type' : [VarType.FEAT_DICT, VarType.IDX, VarType.STR],
    'ret_type' : VarType.FEAT,
    'executor_cls' : ReadExecutor,
}
def READ(fd, row, col, ret=None):
    reg = IR_REGISTRY[OpCode.READ]
    ret = var.new(reg['ret_type']) if ret is None else ret
    get_current_prog().issue(reg['executor_cls'](fd, row, col, ret))
    return ret

class ReadColExecutor(Executor):
    def __init__(self, fd, col, ret):
        self.fd = fd
        self.col = col
        self.ret = ret

    def opcode(self):
        return OpCode.READ_COL

    def arg_vars(self):
        return [self.fd, self.col]

    def ret_var(self):
        return self.ret

    def run(self):
        fd_data = self.fd.data
        col_data = self.col.data
        self.ret.data = fd_data[col_data]

IR_REGISTRY[OpCode.READ_COL] = {
    'name' : 'READ_COL',
    'args_type' : [VarType.FEAT_DICT, VarType.STR],
    'ret_type' : VarType.FEAT,
    'executor_cls' : ReadColExecutor,
}
def READ_COL(fd, col, ret=None):
    reg = IR_REGISTRY[OpCode.READ_COL]
    ret = var.new(reg['ret_type']) if ret is None else ret
    get_current_prog().issue(reg['executor_cls'](fd, col, ret))
    return ret

class ReadRowExecutor(Executor):
    def __init__(self, fd, row, ret):
        self.fd = fd
        self.row = row
        self.ret = ret

    def opcode(self):
        return OpCode.READ_ROW

    def arg_vars(self):
        return [self.fd, self.row]

    def ret_var(self):
        return self.ret

    def run(self):
        fd_data = self.fd.data
        row_data = self.row.data  # idx
        self.ret.data = fd_data[row_data]

IR_REGISTRY[OpCode.READ_ROW] = {
    'name' : 'READ_ROW',
    'args_type' : [VarType.FEAT_DICT, VarType.IDX],
    'ret_type' : VarType.FEAT_DICT,
    'executor_cls' : ReadRowExecutor,
}
def READ_ROW(fd, row, ret=None):
    reg = IR_REGISTRY[OpCode.READ_ROW]
    ret = var.new(reg['ret_type']) if ret is None else ret
    get_current_prog().issue(reg['executor_cls'](fd, row, ret))
    return ret

class SPMVExecutor(Executor):
    def __init__(self, spA, B, ret):
        self.spA = spA
        self.B = B
        self.ret = ret

    def opcode(self):
        return OpCode.SPMV

    def arg_vars(self):
        return [self.spA, self.B]

    def ret_var(self):
        return self.ret

    def run(self):
        assert False
        pass

IR_REGISTRY[OpCode.SPMV] = {
    'name' : 'SPMV',
    'args_type' : [VarType.SPMAT, VarType.FEAT],
    'ret_type' : VarType.FEAT,
    'executor_cls' : SPMVExecutor,
}
def SPMV(spA, B, ret=None):
    reg = IR_REGISTRY[OpCode.SPMV]
    ret = var.new(reg['ret_type']) if ret is None else ret
    get_current_prog().issue(reg['executor_cls'](spA, B, ret))
    return ret

class SPMVWithDataExecutor(Executor):
    def __init__(self, spA, A_data, B, ret):
        self.spA = spA
        self.A_data = A_data
        self.B = B
        self.ret = ret

    def opcode(self):
        return OpCode.SPMV_WITH_DATA

    def arg_vars(self):
        return [self.spA, self.A_data, self.B]

    def ret_var(self):
        return self.ret

    def run(self):
        assert False
        pass

IR_REGISTRY[OpCode.SPMV_WITH_DATA] = {
    'name' : 'SPMV_WITH_DATA',
    'args_type' : [VarType.SPMAT, VarType.FEAT, VarType.FEAT],
    'ret_type' : VarType.FEAT,
    'executor_cls' : SPMVWithDataExecutor,
}
def SPMV_WITH_DATA(spA, A_data, B, ret=None):
    reg = IR_REGISTRY[OpCode.SPMV_WITH_DATA]
    ret = var.new(reg['ret_type']) if ret is None else ret
    get_current_prog().issue(reg['executor_cls'](spA, A_data, B, ret))
    return ret

class MergeRowExecutor(Executor):
    def __init__(self, order, idx_list, fd_list, ret):
        self.order = order
        self.idx_list = idx_list
        self.fd_list = fd_list
        self.ret = ret

    def opcode(self):
        return OpCode.MERGE_ROW

    def arg_vars(self):
        return [self.order] + self.idx_list + self.fd_list

    def ret_var(self):
        return self.ret

    def run(self):
        assert False
        pass

IR_REGISTRY[OpCode.MERGE_ROW] = {
    'name' : 'MERGE_ROW',
    'args_type' : [VarType.IDX, VarType.IDX, '*', VarType.FEAT_DICT, '*'],
    'ret_type' : VarType.FEAT_DICT,
    'executor_cls' : MergeRowExecutor,
}
def MERGE_ROW(order, idx_list, fd_list, ret=None):
    reg = IR_REGISTRY[OpCode.MERGE_ROW]
    ret = var.new(reg['ret_type']) if ret is None else ret
    get_current_prog().issue(reg['executor_cls'](order, idx_list, fd_list, ret))
    return ret

class UpdateDictExecutor(Executor):
    def __init__(self, fd1, fd2, ret):
        self.fd1 = fd1
        self.fd2 = fd2
        self.ret = ret

    def opcode(self):
        return OpCode.UPDATE_DICT

    def arg_vars(self):
        return [self.fd1, self.fd2]

    def ret_var(self):
        return self.ret

    def run(self):
        assert False
        pass

IR_REGISTRY[OpCode.UPDATE_DICT] = {
    'name' : 'UPDATE_DICT',
    'args_type' : [VarType.FEAT_DICT, VarType.FEAT_DICT],
    'ret_type' : VarType.FEAT_DICT,
    'executor_cls' : UpdateDictExecutor,
}
def UPDATE_DICT(fd1, fd2, ret=None):
    reg = IR_REGISTRY[OpCode.UPDATE_DICT]
    ret = var.new(reg['ret_type']) if ret is None else ret
    get_current_prog().issue(reg['executor_cls'](fd1, fd2, ret))
    return ret

class Write_Executor(Executor):
    def __init__(self, fd, row, col, val):
        self.fd = fd
        self.row = row
        self.col = col
        self.val = val

    def opcode(self):
        return OpCode.WRITE_

    def arg_vars(self):
        return [self.fd, self.row, self.col, self.val]

    def ret_var(self):
        return None

    def run(self):
        fd_data = self.fd.data  # feature dict
        row_data = self.row.data  # idx
        col_data = self.col.data  # key str
        val_data = self.val.data
        fd_data[col_data][row_data] = val_data

IR_REGISTRY[OpCode.WRITE_] = {
    'name' : 'WRITE_',
    'args_type' : [VarType.FEAT_DICT, VarType.IDX, VarType.STR, VarType.FEAT],
    'ret_type' : None,
    'executor_cls' : Write_Executor,
}
def WRITE_(fd, row, col, val):
    reg = IR_REGISTRY[OpCode.WRITE_]
    get_current_prog().issue(reg['executor_cls'](fd, row, col, val))

class WriteCol_Executor(Executor):
    def __init__(self, fd, col, val):
        self.fd = fd
        self.col = col
        self.val = val

    def opcode(self):
        return OpCode.WRITE_COL_

    def arg_vars(self):
        return [self.fd, self.col, self.val]

    def ret_var(self):
        return None

    def run(self):
        fd_data = self.fd.data  # feature dict
        col_data = self.col.data  # key str
        val_data = self.val.data
        fd_data[col_data] = val_data

IR_REGISTRY[OpCode.WRITE_COL_] = {
    'name' : 'WRITE_COL_',
    'args_type' : [VarType.FEAT_DICT, VarType.STR, VarType.FEAT],
    'ret_type' : None,
    'executor_cls' : WriteCol_Executor,
}
def WRITE_COL_(fd, col, val):
    reg = IR_REGISTRY[OpCode.WRITE_COL_]
    get_current_prog().issue(reg['executor_cls'](fd, col, val))

class WriteRow_Executor(Executor):
    def __init__(self, fd, row, val):
        self.fd = fd
        self.row = row
        self.val = val

    def opcode(self):
        return OpCode.WRITE_ROW_

    def arg_vars(self):
        return [self.fd, self.row, self.val]

    def ret_var(self):
        return None

    def run(self):
        fd_data = self.fd.data  # feature dict
        row_data = self.row.data  # idx
        val_data = self.val.data
        fd_data[row_data] = val_data

IR_REGISTRY[OpCode.WRITE_ROW_] = {
    'name' : 'WRITE_ROW_',
    'args_type' : [VarType.FEAT_DICT, VarType.IDX, VarType.FEAT_DICT],
    'ret_type' : None,
    'executor_cls' : WriteRow_Executor,
}
def WRITE_ROW_(fd, row, val):
    reg = IR_REGISTRY[OpCode.WRITE_ROW_]
    get_current_prog().issue(reg['executor_cls'](fd, row, val))

class WriteDict_Executor(Executor):
    def __init__(self, fd1, fd2):
        self.fd1 = fd1
        self.fd2 = fd2

    def opcode(self):
        return OpCode.WRITE_DICT_

    def arg_vars(self):
        return [self.fd1, self.fd2]

    def ret_var(self):
        return None

    def run(self):
        fd1_data = self.fd1.data
        fd2_data = self.fd2.data
        for k, v in fd2_data.items():
            fd1_data[k] = v

IR_REGISTRY[OpCode.WRITE_DICT_] = {
    'name' : 'WRITE_DICT_',
    'args_type' : [VarType.FEAT_DICT, VarType.FEAT_DICT],
    'ret_type' : None,
    'executor_cls' : WriteDict_Executor,
}
def WRITE_DICT_(fd1, fd2):
    reg = IR_REGISTRY[OpCode.WRITE_DICT_]
    get_current_prog().issue(reg['executor_cls'](fd1, fd2))

class AppendRow_Executor(Executor):
    def __init__(self, fd1, fd2):
        self.fd1 = fd1
        self.fd2 = fd2

    def opcode(self):
        return OpCode.APPEND_ROW_

    def arg_vars(self):
        return [self.fd1, self.fd2]

    def ret_var(self):
        return None

    def run(self):
        fd1_data = self.fd1.data
        fd2_data = self.fd2.data
        fd1_data.append(fd2_data)

IR_REGISTRY[OpCode.APPEND_ROW_] = {
    'name' : 'APPEND_ROW_',
    'args_type' : [VarType.FEAT_DICT, VarType.FEAT_DICT],
    'ret_type' : None,
    'executor_cls' : AppendRow_Executor,
}
def APPEND_ROW_(fd1, fd2):
    reg = IR_REGISTRY[OpCode.APPEND_ROW_]
    get_current_prog().issue(reg['executor_cls'](fd1, fd2))
