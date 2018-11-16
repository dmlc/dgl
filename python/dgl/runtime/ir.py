from __future__ import absolute_import

from abc import abstractmethod
from collections import namedtuple
from contextlib import contextmanager

class VarType(object):
    # Types for symbolic objects (i.e, they might not be
    #  concretized before evaluation.
    FEAT = 0
    FEAT_DICT = 1
    # Types for concrete objects (i.e, they must have values).
    SPMAT = 2
    IDX = 3
    STR = 4
    FUNC = 5

VAR_TYPE_NAME_MAP = [
    'Feat',
    'FeatDict',
    'SpMat',
    'Idx',
    'Str',
    'Func',
]

class Var(namedtuple('Var', ['name', 'type', 'data'])):
    """Variable
    name : str
    type : int
    data : any, default=None (not concretized)
    """
    @staticmethod
    def new(type, data=None, name=None):
        if name is None:
            cur_prog = get_current_prog()
            name = '_z%d' % cur_prog.varcount
            cur_prog.varcount += 1
        return Var(name, type, data)

    @staticmethod
    def FEAT(data=None, name=None):
        return Var.new(VarType.FEAT, data, name)

    @staticmethod
    def FEAT_DICT(data=None, name=None):
        return Var.new(VarType.FEAT_DICT, data, name)

    @staticmethod
    def SPMAT(data=None, name=None):
        return Var.new(VarType.SPMAT, data, name)

    @staticmethod
    def IDX(data=None, name=None):
        return Var.new(VarType.IDX, data, name)

    @staticmethod
    def STR(data=None, name=None):
        return Var.new(VarType.STR, data, name)

    @staticmethod
    def FUNC(data=None, name=None):
        return Var.new(VarType.FUNC, data, name)

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
    APPEND_ROW = 8
    # mutable op (no return)
    # remember the name is suffixed with "_"
    WRITE_ = 21
    WRITE_COL_ = 22
    WRITE_ROW_ = 23
    WRITE_DICT_ = 24

OP_CODE_NAME_MAP = {
    OpCode.CALL : 'CALL',
    OpCode.SPMV : 'SPMV',
    OpCode.SPMV_WITH_DATA : 'SPMV_WITH_DATA',
    OpCode.READ : 'READ',
    OpCode.READ_COL : 'READ_COL',
    OpCode.READ_ROW : 'READ_ROW',
    OpCode.MERGE_ROW : 'MERGE_ROW',
    OpCode.UPDATE_DICT : 'UPDATE_DICT',
    OpCode.APPEND_ROW : 'APPEND_ROW',
    OpCode.WRITE_ : 'WRITE_',
    OpCode.WRITE_COL_ : 'WRITE_COL_',
    OpCode.WRITE_ROW_ : 'WRITE_ROW_',
    OpCode.WRITE_DICT_ : 'WRITE_DICT_',
}

class InstSpec(namedtuple('Inst', ['opcode', 'args_type', 'ret_type'])):
    pass

class Inst(namedtuple('Inst', ['opcode', 'args', 'ret'])):
    """Instruction.
    opcode : int
    args : a list of str
    ret : str
    """
    # immutable ops
    # '*' means there might be more than one arg;
    # should be only two type of calls: NodeUDF and EdgeUDF
    CALL = InstSpec(OpCode.CALL,
            [VarType.FUNC, VarType.FEAT_DICT, '*'],
            VarType.FEAT_DICT)
    SPMV = InstSpec(OpCode.SPMV,
            [VarType.SPMAT, VarType.FEAT],
            VarType.FEAT)
    SPMV_WITH_DATA = InstSpec(OpCode.SPMV_WITH_DATA,
            [VarType.SPMAT, VarType.FEAT, VarType.FEAT],
            VarType.FEAT)
    READ = InstSpec(OpCode.READ,
            [VarType.FEAT_DICT, VarType.IDX, VarType.STR],
            VarType.FEAT)
    READ_COL = InstSpec(OpCode.READ_COL,
            [VarType.FEAT_DICT, VarType.STR],
            VarType.FEAT)
    READ_ROW = InstSpec(OpCode.READ_ROW,
            [VarType.FEAT_DICT, VarType.IDX],
            VarType.FEAT_DICT)
    # MERGE_ROW(sorted_row, [idx1, idx2, ...], [fd1, fd2, ...])
    MERGE_ROW = InstSpec(OpCode.MERGE_ROW,
            [VarType.IDX, VarType.IDX, '*', VarType.FEAT_DICT, '*'],
            VarType.FEAT_DICT)
    APPEND_ROW = InstSpec(OpCode.APPEND_ROW,
            [VarType.FEAT_DICT, VarType.FEAT_DICT],
            VarType.FEAT_DICT)
    UPDATE_DICT = InstSpec(OpCode.UPDATE_DICT,
            [VarType.FEAT_DICT, VarType.FEAT_DICT],
            VarType.FEAT_DICT)
    # mutable ops
    WRITE_ = InstSpec(OpCode.WRITE_,
            [VarType.FEAT_DICT, VarType.IDX, VarType.STR, VarType.FEAT],
            None)
    WRITE_COL_ = InstSpec(OpCode.WRITE_COL_,
            [VarType.FEAT_DICT, VarType.STR, VarType.FEAT],
            None)
    WRITE_ROW_ = InstSpec(OpCode.WRITE_ROW_,
            [VarType.FEAT_DICT, VarType.IDX, VarType.FEAT_DICT],
            None)

class Prog(object):
    """The program."""
    def __init__(self):
        self.insts = []
        self.varcount = 0

    def issue(self, inst):
        self.insts.append(inst)

    @staticmethod
    def _argtostr(a):
        if a.type == VarType.STR:
            return '"%s"' % a.data
        else:
            return a.name

    def pprint(self):
        for inst in self.insts:
            argstr = ', '.join([Prog._argtostr(a) for a in inst.args])
            if inst.ret is None:
                # stmt
                print("%s(%s)" % (
                    OP_CODE_NAME_MAP[inst.opcode],
                    argstr))
            else:
                print("%s %s = %s(%s)" % (
                    VAR_TYPE_NAME_MAP[inst.ret.type],
                    inst.ret.name,
                    OP_CODE_NAME_MAP[inst.opcode],
                    argstr))

_current_prog = None

def get_current_prog():
    global _current_prog
    return _current_prog

def set_current_prog(prog):
    global _current_prog
    _current_prog = prog

@contextmanager
def prog():
    set_current_prog(Prog())
    yield get_current_prog()
    set_current_prog(None)

def CALL(func, args, ret=None):
    ret = Var.new(Inst.CALL.ret_type) if ret is None else ret
    get_current_prog().issue(Inst(OpCode.CALL, [func] + args, ret))
    return ret

def READ(fd, row, col, ret=None):
    ret = Var.new(Inst.READ.ret_type) if ret is None else ret
    get_current_prog().issue(Inst(OpCode.READ, [fd, row, col], ret))
    return ret

def READ_COL(fd, col, ret=None):
    ret = Var.new(Inst.READ_COL.ret_type) if ret is None else ret
    get_current_prog().issue(Inst(OpCode.READ_COL, [fd, col], ret))
    return ret

def READ_ROW(fd, row, ret=None):
    ret = Var.new(Inst.READ_ROW.ret_type) if ret is None else ret
    get_current_prog().issue(Inst(OpCode.READ_ROW, [fd, row], ret))
    return ret

def SPMV(spA, B, ret=None):
    ret = Var.new(Inst.SPMV.ret_type) if ret is None else ret
    get_current_prog().issue(Inst(OpCode.SPMV, [spA, B], ret))
    return ret

def SPMV_WITH_DATA(spA, A_data, B, ret=None):
    ret = Var.new(Inst.SPMV.ret_type) if ret is None else ret
    get_current_prog().issue(Inst(OpCode.SPMV_WITH_DATA, [spA, A_data, B], ret))
    return ret

def MERGE_ROW(order, idx_list, fd_list, ret=None):
    ret = Var.new(Inst.MERGE_ROW.ret_type) if ret is None else ret
    get_current_prog().issue(Inst(OpCode.MERGE_ROW, [order] + idx_list + fd_list, ret))
    return ret

def APPEND_ROW(fd1, fd2, ret=None):
    ret = Var.new(Inst.APPEND_ROW.ret_type) if ret is None else ret
    get_current_prog().issue(Inst(OpCode.APPEND_ROW, [fd1, fd2], ret))
    return ret

def UPDATE_DICT(fd1, fd2, ret=None):
    ret = Var.new(Inst.UPDATE_DICT.ret_type) if ret is None else ret
    get_current_prog().issue(Inst(OpCode.UPDATE_DICT, [fd1, fd2], ret))
    return ret

def WRITE_(fd, row, col, val):
    get_current_prog().issue(Inst(OpCode.WRITE_, [fd, row, col, val], None))

def WRITE_COL_(fd, col, val):
    get_current_prog().issue(Inst(OpCode.WRITE_COL_, [fd, col, val], None))

def WRITE_ROW_(fd, row, val):
    get_current_prog().issue(Inst(OpCode.WRITE_ROW_, [fd, row, val], None))

def WRITE_DICT_(fd1, fd2):
    get_current_prog().issue(Inst(OpCode.WRITE_DICT_, [fd1, fd2], None))
