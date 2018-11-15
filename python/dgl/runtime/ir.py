from __future__ import absolute_import

from abc import abstractmethod
from collections import namedtuple
from contextlib import contextmanager

class VarType(object):
    FEAT = 0
    FEAT_DICT = 1
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
    data : any, default=None
    """
    count = 0  # TODO: move counter to prog

    @staticmethod
    def new(type, data=None):
        name = '_z%d' % Var.count
        Var.count += 1
        return Var(name, type, data)

    @staticmethod
    def FEAT(data=None):
        return Var.new(VarType.FEAT, data)

    @staticmethod
    def FEAT_DICT(data=None):
        return Var.new(VarType.FEAT_DICT, data)

    @staticmethod
    def SPMAT(data=None):
        return Var.new(VarType.SPMAT, data)

    @staticmethod
    def IDX(data=None):
        return Var.new(VarType.IDX, data)

    @staticmethod
    def STR(data=None):
        return Var.new(VarType.STR, data)

    @staticmethod
    def FUNC(data=None):
        return Var.new(VarType.FUNC, data)

class OpCode(object):
    CALL = 0
    READ = 1
    READ_COL = 2
    READ_ROW = 3
    WRITE = 4
    WRITE_COL = 5
    WRITE_ROW = 6
    SPMV = 7
    SPMV_WITH_DATA = 8
    MERGE = 9

OP_CODE_NAME_MAP = [
    'CALL',
    'READ',
    'READ_COL',
    'READ_ROW',
    'WRITE',
    'WRITE_COL',
    'WRITE_ROW',
    'SPMV',
    'SPMV_WITH_DATA',
    'MERGE',
]

class InstSpec(namedtuple('Inst', ['opcode', 'args_type', 'ret_type'])):
    pass

class Inst(namedtuple('Inst', ['opcode', 'args', 'ret'])):
    """Instruction.
    opcode : int
    args : a list of str
    ret : str
    """
    # '+' means there might be more than one arg;
    CALL = InstSpec(OpCode.CALL,
            [VarType.FUNC, VarType.FEAT_DICT, '*'],
            VarType.FEAT_DICT)
    READ = InstSpec(OpCode.READ,
            [VarType.FEAT_DICT, VarType.IDX, VarType.STR],
            VarType.FEAT)
    READ_COL = InstSpec(OpCode.READ_COL,
            [VarType.FEAT_DICT, VarType.STR],
            VarType.FEAT)
    READ_ROW = InstSpec(OpCode.READ_ROW,
            [VarType.FEAT_DICT, VarType.IDX],
            VarType.FEAT_DICT)
    WRITE = InstSpec(OpCode.WRITE,
            [VarType.FEAT_DICT, VarType.IDX, VarType.STR, VarType.FEAT],
            None)
    WRITE_COL = InstSpec(OpCode.WRITE_COL,
            [VarType.FEAT_DICT, VarType.STR, VarType.FEAT],
            None)
    WRITE_ROW = InstSpec(OpCode.WRITE_ROW,
            [VarType.FEAT_DICT, VarType.IDX, VarType.FEAT_DICT],
            None)
    SPMV = InstSpec(OpCode.SPMV,
            [VarType.SPMAT, VarType.FEAT],
            VarType.FEAT)
    SPMV_WITH_DATA = InstSpec(OpCode.SPMV_WITH_DATA,
            [VarType.SPMAT, VarType.FEAT, VarType.FEAT],
            VarType.FEAT)
    # merge always sort
    MERGE = InstSpec(OpCode.MERGE,
            [VarType.IDX, '*', VarType.FEAT_DICT, '*'],
            VarType.FEAT_DICT)

class Prog(object):
    """The program."""
    def __init__(self):
        self.insts = []

    def issue(self, inst):
        self.insts.append(inst)

    def pprint(self):
        for inst in self.insts:
            argstr = ', '.join([a.name for a in inst.args])
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

def WRITE(fd, row, col, val):
    get_current_prog().issue(Inst(OpCode.WRITE, [fd, row, col, val], None))

def WRITE_COL(fd, col, val):
    get_current_prog().issue(Inst(OpCode.WRITE_COL, [fd, col, val], None))

def WRITE_ROW(fd, row, val):
    get_current_prog().issue(Inst(OpCode.WRITE_ROW, [fd, row, val], None))

def SPMV(spA, B, ret=None):
    ret = Var.new(Inst.SPMV.ret_type) if ret is None else ret
    get_current_prog().issue(Inst(OpCode.SPMV, [spA, B], ret))
    return ret

def SPMV_WITH_DATA(spA, A_data, B, ret=None):
    ret = Var.new(Inst.SPMV.ret_type) if ret is None else ret
    get_current_prog().issue(Inst(OpCode.SPMV_WITH_DATA, [spA, A_data, B], ret))
    return ret

def MERGE(idx_list, fd_list, ret=None):
    ret = Var.new(Inst.MERGE.ret_type) if ret is None else ret
    get_current_prog().issue(Inst(OpCode.MERGE, idx_list + fd_list, ret))
    return ret

"""
=> v2v case
env={nf, ef, u, e, adj, unique_v}
 Feat t1 = READ(nf, u, 'x')
 Feat t2 = READ(ef, e, 'w')
 Feat t3 = SPMV_WITH_DATA(adj, t1, t2)  # snr1
 WRITE(nf, unique_v, 'y', t3)

=> e2v case
env={nf, u, v, e, inc, unique_v, mfunc}
 Feat t3 = READ(nf, u, 'x')
 Feat t4 = READ(nf, v, 'x')
 Feat t5 = READ(ef, e, 'w')
 FeatDict t6 = CALL(mfunc, t3, t4, t5)
 Feat t8 = READ_COL(t6, 'w')
 Feat t9 = SPMV(inc, t8)
 WRITE(nf, unique_v, 'z', t9)

=> degree bucket case
env={nf, u, v, e, inc, unique_v, mfunc, rfunc
     vb1, vb2, ..., eb1, eb2, ...}
 FeatDict t1 = READ_ROW(nf, u)
 FeatDict t2 = READ_ROW(nf, v)
 FeatDict t3 = READ_ROW(ef, e)
 FeatDict t10 = CALL(mfunc, t1, t2, t3)

 FeatDict fdvb1 = READ_ROW(nf, vb1)
 FeatDict fdeb1 = READ_ROW(t10, eb1)
 FeatDict fdvb1 = CALL(rfunc, fdvb1, fdeb1)  # bkt1

 FeatDict fdvb2 = READ_ROW(nf, vb2)
 FeatDict fdeb2 = READ_ROW(t10, eb2)
 FeatDict fdvb2 = CALL(rfunc, fdvb2, fdeb2)  # bkt2

 FeatDict fdvb3 = READ_ROW(nf, vb3)
 FeatDict fdeb3 = READ_ROW(t10, eb3)
 FeatDict fdvb3 = CALL(rfunc, fdvb3, fdeb3)  # bkt3

 FeatDict t15 = MERGE(vb1, fdvb1,
                      vb2, fdvb2,
                      vb3, fdvb3,
                      unique_v)

 WRITE(nf, unique_v, t15)
"""
