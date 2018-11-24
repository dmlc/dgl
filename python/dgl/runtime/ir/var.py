from __future__ import absolute_import

from .program import get_current_prog

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

class Var(object):
    """Variable
    name : str
    type : int
    data : any, default=None (not concretized)
    """
    __slots__ = ['name', 'type', 'data']
    def __init__(self, name, type, data):
        self.name = name
        self.type = type
        self.data = data

    def __str__(self):
        if self.type == VarType.STR:
            return '"%s"' % self.data
        else:
            return self.name

    def typestr(self):
        return VAR_TYPE_NAME_MAP[self.type]

def new(type, data=None, name=None):
    if name is None:
        cur_prog = get_current_prog()
        name = '_z%d' % cur_prog.varcount
        cur_prog.varcount += 1
    return Var(name, type, data)

def FEAT(data=None, name=None):
    return new(VarType.FEAT, data, name)

def FEAT_DICT(data=None, name=None):
    return new(VarType.FEAT_DICT, data, name)

def SPMAT(data=None, name=None):
    return new(VarType.SPMAT, data, name)

def IDX(data=None, name=None):
    return new(VarType.IDX, data, name)

def STR(data=None, name=None):
    return new(VarType.STR, data, name)

def FUNC(data=None, name=None):
    return new(VarType.FUNC, data, name)
