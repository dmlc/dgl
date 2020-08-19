"""Module for variables."""
# pylint: disable=invalid-name
from __future__ import absolute_import

from .program import get_current_prog

class VarType(object):
    """Variable types."""
    # Types for symbolic objects (i.e, they might not be
    #  concretized before evaluation.
    FEAT = 0
    FEAT_DICT = 1
    # Types for concrete objects (i.e, they must have values).
    GRAPH = 2
    IDX = 3
    STR = 4
    FUNC = 5
    MAP = 6
    INT = 7

VAR_TYPE_NAME_MAP = [
    'Feat',
    'FeatDict',
    'GRAPH',
    'Idx',
    'Str',
    'Func',
    'Map',
    'Int',
]

class Var(object):
    """Class for variables in IR.

    Variables represent data in the IR. A variable can contain concrete values.
    Otherwise, it can act as a "symbol", whose values are not materialized at
    the moment, but later.

    Parameters
    ----------
    name : str
        The variable name.
    type : int
        The type code.
    data : any, default=None (not concretized)
        The data.
    """
    __slots__ = ['name', 'typecode', 'data']

    def __init__(self, name, typecode, data):
        self.name = name
        self.typecode = typecode
        self.data = data

    def __str__(self):
        if self.typecode == VarType.STR:
            return '"%s"' % self.data
        else:
            return self.name

    def typestr(self):
        """Return the type string of this variable."""
        return VAR_TYPE_NAME_MAP[self.typecode]

def new(typecode, data=None, name=None):
    """Create a new variable."""
    if name is None:
        cur_prog = get_current_prog()
        name = '_z%d' % cur_prog.varcount
        cur_prog.varcount += 1
    return Var(name, typecode, data)

def FEAT(data=None, name=None):
    """Create a variable for feature tensor."""
    return new(VarType.FEAT, data, name)

def FEAT_DICT(data=None, name=None):
    """Create a variable for feature dict."""
    return new(VarType.FEAT_DICT, data, name)

def GRAPH(data=None, name=None):
    """Create a variable for graph index lambda."""
    return new(VarType.GRAPH, data, name)

def IDX(data=None, name=None):
    """Create a variable for index."""
    return new(VarType.IDX, data, name)

def STR(data=None, name=None):
    """Create a variable for string value."""
    return new(VarType.STR, data, name)

def FUNC(data=None, name=None):
    """Create a variable for function."""
    return new(VarType.FUNC, data, name)

def MAP(data=None, name=None):
    """Create a variable for mapping lambda"""
    return new(VarType.MAP, data, name)

def INT(data=None, name=None):
    """Create a variable for int value"""
    return new(VarType.INT, data, name)
