from __future__ import absolute_import

from contextlib import contextmanager

from .registry import IR_REGISTRY

class Prog(object):
    """The program."""
    def __init__(self):
        self.execs = []
        self.varcount = 0

    def issue(self, exe):
        self.execs.append(exe)

    def pprint_exe(self, exe):
        argstr = ', '.join([str(av) for av in exe.arg_vars()])
        if exe.ret_var() is None:
            # stmt
            print("%s(%s)" % (
                IR_REGISTRY[exe.opcode()]['name'],
                argstr))
        else:
            print("%s %s = %s(%s)" % (
                exe.ret_var().typestr(),
                exe.ret.name,
                IR_REGISTRY[exe.opcode()]['name'],
                argstr))

    def pprint(self):
        for exe in self.execs:
            self.pprint_exe(exe)

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
