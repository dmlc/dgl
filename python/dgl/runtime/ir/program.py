"""Module for program."""
from __future__ import absolute_import

from contextlib import contextmanager

from .registry import IR_REGISTRY

class Prog(object):
    """The program.

    A program is simply a list of executors.
    """
    def __init__(self):
        self.execs = []
        self.varcount = 0

    def issue(self, exe):
        """Issue an executor to this program.

        Parameters
        ----------
        exe : Executor
            The executor.
        """
        self.execs.append(exe)

    def pprint_exe(self, exe):
        """Internal function to pretty-print the executor."""
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
        """Pretty-print the program."""
        for exe in self.execs:
            self.pprint_exe(exe)

# current program
CURRENT_PROG = None

def get_current_prog():
    """Get the current program."""
    global CURRENT_PROG
    return CURRENT_PROG

def set_current_prog(program):
    """Set the current program."""
    global CURRENT_PROG
    CURRENT_PROG = program

@contextmanager
def prog():
    """A context manager to create a new program."""
    set_current_prog(Prog())
    yield get_current_prog()
    set_current_prog(None)
