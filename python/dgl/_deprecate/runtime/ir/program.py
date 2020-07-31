"""Module for program."""
from __future__ import absolute_import

from contextlib import contextmanager
import threading

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

class CurrentProgram(threading.local):
    """Thread local storage to keep the reference of current thread's program"""
    def __init__(self):
        super(CurrentProgram, self).__init__()
        self.prog = None

    def get_prog(self):
        """Get program"""
        return self.prog

    def set_prog(self, program):
        """Set program"""
        self.prog = program

# current program
CURRENT_PROG = CurrentProgram()

def get_current_prog():
    """Get the current program."""
    return CURRENT_PROG.get_prog()

def set_current_prog(program):
    """Set the current program."""
    CURRENT_PROG.set_prog(program)

@contextmanager
def prog():
    """A context manager to create a new program."""
    set_current_prog(Prog())
    yield get_current_prog()
    set_current_prog(None)
