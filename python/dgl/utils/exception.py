"""Exception wrapper classes to properly display exceptions under multithreading or
multiprocessing.
"""
import sys
import traceback

# The following code is borrowed from PyTorch.  Basically when a subprocess or thread
# throws an exception, you will need to wrap the exception with ExceptionWrapper class
# and put it in the queue you are normally retrieving from.

# NOTE [ Python Traceback Reference Cycle Problem ]
#
# When using sys.exc_info(), it is important to **not** store the exc_info[2],
# which is the traceback, because otherwise you will run into the traceback
# reference cycle problem, i.e., the traceback holding reference to the frame,
# and the frame (which holds reference to all the object in its temporary scope)
# holding reference the traceback.


class KeyErrorMessage(str):
    r"""str subclass that returns itself in repr"""

    def __repr__(self):  # pylint: disable=invalid-repr-returned
        return self


class ExceptionWrapper(object):
    r"""Wraps an exception plus traceback to communicate across threads"""

    def __init__(self, exc_info=None, where="in background"):
        # It is important that we don't store exc_info, see
        # NOTE [ Python Traceback Reference Cycle Problem ]
        if exc_info is None:
            exc_info = sys.exc_info()
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))
        self.where = where

    def reraise(self):
        r"""Reraises the wrapped exception in the current thread"""
        # Format a message such as: "Caught ValueError in DataLoader worker
        # process 2. Original Traceback:", followed by the traceback.
        msg = "Caught {} {}.\nOriginal {}".format(
            self.exc_type.__name__, self.where, self.exc_msg
        )
        if self.exc_type == KeyError:
            # KeyError calls repr() on its argument (usually a dict key). This
            # makes stack traces unreadable. It will not be changed in Python
            # (https://bugs.python.org/issue2651), so we work around it.
            msg = KeyErrorMessage(msg)
        elif getattr(self.exc_type, "message", None):
            # Some exceptions have first argument as non-str but explicitly
            # have message field
            raise self.exc_type(message=msg)
        try:
            exception = self.exc_type(msg)
        except TypeError:
            # If the exception takes multiple arguments, don't try to
            # instantiate since we don't know how to
            raise RuntimeError(msg) from None
        raise exception
