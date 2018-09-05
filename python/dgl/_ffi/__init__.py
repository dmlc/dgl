"""C interfacing code.

This namespace contains everything that interacts with C code.
Most C related object are ctypes compatible, which means
they contains a handle field that is ctypes.c_void_p and can
be used via ctypes function calls.

Some performance critical functions are implemented by cython
and have a ctypes fallback implementation.
"""
