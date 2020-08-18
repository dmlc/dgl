"""Library information."""
from __future__ import absolute_import
import sys
import os
import pathlib


def find_lib_path(name=None, search_path=None, optional=False):
    """Find dynamic library files.

    Parameters
    ----------
    name : list of str
        List of names to be found.

    Returns
    -------
    lib_path : list(string)
        List of all found path to the libraries
    """
    # See https://github.com/dmlc/tvm/issues/281 for some background.

    # NB: This will either be the source directory (if DGL is run
    # inplace) or the install directory (if DGL is installed).
    # An installed DGL's curr_path will look something like:
    #   $PREFIX/lib/python3.6/site-packages/dgl/_ffi
    ffi_dir = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    source_dir = os.path.join(ffi_dir, "..", "..", "..")
    install_lib_dir = os.path.join(ffi_dir, "..", "..", "..", "..")

    dll_path = []

    if os.environ.get('DGL_LIBRARY_PATH', None):
        dll_path.append(os.environ['DGL_LIBRARY_PATH'])

    if sys.platform.startswith('linux') and os.environ.get('LD_LIBRARY_PATH', None):
        dll_path.extend([p.strip() for p in os.environ['LD_LIBRARY_PATH'].split(":")])
    elif sys.platform.startswith('darwin') and os.environ.get('DYLD_LIBRARY_PATH', None):
        dll_path.extend([p.strip() for p in os.environ['DYLD_LIBRARY_PATH'].split(":")])

    # Pip lib directory
    dll_path.append(os.path.join(ffi_dir, ".."))
    # Default cmake build directory
    dll_path.append(os.path.join(source_dir, "build"))
    dll_path.append(os.path.join(source_dir, "build", "Release"))
    # Default make build directory
    dll_path.append(os.path.join(source_dir, "lib"))

    dll_path.append(install_lib_dir)

    if search_path is not None:
        if isinstance(search_path, (list, tuple, set)):
            dll_path = dll_path + list(search_path)
        elif isinstance(search_path, str):
            dll_path.append(search_path)
        else:
            raise ValueError("type(search_path)={} is invalid".format(type(search_path)))
    dll_path = [str(x.absolute()) if isinstance(x, pathlib.Path)
                else os.path.abspath(x) for x in dll_path]

    if name is None:
        if sys.platform.startswith('win32'):
            name = ['libdgl.dll', 'dgl.dll']
        elif sys.platform.startswith('darwin'):
            name = 'libdgl.dylib'
        else:
            name = 'libdgl.so'

    if isinstance(name, str):
        name = [name]

    lib_dll_path = []
    for n in name:
        lib_dll_path += [os.path.join(p, n) for p in dll_path]

    lib_found = [p for p in lib_dll_path if os.path.isfile(p)]

    if not lib_found:
        message = ('Cannot find the files.\n' +
                   'List of candidates:\n' +
                   str('\n'.join(lib_dll_path)))
        if not optional:
            raise RuntimeError(message)
        return None

    return lib_found


# current version
# We use the version of the incoming release for code
# that is under development.
# The following line is set by dgl/python/update_version.py
__version__ = "0.5.0"
