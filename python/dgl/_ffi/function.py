# pylint: disable=invalid-name, unused-import
"""Function namespace."""
from __future__ import absolute_import

import sys
import ctypes
from .base import _LIB, check_call, py_str, c_str, string_types, _FFI_MODE

IMPORT_EXCEPT = RuntimeError if _FFI_MODE == "cython" else ImportError

try:
    # pylint: disable=wrong-import-position
    if _FFI_MODE == "ctypes":
        raise ImportError()
    if sys.version_info >= (3, 0):
        from ._cy3.core import _set_class_function, _set_class_module
        from ._cy3.core import FunctionBase as _FunctionBase
        from ._cy3.core import convert_to_dgl_func
    else:
        from ._cy2.core import _set_class_function, _set_class_module
        from ._cy2.core import FunctionBase as _FunctionBase
        from ._cy2.core import convert_to_dgl_func
except IMPORT_EXCEPT:
    # pylint: disable=wrong-import-position
    from ._ctypes.function import _set_class_function, _set_class_module
    from ._ctypes.function import FunctionBase as _FunctionBase
    from ._ctypes.function import convert_to_dgl_func

FunctionHandle = ctypes.c_void_p

class Function(_FunctionBase):
    """The PackedFunc object.

    Function plays an key role to bridge front and backend in DGL.
    Function provide a type-erased interface, you can call function with positional arguments.

    The compiled module returns Function.
    DGL backend also registers and exposes its API as Functions.
    For example, the developer function exposed in dgl.ir_pass are actually
    C++ functions that are registered as PackedFunc

    The following are list of common usage scenario of dgl.Function.

    - Automatic exposure of C++ API into python
    - To call PackedFunc from python side
    - To call python callbacks to inspect results in generated code
    - Bring python hook into C++ backend

    See Also
    --------
    dgl.register_func: How to register global function.
    dgl.get_global_func: How to get global function.
    """
    pass  # pylint: disable=unnecessary-pass


class ModuleBase(object):
    """Base class for module"""
    __slots__ = ["handle", "_entry", "entry_name"]

    def __init__(self, handle):
        self.handle = handle
        self._entry = None
        self.entry_name = "__dgl_main__"

    def __del__(self):
        check_call(_LIB.DGLModFree(self.handle))

    @property
    def entry_func(self):
        """Get the entry function

        Returns
        -------
        f : Function
            The entry function if exist
        """
        if self._entry:
            return self._entry
        self._entry = self.get_function(self.entry_name)
        return self._entry

    def get_function(self, name, query_imports=False):
        """Get function from the module.

        Parameters
        ----------
        name : str
            The name of the function

        query_imports : bool
            Whether also query modules imported by this module.

        Returns
        -------
        f : Function
            The result function.
        """
        ret_handle = FunctionHandle()
        check_call(_LIB.DGLModGetFunction(
            self.handle, c_str(name),
            ctypes.c_int(query_imports),
            ctypes.byref(ret_handle)))
        if not ret_handle.value:
            raise AttributeError(
                "Module has no function '%s'" %  name)
        return Function(ret_handle, False)

    def import_module(self, module):
        """Add module to the import list of current one.

        Parameters
        ----------
        module : Module
            The other module.
        """
        check_call(_LIB.DGLModImport(self.handle, module.handle))

    def __getitem__(self, name):
        if not isinstance(name, string_types):
            raise ValueError("Can only take string as function name")
        return self.get_function(name)

    def __call__(self, *args):
        if self._entry:
            return self._entry(*args)
        f = self.entry_func
        return f(*args)


def register_func(func_name, f=None, override=False):
    """Register global function

    Parameters
    ----------
    func_name : str or function
        The function name

    f : function, optional
        The function to be registered.

    override: boolean optional
        Whether override existing entry.

    Returns
    -------
    fregister : function
        Register function if f is not specified.

    Examples
    --------
    The following code registers my_packed_func as global function.
    Note that we simply get it back from global function table to invoke
    it from python side. However, we can also invoke the same function
    from C++ backend, or in the compiled DGL code.

    .. code-block:: python

      targs = (10, 10.0, "hello")
      @dgl.register_func
      def my_packed_func(*args):
          assert(tuple(args) == targs)
          return 10
      # Get it out from global function table
      f = dgl.get_global_func("my_packed_func")
      assert isinstance(f, dgl.nd.Function)
      y = f(*targs)
      assert y == 10
    """
    if callable(func_name):
        f = func_name
        func_name = f.__name__

    if not isinstance(func_name, str):
        raise ValueError("expect string function name")

    ioverride = ctypes.c_int(override)
    def register(myf):
        """internal register function"""
        if not isinstance(myf, Function):
            myf = convert_to_dgl_func(myf)
        check_call(_LIB.DGLFuncRegisterGlobal(
            c_str(func_name), myf.handle, ioverride))
        return myf
    if f:
        return register(f)
    return register


def get_global_func(name, allow_missing=False):
    """Get a global function by name

    Parameters
    ----------
    name : str
        The name of the global function

    allow_missing : bool
        Whether allow missing function or raise an error.

    Returns
    -------
    func : dgl.Function
        The function to be returned, None if function is missing.
    """
    handle = FunctionHandle()
    check_call(_LIB.DGLFuncGetGlobal(c_str(name), ctypes.byref(handle)))
    if handle.value:
        return Function(handle, False)
    else:
        if allow_missing:
            return None
        else:
            raise ValueError("Cannot find global function %s" % name)



def list_global_func_names():
    """Get list of global functions registered.

    Returns
    -------
    names : list
       List of global functions names.
    """
    plist = ctypes.POINTER(ctypes.c_char_p)()
    size = ctypes.c_uint()

    check_call(_LIB.DGLFuncListGlobalNames(ctypes.byref(size),
                                           ctypes.byref(plist)))
    fnames = []
    for i in range(size.value):
        fnames.append(py_str(plist[i]))
    return fnames


def extract_ext_funcs(finit):
    """
    Extract the extension PackedFuncs from a C module.

    Parameters
    ----------
    finit : ctypes function
        a ctypes that takes signature of DGLExtensionDeclarer

    Returns
    -------
    fdict : dict of str to Function
        The extracted functions
    """
    fdict = {}
    def _list(name, func):
        fdict[name] = func
    myf = convert_to_dgl_func(_list)
    ret = finit(myf.handle)
    _ = myf
    if ret != 0:
        raise RuntimeError("cannot initialize with %s" % finit)
    return fdict

def _get_api(f):
    flocal = f
    flocal.is_global = True
    return flocal

def _init_api(namespace, target_module_name=None):
    """Initialize api for a given module name

    namespace : str
       The namespace of the source registry

    target_module_name : str
       The target module name if different from namespace
    """
    target_module_name = (
        target_module_name if target_module_name else namespace)
    if namespace.startswith("dgl."):
        _init_api_prefix(target_module_name, namespace[4:])
    else:
        _init_api_prefix(target_module_name, namespace)


def _init_api_prefix(module_name, prefix):
    module = sys.modules[module_name]

    for name in list_global_func_names():
        if name.startswith("_") and not name.startswith('_deprecate'):
            # internal APIs are ignored
            continue
        name_split = name.rsplit('.', 1)
        if name_split[0] != prefix:
            continue

        if len(name_split) == 1:
            print('Warning: invalid API name "%s".' % name)
            continue
        fname = name_split[1]
        target_module = module

        f = get_global_func(name)
        ff = _get_api(f)
        ff.__name__ = fname
        ff.__doc__ = ("DGL PackedFunc %s. " % fname)
        setattr(target_module, ff.__name__, ff)

def _init_internal_api():
    for name in list_global_func_names():
        if not name.startswith("_") or name.startswith('_deprecate'):
            # normal APIs are ignored
            continue
        target_module = sys.modules["dgl._api_internal"]
        fname = name
        if fname.find(".") != -1:
            print('Warning: invalid API name "%s".' % fname)
            continue
        f = get_global_func(name)
        ff = _get_api(f)
        ff.__name__ = fname
        ff.__doc__ = ("DGL PackedFunc %s. " % fname)
        setattr(target_module, ff.__name__, ff)

_set_class_function(Function)
