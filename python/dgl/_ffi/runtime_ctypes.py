"""Common runtime ctypes."""
# pylint: disable=invalid-name, super-init-not-called
from __future__ import absolute_import

import ctypes
import json

import numpy as np

from .. import _api_internal
from .base import _LIB, check_call

dgl_shape_index_t = ctypes.c_int64


class TypeCode(object):
    """Type code used in API calls"""

    INT = 0
    UINT = 1
    FLOAT = 2
    HANDLE = 3
    NULL = 4
    DGL_DATA_TYPE = 5
    DGL_CONTEXT = 6
    ARRAY_HANDLE = 7
    OBJECT_HANDLE = 8
    MODULE_HANDLE = 9
    FUNC_HANDLE = 10
    STR = 11
    BYTES = 12
    NDARRAY_CONTAINER = 13
    EXT_BEGIN = 15


class DGLByteArray(ctypes.Structure):
    """Temp data structure for byte array."""

    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_byte)),
        ("size", ctypes.c_size_t),
    ]


class DGLDataType(ctypes.Structure):
    """DGL datatype structure"""

    _fields_ = [
        ("type_code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]
    CODE2STR = {0: "int", 1: "uint", 2: "float", 4: "handle"}
    _cache = {}

    def __new__(cls, type_str):
        if type_str in cls._cache:
            return cls._cache[type_str]

        inst = super(DGLDataType, cls).__new__(DGLDataType)

        if isinstance(type_str, np.dtype):
            type_str = str(type_str)
        arr = type_str.split("x")
        head = arr[0]
        inst.lanes = int(arr[1]) if len(arr) > 1 else 1
        bits = 32

        if head.startswith("int"):
            inst.type_code = 0
            head = head[3:]
        elif head.startswith("uint"):
            inst.type_code = 1
            head = head[4:]
        elif head.startswith("float"):
            inst.type_code = 2
            head = head[5:]
        elif head.startswith("handle"):
            inst.type_code = 4
            bits = 64
            head = ""
        else:
            raise ValueError("Do not know how to handle type %s" % type_str)
        bits = int(head) if head else bits
        inst.bits = bits

        cls._cache[type_str] = inst
        return inst

    def __init__(self, type_str):
        pass

    def __repr__(self):
        x = "%s%d" % (DGLDataType.CODE2STR[self.type_code], self.bits)
        if self.lanes != 1:
            x += "x%d" % self.lanes
        return x

    def __eq__(self, other):
        return (
            self.bits == other.bits
            and self.type_code == other.type_code
            and self.lanes == other.lanes
        )

    def __ne__(self, other):
        return not self.__eq__(other)


RPC_SESS_MASK = 128


class DGLContext(ctypes.Structure):
    """DGL context strucure."""

    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]
    MASK2STR = {
        1: "cpu",
        2: "gpu",
        4: "opencl",
        5: "aocl",
        6: "sdaccel",
        7: "vulkan",
        8: "metal",
        9: "vpi",
        10: "rocm",
        11: "opengl",
        12: "ext_dev",
    }
    STR2MASK = {
        "llvm": 1,
        "stackvm": 1,
        "cpu": 1,
        "gpu": 2,
        "cuda": 2,
        "nvptx": 2,
        "cl": 4,
        "opencl": 4,
        "aocl": 5,
        "aocl_sw_emu": 5,
        "sdaccel": 6,
        "vulkan": 7,
        "metal": 8,
        "vpi": 9,
        "rocm": 10,
        "opengl": 11,
        "ext_dev": 12,
    }
    _cache = {}

    def __new__(cls, device_type, device_id):
        if (device_type, device_id) in cls._cache:
            return cls._cache[(device_type, device_id)]

        inst = super(DGLContext, cls).__new__(DGLContext)

        inst.device_type = device_type
        inst.device_id = device_id

        cls._cache[(device_type, device_id)] = inst
        return inst

    def __init__(self, device_type, device_id):
        pass

    @property
    def exist(self):
        """Whether this device exist."""
        return (
            _api_internal._GetDeviceAttr(self.device_type, self.device_id, 0)
            != 0
        )

    @property
    def max_threads_per_block(self):
        """Maximum number of threads on each block."""
        return _api_internal._GetDeviceAttr(self.device_type, self.device_id, 1)

    @property
    def warp_size(self):
        """Number of threads that executes in concurrent."""
        return _api_internal._GetDeviceAttr(self.device_type, self.device_id, 2)

    @property
    def max_shared_memory_per_block(self):
        """Total amount of shared memory per block in bytes."""
        return _api_internal._GetDeviceAttr(self.device_type, self.device_id, 3)

    @property
    def compute_version(self):
        """Get compute verison number in string.

        Currently used to get compute capability of CUDA device.

        Returns
        -------
        version : str
            The version string in `major.minor` format.
        """
        return _api_internal._GetDeviceAttr(self.device_type, self.device_id, 4)

    @property
    def device_name(self):
        """Return the string name of device."""
        return _api_internal._GetDeviceAttr(self.device_type, self.device_id, 5)

    @property
    def max_clock_rate(self):
        """Return the max clock frequency of device."""
        return _api_internal._GetDeviceAttr(self.device_type, self.device_id, 6)

    @property
    def multi_processor_count(self):
        """Return the number of compute units of device."""
        return _api_internal._GetDeviceAttr(self.device_type, self.device_id, 7)

    @property
    def max_thread_dimensions(self):
        """Return the maximum size of each thread axis

        Returns
        -------
        dims: List of int
            The maximum length of threadIdx.x, threadIdx.y, threadIdx.z
        """
        return json.loads(
            _api_internal._GetDeviceAttr(self.device_type, self.device_id, 8)
        )

    def sync(self):
        """Synchronize until jobs finished at the context."""
        check_call(_LIB.DGLSynchronize(self.device_type, self.device_id, None))

    def __eq__(self, other):
        return (
            isinstance(other, DGLContext)
            and self.device_id == other.device_id
            and self.device_type == other.device_type
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        if self.device_type >= RPC_SESS_MASK:
            tbl_id = self.device_type / RPC_SESS_MASK - 1
            dev_type = self.device_type % RPC_SESS_MASK
            return "remote[%d]:%s(%d)" % (
                tbl_id,
                DGLContext.MASK2STR[dev_type],
                self.device_id,
            )
        return "%s(%d)" % (
            DGLContext.MASK2STR[self.device_type],
            self.device_id,
        )

    def __hash__(self):
        return hash((self.device_type, self.device_id))


class DGLArray(ctypes.Structure):
    """DGLValue in C API"""

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("ctx", DGLContext),
        ("ndim", ctypes.c_int),
        ("dtype", DGLDataType),
        ("shape", ctypes.POINTER(dgl_shape_index_t)),
        ("strides", ctypes.POINTER(dgl_shape_index_t)),
        ("byte_offset", ctypes.c_uint64),
    ]


DGLArrayHandle = ctypes.POINTER(DGLArray)

DGLStreamHandle = ctypes.c_void_p
