""" The compute function and schedules for segment operators written in TVM. """
import tvm
from tvm import te
from tvm import topi
from tvm.tir import IntImm
from .utils import binary_op_map

# TODO(zihao)
