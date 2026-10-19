"""concat 算子的 Python 封装。

C++ 侧 `aten::ConcatND`（src/array/array.cc）通过
`DGL_REGISTER_GLOBAL("ndarray._CAPI_DGLConcatND")` 暴露给 Python。
本模块负责 torch_npu 张量 <-> dgl NDArray 的零拷贝转换，供 test_concat.py
与 concat_golden.py 做逐位对比。

- Ascend 设备 + fp32/fp16 走 AscendC kernel（src/array/ascend/concat_kernel.cpp）
- 其他设备或 dtype 走 C++ 侧的通用 CopyDataFromTo 分段拷贝路径
"""

import torch
import dgl.backend as F
from dgl.ndarray import _CAPI_DGLConcatND


def concat_impl(tensors, dim=0):
    """沿 dim 轴拼接一组张量。

    Args:
        tensors: list[torch.Tensor]，同 dtype / 同 device，除 dim 轴外形状一致。
        dim: 拼接轴，支持负数索引。

    Returns:
        torch.Tensor，与 tensors[0] 同 device 同 dtype。
    """
    nds = [F.zerocopy_to_dgl_ndarray(t) for t in tensors]
    # torch_npu 在自有流上异步下发；libdgl 的 AscendC kernel 跑在默认流。
    # _CAPI_DGLConcatND 这类 global func 在 import 时经 get_global_func 获取，
    # 流桥接只在获取句柄时同步过一次，此处显式同步保证输入写入先于 kernel。
    torch.npu.synchronize()
    out = _CAPI_DGLConcatND(nds, dim)
    return F.zerocopy_from_dgl_ndarray(out)
