# dgl-ascend

Deep Graph Library (DGL) with Ascend NPU support, developed by the BUPT-GAMMA group.

## Installation

```bash
pip install dgl-ascend
```

Requirements: Ascend 910B NPU, CANN >= 8.5, aarch64 Linux, Python 3.11.

## Quick Start

```python
import torch
import torch_npu
import dgl

device = torch.device("npu:0")
g = dgl.graph(([0, 1, 2], [1, 2, 3])).to(device)
g.update_all(dgl.function.copy_u("x", "m"), dgl.function.sum("m", "h"))
```

## License

Apache License 2.0.
