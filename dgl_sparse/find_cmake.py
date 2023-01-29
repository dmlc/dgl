import os

import torch

cmake_prefix_path = getattr(
    torch.utils,
    "cmake_prefix_path",
    os.path.join(os.path.dirname(torch.__file__), "share", "cmake"),
)
version = torch.__version__.split("+")[0]
print(";".join([cmake_prefix_path, version]))
