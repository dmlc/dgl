import torch
import os

print(getattr(
    torch.utils,
    "cmake_prefix_path",
    os.path.join(os.path.dirname(torch.__file__), "share", "cmake")),
    torch.__version__.split('+')[0], sep=';')
