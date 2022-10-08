import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from .sparse import *
from .tensor import *
