import pytest
import backend as F

parametrize_idtype = pytest.mark.parametrize("idtype", [F.int32, F.int64])

from .checks import *
from .graph_cases import get_cases
