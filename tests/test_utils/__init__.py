import backend as F
import pytest

parametrize_idtype = pytest.mark.parametrize("idtype", [F.int32, F.int64])
parametrize_idtype64 = pytest.mark.parametrize("idtype", [F.int64])

from .checks import *
from .graph_cases import get_cases
