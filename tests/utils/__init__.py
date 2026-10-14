import os

import backend as F
import pytest

parametrize_idtype = pytest.mark.parametrize("idtype", [F.int32, F.int64])


def rendezvous_port(offset=0):
    """Return the TCP port for distributed test rendezvous.

    Set DGL_TEST_MASTER_PORT to move the tests off the default ports when
    something else on the machine already listens on them. This matters when
    tests run in a container that shares the host network namespace.

    Pass an offset to claim a port distinct from the base one, so that suites
    using different offsets stay independent if they ever run concurrently.
    """
    return int(os.environ.get("DGL_TEST_MASTER_PORT", 12345)) + offset


from .checks import *
from .graph_cases import get_cases
