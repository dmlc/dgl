import os
import unittest

import backend as F


def test_set_default_backend():
    default_dir = os.path.join(os.path.expanduser("~"), ".dgl_unit_test")
    F.set_default_backend(default_dir, "pytorch")

    # make sure the config file was created
    assert os.path.exists(os.path.join(default_dir, "config.json"))
