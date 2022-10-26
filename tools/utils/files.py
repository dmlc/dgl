import logging
import os
from contextlib import contextmanager

from numpy.lib.format import open_memmap


@contextmanager
def setdir(path):
    try:
        os.makedirs(path, exist_ok=True)
        cwd = os.getcwd()
        logging.info("Changing directory to %s" % path)
        logging.info("Previously: %s" % cwd)
        os.chdir(path)
        yield
    finally:
        logging.info("Restoring directory to %s" % cwd)
        os.chdir(cwd)
