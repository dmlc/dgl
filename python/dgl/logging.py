"""logging module for DGL"""
import logging
import os


def enable_verbose_logging():
    """
    Enable debug level logging for DGL
    """
    os.environ["DMLC_LOG_DEBUG"] = "1"
    logger = logging.getLogger("dgl-core")
    logger.setLevel(logging.DEBUG)
    logging.info("DGL's logging level is set to DEBUG")


def _setup_logger():
    """setup logger"""
    logger = logging.getLogger("dgl-core")
    console = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)s %(levelname)s p:%(processName)s \
        t:%(threadName)s: %(message)s"
    )
    console.setFormatter(formatter)
    console.setLevel(logging.DEBUG)
    logger.addHandler(console)
    logger.propagate = False
    logger.setLevel(logging.INFO)


_setup_logger()

if os.environ.get("DGL_LOG_DEBUG", None) == "1":
    enable_verbose_logging()
