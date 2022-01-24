import logging
import os


def enable_verbose_logging():
    """
    Enable debug level logging for DGL
    """
    import os
    import logging
    os.environ["DMLC_LOG_DEBUG"] = "1"
    logger = logging.getLogger("dgl-core")
    logger.setLevel(logging.DEBUG)
    logging.info("DGL's logging level is set to DEBUG")


logger = logging.getLogger("dgl-core")
console = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s %(filename)s:%(lineno)s %(levelname)s p:%(processName)s t:%(threadName)s: %(message)s"
)
console.setFormatter(formatter)
console.setLevel(logging.DEBUG)
# add the handlers to the logger
logger.addHandler(console)
logger.propagate = False

if "DGL_LOG_DEBUG" in os.environ and os.environ["DGL_LOG_DEBUG"] == "1":
    enable_verbose_logging()
else:
    logger = logging.getLogger("dgl-core")
    logger.setLevel(logging.INFO)
