"""Define all the constants used by DGL rpc"""

# Maximum size of message queue in bytes
MAX_QUEUE_SIZE = 20 * 1024 * 1024 * 1024

SERVER_EXIT = "server_exit"
SERVER_KEEP_ALIVE = "server_keep_alive"

DEFAULT_NTYPE = "_N"
DEFAULT_ETYPE = (DEFAULT_NTYPE, "_E", DEFAULT_NTYPE)
