"""Define all the constants used by DGL rpc"""

# Maximum size of message queue in bytes
MAX_QUEUE_SIZE = 20 * 1024 * 1024 * 1024

SERVER_EXIT = "server_exit"

DEFAULT_NTYPE = "_N"
DEFAULT_ETYPE = (DEFAULT_NTYPE, "_E", DEFAULT_NTYPE)

DGL2GB_EID = "_dgl2gb_eid"
GB_DST_ID = "_gb_dst_id"
