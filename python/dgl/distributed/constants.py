"""Define all the constants used by DGL rpc"""

# Maximum size of message queue in bytes
MAX_QUEUE_SIZE = 20*1024*1024*1024


def get_type_str(dtype):
    """Get data type string
    """
    if 'float16' in str(dtype):
        return 'float16'
    elif 'float32' in str(dtype):
        return 'float32'
    elif 'float64' in str(dtype):
        return 'float64'
    elif 'uint8' in str(dtype):
        return 'uint8'
    elif 'int8' in str(dtype):
        return 'int8'
    elif 'int16' in str(dtype):
        return 'int16'
    elif 'int32' in str(dtype):
        return 'int32'
    elif 'int64' in str(dtype):
        return 'int64'
    else:
        raise RuntimeError('Unknown data type: %s' % str(dtype))
