import pytest
import backend as F

if F._default_context_str == 'cpu':
    parametrize_dtype = pytest.mark.parametrize("index_dtype", ['int32', 'int64'])
else:
    # only test int32 on GPU because many graph operators are not supported for int64.
    parametrize_dtype = pytest.mark.parametrize("index_dtype", ['int32'])

def check_fail(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        return False
    except:
        return True
