import pytest
import backend as F

if F._default_context_str == 'cpu':
    parametrize_dtype = pytest.mark.parametrize("idtype", [F.int32, F.int64])
else:
    parametrize_dtype = pytest.mark.parametrize("idtype", [F.int32, F.int64])

def check_fail(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        return False
    except:
        return True
