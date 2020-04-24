import pytest
parametrize_dtype = pytest.mark.parametrize("index_dtype", ['int32', 'int64'])

def check_fail(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        return False
    except:
        return True
