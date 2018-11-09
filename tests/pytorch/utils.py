import torch as th

def allclose(a, b):
    return th.allclose(a, b, rtol=1e-4, atol=1e-4)

def check_fail(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        return False
    except:
        return True
