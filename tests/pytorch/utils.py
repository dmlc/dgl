import torch as th

def allclose(a, b):
    return th.allclose(a, b, rtol=1e-4, atol=1e-4)
