import torch as th

def sbm(y, p, q):
    """
    Parameters
    ----------
    y: torch.Tensor (N, 1)
    """
    i = (y == y.t()).float()
    r = i * p + (1 - i) * q
    a = th.distributions.Bernoulli(r).sample()
    b = th.triu(a) + th.triu(a, 1).t()
    return b

if __name__ == '__main__':
    N = 10000
    y = th.ones(N, 1)
    p = 1 / N
    q = 0
    a = sbm(y, p, q)
    print(th.sum(a))
