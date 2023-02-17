import unittest

import backend as F

import dgl
import numpy as np


@unittest.skipIf(
    F._default_context_str == "gpu", reason="GPU random choice not implemented"
)
def test_random_choice():
    # test 1
    a = F.arange(0, 100)
    x = dgl.random.choice(a, 10, replace=True, prob=None)
    assert len(x) == 10
    for i in range(len(x)):
        assert F.asnumpy(x[i]) >= 0 and F.asnumpy(x[i]) < 100
    # test 2, replace=False, small num
    a = F.arange(0, 100)
    x = dgl.random.choice(a, 10, replace=False, prob=None)
    assert len(x) == 10
    for i in range(len(x)):
        assert F.asnumpy(x[i]) >= 0 and F.asnumpy(x[i]) < 100
    # test 3, replace=False, large num
    a = F.arange(0, 100)
    x = dgl.random.choice(a, 100, replace=False, prob=None)
    assert len(x) == 100
    assert np.array_equal(np.sort(F.asnumpy(x)), F.asnumpy(a))
    # test 4, first arg is integer
    x = dgl.random.choice(100, 100, replace=False, prob=None)
    assert len(x) == 100
    assert np.array_equal(np.sort(F.asnumpy(x)), F.asnumpy(a))
    # test 5, with prob
    prob = np.ones((100,))
    prob[37:40] = 0.0
    prob -= prob.min()
    prob /= prob.sum()
    prob = F.tensor(prob)
    x = dgl.random.choice(100, 97, replace=False, prob=prob)
    assert len(x) == 97
    for i in range(len(x)):
        assert F.asnumpy(x[i]) < 37 or F.asnumpy(x[i]) >= 40


if __name__ == "__main__":
    test_random_choice()
