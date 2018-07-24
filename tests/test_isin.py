import sys
import time
import numpy as np
import numpy.random as npr

N = int(sys.argv[1])
x = npr.randint(N, size=N)
y = npr.randint(N, size=N)
t0 = time.time()
for i in range(int(sys.argv[2])):
    isin = np.isin(x, y)
print((time.time() - t0) / (i + 1))
