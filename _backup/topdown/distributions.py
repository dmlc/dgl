import torch as T
import torch.nn.functional as F
from torch.distributions import Normal

class LogNormal(Normal):
    def sample(self):
        x = Normal.sample(self)
        return T.exp(x)

    def sample_n(self, n):
        x = Normal.sample_n(self, n)
        return T.exp(x)

    def log_prob(self, x):
        y = T.log(x)
        return Normal.log_prob(self, y) - y


class SigmoidNormal(Normal):
    def sample(self):
        x = Normal.sample(self)
        return F.sigmoid(x)

    def sample_n(self, n):
        x = Normal.sample_n(self, n)
        return F.sigmoid(x)

    def log_prob(self, x):
        # sigmoid^{-1}(x) = log(x) - log(1 - x)
        y = T.log(x + 1e-8) - T.log(1 - x + 1e-8)
        return Normal.log_prob(self, y) - T.log(x + 1e-8) - T.log(1 - x + 1e-8)
