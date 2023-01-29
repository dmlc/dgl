import torch.nn as nn


class Envelope(nn.Module):
    """
    Envelope function that ensures a smooth cutoff
    """

    def __init__(self, exponent):
        super(Envelope, self).__init__()

        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        # Envelope function divided by r
        x_p_0 = x.pow(self.p - 1)
        x_p_1 = x_p_0 * x
        x_p_2 = x_p_1 * x
        env_val = 1 / x + self.a * x_p_0 + self.b * x_p_1 + self.c * x_p_2
        return env_val
