import torch
import torch.nn as nn


class SilencedModule(nn.Module):
    def forward(self, x):
        return x

class Modulesilencer():
    def __init__(self, module):
        self.module = module
        self.silenced = []

    def silence(self, silence_modules):
        for k in self.module._modules.keys():
            v = self.module._modules[k]
            for clazz in silence_modules:
                if isinstance(v, clazz):
                    self.silenced.append((self.module._modules, k, v))
                    self.module._modules[k] = SilencedModule()

    def unsilence(self):
        for (obj, k, v) in self.silenced:
            obj[k] = v
