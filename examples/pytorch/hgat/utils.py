# TODO: Add the data loader related here.
import torch as th
import torch.nn as nn

class Zeros(nn.Module):
    '''
    Residual Place holder, which returns zeros for non residual case
    '''
    def __init__(self):
        super(Zeros,self).__init__()

    def forward(self,x):
        return 0