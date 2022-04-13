import torch.nn as nn

class PNA(nn.Module):
    def __init__(self,
                 data_info: dict):
        """"""
        super(PNA, self).__init__()
