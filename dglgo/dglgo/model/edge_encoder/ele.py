import torch
import torch.nn as nn
import torch.nn.functional as F


class ElementWiseProductPredictor(nn.Module):
    def __init__(
        self,
        data_info: dict,
        hidden_size: int = 64,
        num_layers: int = 2,
        bias: bool = True,
    ):
        """Elementwise product model for edge scores

        Parameters
        ----------
        data_info : dict
            The information about the input dataset.
        hidden_size : int
            Hidden size.
        num_layers : int
            Number of hidden layers.
        bias : bool
            Whether to use bias in the linaer layer.
        """
        super(ElementWiseProductPredictor, self).__init__()
        lins_list = []
        in_size, out_size = data_info["in_size"], data_info["out_size"]
        for i in range(num_layers):
            in_hiddnen = in_size if i == 0 else hidden_size
            out_hidden = hidden_size if i < num_layers - 1 else out_size
            lins_list.append(nn.Linear(in_hiddnen, out_hidden, bias=bias))
            if i < num_layers - 1:
                lins_list.append(nn.ReLU())
        self.linear = nn.Sequential(*lins_list)

    def forward(self, h_src, h_dst):
        h = h_src * h_dst
        h = self.linear(h)
        h = torch.sigmoid(h)
        return h
