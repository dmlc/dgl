import torch
import torch.nn as nn
import torch.nn.functional as F


class Aggregator(nn.Module):
    """
    Base Aggregator class. Adapting
    from PR# 403

    This class is not supposed to be called
    """

    def __init__(self):
        super(Aggregator, self).__init__()

    def forward(self, node):
        neighbour = node.mailbox["m"]
        c = self.aggre(neighbour)
        return {"c": c}

    def aggre(self, neighbour):
        # N x F
        raise NotImplementedError


class MeanAggregator(Aggregator):
    """
    Mean Aggregator for graphsage
    """

    def __init__(self):
        super(MeanAggregator, self).__init__()

    def aggre(self, neighbour):
        mean_neighbour = torch.mean(neighbour, dim=1)
        return mean_neighbour


class MaxPoolAggregator(Aggregator):
    """
    Maxpooling aggregator for graphsage
    """

    def __init__(self, in_feats, out_feats, activation, bias):
        super(MaxPoolAggregator, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)
        self.activation = activation
        # Xavier initialization of weight
        nn.init.xavier_uniform_(
            self.linear.weight, gain=nn.init.calculate_gain("relu")
        )

    def aggre(self, neighbour):
        neighbour = self.linear(neighbour)
        if self.activation:
            neighbour = self.activation(neighbour)
        maxpool_neighbour = torch.max(neighbour, dim=1)[0]
        return maxpool_neighbour


class LSTMAggregator(Aggregator):
    """
    LSTM aggregator for graphsage
    """

    def __init__(self, in_feats, hidden_feats):
        super(LSTMAggregator, self).__init__()
        self.lstm = nn.LSTM(in_feats, hidden_feats, batch_first=True)
        self.hidden_dim = hidden_feats
        self.hidden = self.init_hidden()

        nn.init.xavier_uniform_(
            self.lstm.weight, gain=nn.init.calculate_gain("relu")
        )

    def init_hidden(self):
        """
        Defaulted to initialite all zero
        """
        return (
            torch.zeros(1, 1, self.hidden_dim),
            torch.zeros(1, 1, self.hidden_dim),
        )

    def aggre(self, neighbours):
        """
        aggregation function
        """
        # N X F
        rand_order = torch.randperm(neighbours.size()[1])
        neighbours = neighbours[:, rand_order, :]

        (lstm_out, self.hidden) = self.lstm(
            neighbours.view(neighbours.size()[0], neighbours.size()[1], -1)
        )
        return lstm_out[:, -1, :]

    def forward(self, node):
        neighbour = node.mailbox["m"]
        c = self.aggre(neighbour)
        return {"c": c}
