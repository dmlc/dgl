"""MLP for prediction on the output of readout."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch.nn as nn

# pylint: disable=W0221
class MLPPredictor(nn.Module):
    """Two-layer MLP for regression or soft classification
    over multiple tasks from graph representations.

    For classification tasks, the output will be logits, i.e.
    values before sigmoid or softmax.

    Parameters
    ----------
    in_feats : int
        Number of input graph features
    hidden_feats : int
        Number of graph features in hidden layers
    n_tasks : int
        Number of tasks, which is also the output size.
    dropout : float
        The probability for dropout. Default to be 0., i.e. no
        dropout is performed.
    """
    def __init__(self, in_feats, hidden_feats, n_tasks, dropout=0.):
        super(MLPPredictor, self).__init__()

        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_feats),
            nn.Linear(hidden_feats, n_tasks)
        )

    def forward(self, feats):
        """Make prediction.

        Parameters
        ----------
        feats : FloatTensor of shape (B, M3)
            * B is the number of graphs in a batch
            * M3 is the input graph feature size, must match in_feats in initialization

        Returns
        -------
        FloatTensor of shape (B, n_tasks)
        """
        return self.predict(feats)
