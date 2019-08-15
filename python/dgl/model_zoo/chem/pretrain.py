"""Utilities for using pretrained models."""
import torch
from .gcn import GCNClassifier
from ...data.utils import _get_dgl_url, download

def load_pretrained(model_name):
    """Load a pretrained model

    Parameters
    ----------
    model_name : str

    Returns
    -------
    model
    """
    if model_name == "GCN_Tox21":
        print('Loading pretrained model...')
        url_to_pretrained = _get_dgl_url('pre_trained/gcn_tox21.pth')
        local_pretrained_path = 'pre_trained.pth'
        download(url_to_pretrained, path=local_pretrained_path)
        model = GCNClassifier(in_feats=74,
                              gcn_hidden_feats=[64, 64],
                              n_tasks=12,
                              classifier_hidden_feats=64)
        checkpoint = torch.load(local_pretrained_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    else:
        raise RuntimeError("Cannot find a pretrained model with name {}".format(model_name))
