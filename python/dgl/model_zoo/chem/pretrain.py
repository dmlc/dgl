"""Utilities for using pretrained models."""
import torch
from .dgmg import DGMG
from .gcn import GCNClassifier
from ...data.utils import _get_dgl_url, download

URL = {
    'GCN_Tox21' : 'pre_trained/gcn_tox21.pth',
    'DGMG_ChEMBL_canonical' : 'pre_trained/dgmg_ChEMBL_canonical.pth',
    'DGMG_ChEMBL_random' : 'pre_trained/dgmg_ChEMBL_random.pth',
    'DGMG_ZINC_canonical' : 'pre_trained/dgmg_ZINC_canonical.pth',
    'DGMG_ZINC_random' : 'pre_trained/dgmg_ZINC_random.pth'
}

def download_and_load_checkpoint(model, model_postfix,
                                 local_pretrained_path='pre_trained.pth'):
    """Download pretrained model checkpoint

    Parameters
    ----------
    model : nn.Module
        Instantiated model instance
    model_postfix : str
        Postfix for pretrained model checkpoint
    local_pretrained_path : str
        Local name for the downloaded model checkpoint

    Returns
    -------
    model : nn.Module
        Pretrained model
    """
    url_to_pretrained = _get_dgl_url(model_postfix)
    download(url_to_pretrained, path=local_pretrained_path)
    checkpoint = torch.load(local_pretrained_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def load_pretrained(model_name, **kwargs):
    """Load a pretrained model

    Parameters
    ----------
    model_name : str

    Returns
    -------
    model
    """
    if model_name not in URL:
        return RuntimeError("Cannot find a pretrained model with name {}".format(model_name))

    if model_name == 'GCN_Tox21':
        model = GCNClassifier(in_feats=74,
                              gcn_hidden_feats=[64, 64],
                              n_tasks=12,
                              classifier_hidden_feats=64)
    elif model_name.startswith('DGMG'):
        env = kwargs.get('env')
        model = DGMG(env=env,
                     node_hidden_size=128,
                     num_prop_rounds=2,
                     dropout=0.2)

    print('Pretrained model loaded')

    return download_and_load_checkpoint(model, URL[model_name])
