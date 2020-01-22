"""Utilities for using pretrained models."""
import os
import torch
import torch.nn.functional as F

from dgl.data.utils import _get_dgl_url, download, get_download_dir, extract_archive
from rdkit import Chem

from ..model import GCNPredictor, GATPredictor, DGMG, DGLJTNNVAE

__all__ = ['load_pretrained']

URL = {
    'GCN_Tox21': 'dglls/pre_trained/gcn_tox21.pth',
    'GAT_Tox21': 'dglls/pre_trained/gat_tox21.pth',
    'DGMG_ChEMBL_canonical': 'pre_trained/dgmg_ChEMBL_canonical.pth',
    'DGMG_ChEMBL_random': 'pre_trained/dgmg_ChEMBL_random.pth',
    'DGMG_ZINC_canonical': 'pre_trained/dgmg_ZINC_canonical.pth',
    'DGMG_ZINC_random': 'pre_trained/dgmg_ZINC_random.pth',
    'JTNN_ZINC': 'pre_trained/JTNN_ZINC.pth'
}

def download_and_load_checkpoint(model_name, model, model_postfix,
                                 local_pretrained_path='pre_trained.pth', log=True):
    """Download pretrained model checkpoint

    The model will be loaded to CPU.

    Parameters
    ----------
    model_name : str
        Name of the model
    model : nn.Module
        Instantiated model instance
    model_postfix : str
        Postfix for pretrained model checkpoint
    local_pretrained_path : str
        Local name for the downloaded model checkpoint
    log : bool
        Whether to print progress for model loading

    Returns
    -------
    model : nn.Module
        Pretrained model
    """
    url_to_pretrained = _get_dgl_url(model_postfix)
    local_pretrained_path = '_'.join([model_name, local_pretrained_path])
    download(url_to_pretrained, path=local_pretrained_path, log=log)
    checkpoint = torch.load(local_pretrained_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    if log:
        print('Pretrained model loaded')

    return model

def load_pretrained(model_name, log=True):
    """Load a pretrained model

    Parameters
    ----------
    model_name : str
        Currently supported options include

        * ``'GCN_Tox21'``
        * ``'GAT_Tox21'``
        * ``'AttentiveFP_Aromaticity'``
        * ``'DGMG_ChEMBL_canonical'``
        * ``'DGMG_ChEMBL_random'``
        * ``'DGMG_ZINC_canonical'``
        * ``'DGMG_ZINC_random'``
        * ``'JTNN_ZINC'``

    log : bool
        Whether to print progress for model loading

    Returns
    -------
    model
    """
    if model_name not in URL:
        raise RuntimeError("Cannot find a pretrained model with name {}".format(model_name))

    if model_name == 'GCN_Tox21':
        model = GCNPredictor(in_feats=74,
                             hidden_feats=[64, 64],
                             classifier_hidden_feats=64,
                             n_tasks=12)

    elif model_name == 'GAT_Tox21':
        model = GATPredictor(in_feats=74,
                             hidden_feats=[32, 32],
                             num_heads=[4, 4],
                             agg_modes=['flatten', 'mean'],
                             activations=[F.elu, None],
                             classifier_hidden_feats=64,
                             n_tasks=12)

    elif model_name.startswith('DGMG'):
        if model_name.startswith('DGMG_ChEMBL'):
            atom_types = ['O', 'Cl', 'C', 'S', 'F', 'Br', 'N']
        elif model_name.startswith('DGMG_ZINC'):
            atom_types = ['Br', 'S', 'C', 'P', 'N', 'O', 'F', 'Cl', 'I']
        bond_types = [Chem.rdchem.BondType.SINGLE,
                      Chem.rdchem.BondType.DOUBLE,
                      Chem.rdchem.BondType.TRIPLE]

        model = DGMG(atom_types=atom_types,
                     bond_types=bond_types,
                     node_hidden_size=128,
                     num_prop_rounds=2,
                     dropout=0.2)

    elif model_name == "JTNN_ZINC":
        default_dir = get_download_dir()
        vocab_file = '{}/jtnn/{}.txt'.format(default_dir, 'vocab')
        if not os.path.exists(vocab_file):
            zip_file_path = '{}/jtnn.zip'.format(default_dir)
            download(_get_dgl_url('dglls/jtnn.zip'), path=zip_file_path)
            extract_archive(zip_file_path, '{}/jtnn'.format(default_dir))
        model = DGLJTNNVAE(vocab_file=vocab_file,
                           depth=3,
                           hidden_size=450,
                           latent_size=56)

    return download_and_load_checkpoint(model_name, model, URL[model_name], log=log)
