from . import GCNClassifier
import torch
from ...data.utils import _get_dgl_url, download


def load_pretrained(model_name):
    if model_name == "GCN_Tox21":
        print('Loading pretrained model...')
        url_to_pretrained = _get_dgl_url('pre_trained/gcn_tox21.pth')
        local_pretrained_path = 'pre_trained.pth'
        download(url_to_pretrained, path=local_pretrained_path)
        model = GCNClassifier(74, [64, 64], 12, 64)
        checkpoint = torch.load(local_pretrained_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    else:
        raise RuntimeError("No {} pretrained model found ".format(model_name))

