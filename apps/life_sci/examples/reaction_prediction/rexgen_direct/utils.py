import errno
import numpy as np
import os
import random
import torch

from dgllife.data import USPTO

def mkdir_p(path):
    """Create a folder for the given path.
    Parameters
    ----------
    path: str
        Folder to create
    """
    try:
        os.makedirs(path)
        print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            print('Directory {} already exists.'.format(path))
        else:
            raise

def setup(args, seed=0):
    """Setup for the experiment:

    1. Decide whether to use CPU or GPU for training
    2. Fix random seed for python, NumPy and PyTorch.

    Parameters
    ----------
    seed : int
        Random seed to use.

    Returns
    -------
    args
        Updated configuration
    """
    if torch.cuda.is_available():
        args['device'] = 'cuda:0'
    else:
        args['device'] = 'cpu'

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    mkdir_p(args['result_path'])

    return args

def load_data():
    """Load and pre-process the dataset.

    Construct DGLGraphs and featurize their nodes/edges.

    Returns
    -------
    train_set
        Training subset
    val_set
        Validation subset
    test_set
        Test subset
    """
    train_set = USPTO('train')
    val_set = USPTO('val')
    test_set = USPTO('test')

    return train_set, val_set, test_set
