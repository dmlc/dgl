import datetime
import dgl
import math
import numpy as np
import os
import pickle
import random
import torch
import torch.distributed as dist
import torch.nn as nn
from collections import defaultdict
from datetime import timedelta
from dgl import DGLGraph
from dgl.data.utils import get_download_dir, download, _get_dgl_url
from dgl.model_zoo.chem.dgmg import MoleculeEnv
from multiprocessing import Pool
from pprint import pprint
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.QED import qed
from torch.utils.data import Dataset

from sascorer import calculateScore

########################################################################################################################
#                                                    configuration                                                     #
########################################################################################################################

def mkdir_p(path, log=True):
    """Create a directory for the specified path.

    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    import errno
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise

def get_date_postfix():
    """Get a date based postfix for directory name.

    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second)

    return post_fix

def setup_log_dir(args):
    """Name and create directory for logging.

    Parameters
    ----------
    args : dict
        Configuration

    Returns
    -------
    log_dir : str
        Path for logging directory
    """
    date_postfix = get_date_postfix()
    log_dir = os.path.join(
        args['log_dir'],
        '{}_{}_{}'.format(args['dataset'], args['order'], date_postfix))
    mkdir_p(log_dir)
    return log_dir

def save_arg_dict(args, filename='settings.txt'):
    """Save all experiment settings in a file.

    Parameters
    ----------
    args : dict
        Configuration
    filename : str
        Name for the file to save settings
    """
    def _format_value(v):
        if isinstance(v, float):
            return '{:.4f}'.format(v)
        elif isinstance(v, int):
            return '{:d}'.format(v)
        else:
            return '{}'.format(v)

    save_path = os.path.join(args['log_dir'], filename)
    with open(save_path, 'w') as f:
        for key, value in args.items():
            f.write('{}\t{}\n'.format(key, _format_value(value)))
    print('Saved settings to {}'.format(save_path))

def configure(args):
    """Use default hyperparameters.

    Parameters
    ----------
    args : dict
        Old configuration

    Returns
    -------
    args : dict
        Updated configuration
    """
    configure = {
        'node_hidden_size': 128,
        'num_propagation_rounds': 2,
        'lr': 1e-4,
        'dropout': 0.2,
        'nepochs': 400,
        'batch_size': 1,
    }
    args.update(configure)
    return args

def set_random_seed(seed):
    """Fix random seed for reproducible results.

    Parameters
    ----------
    seed : int
        Random seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def setup_dataset(args):
    """Dataset setup

    For unsupported dataset, we need to perform data preprocessing.

    Parameters
    ----------
    args : dict
        Configuration
    """
    if args['dataset'] in ['ChEMBL', 'ZINC']:
        print('Built-in support for dataset {} exists.'.format(args['dataset']))
    else:
        print('Configure for new dataset {}...'.format(args['dataset']))
        configure_new_dataset(args['dataset'], args['train_file'], args['val_file'])

def setup(args, train=True):
    """Setup

    Parameters
    ----------
    args : argparse.Namespace
        Configuration
    train : bool
        Whether the setup is for training or evaluation
    """
    # Convert argparse.Namespace into a dict
    args = args.__dict__.copy()
    # Dataset
    args = configure(args)

    # Log
    print('Prepare logging directory...')
    log_dir = setup_log_dir(args)
    args['log_dir'] = log_dir
    save_arg_dict(args)

    if train:
        setup_dataset(args)
        args['checkpoint_dir'] = os.path.join(log_dir, 'checkpoint.pth')
        pprint(args)

    return args

########################################################################################################################
#                                                   multi-process                                                      #
########################################################################################################################

def synchronize(num_processes):
    """Synchronize all processes.

    Parameters
    ----------
    num_processes : int
        Number of subprocesses used
    """
    if num_processes > 1:
        dist.barrier()

def launch_a_process(rank, args, target, minutes=720):
    """Launch a subprocess for training.

    Parameters
    ----------
    rank : int
        Subprocess id
    args : dict
        Configuration
    target : callable
        Target function for the subprocess
    minutes : int
        Timeout minutes for operations executed against the process group
    """
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip=args['master_ip'], master_port=args['master_port'])
    dist.init_process_group(backend='gloo',
                            init_method=dist_init_method,
                            # If you have a larger dataset, you will need to increase it.
                            timeout=timedelta(minutes=minutes),
                            world_size=args['num_processes'],
                            rank=rank)
    assert torch.distributed.get_rank() == rank
    target(rank, args)

########################################################################################################################
#                                                  optimization                                                        #
########################################################################################################################

class Optimizer(nn.Module):
    """Wrapper for optimization

    Parameters
    ----------
    lr : float
        Initial learning rate
    optimizer
        model optimizer
    """
    def __init__(self, lr, optimizer):
        super(Optimizer, self).__init__()
        self.lr = lr
        self.optimizer = optimizer
        self._reset()

    def _reset(self):
        self.optimizer.zero_grad()

    def backward_and_step(self, loss):
        """Backward and update model.

        Parameters
        ----------
        loss : torch.tensor consisting of a float only
        """
        loss.backward()
        self.optimizer.step()
        self._reset()

    def decay_lr(self, decay_rate=0.99):
        """Decay learning rate.

        Parameters
        ----------
        decay_rate : float
            Multiply the current learning rate by the decay_rate
        """
        self.lr *= decay_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

class MultiProcessOptimizer(Optimizer):
    """Wrapper for optimization with multiprocess

    Parameters
    ----------
    n_processes : int
        Number of processes used
    lr : float
        Initial learning rate
    optimizer
        model optimizer
    """
    def __init__(self, n_processes, lr, optimizer):
        super(MultiProcessOptimizer, self).__init__(lr=lr, optimizer=optimizer)
        self.n_processes = n_processes

    def _sync_gradient(self):
        """Average gradients across all subprocesses."""
        for param_group in self.optimizer.param_groups:
            for p in param_group['params']:
                if p.requires_grad and p.grad is not None:
                    dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                    p.grad.data /= self.n_processes

    def backward_and_step(self, loss):
        """Backward and update model.

        Parameters
        ----------
        loss : torch.tensor consisting of a float only
        """
        loss.backward()
        self._sync_gradient()
        self.optimizer.step()
        self._reset()

########################################################################################################################
#                                                         data                                                         #
########################################################################################################################

def initialize_neuralization_reactions():
    """Reference neuralization reactions

    Code adapted from RDKit Cookbook, by Hans de Winter.
    """
    patts = (
        # Imidazoles
        ('[n+;H]', 'n'),
        # Amines
        ('[N+;!H0]', 'N'),
        # Carboxylic acids and alcohols
        ('[$([O-]);!$([O-][#7])]', 'O'),
        # Thiols
        ('[S-;X1]', 'S'),
        # Sulfonamides
        ('[$([N-;X2]S(=O)=O)]', 'N'),
        # Enamines
        ('[$([N-;X2][C,N]=C)]', 'N'),
        # Tetrazoles
        ('[n-]', '[n]'),
        # Sulfoxides
        ('[$([S-]=O)]', 'S'),
        # Amides
        ('[$([N-]C=O)]', 'N'),
    )
    return [(Chem.MolFromSmarts(x), Chem.MolFromSmiles(y, False)) for x, y in patts]

def neutralize_charges(mol, reactions=None):
    """Deprotonation for molecules.

    Code adapted from RDKit Cookbook, by Hans de Winter.

    DGMG currently cannot generate protonated molecules.
    For example, it can only generate
    CC(C)(C)CC1CCC[NH+]1Cc1nnc(-c2ccccc2F)o1
    from
    CC(C)(C)CC1CCCN1Cc1nnc(-c2ccccc2F)o1
    even with correct decisions.

    Deprotonation is therefore an important step to avoid
    false novel molecules.

    Parameters
    ----------
    mol : Chem.rdchem.Mol
    reactions : list of 2-tuples
        Rules for deprotonation

    Returns
    -------
    mol : Chem.rdchem.Mol
        Deprotonated molecule
    """
    if reactions is None:
        reactions = initialize_neuralization_reactions()
    for i, (reactant, product) in enumerate(reactions):
        while mol.HasSubstructMatch(reactant):
            rms = AllChem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]
    return mol

def standardize_mol(mol):
    """Standardize molecule to avoid false novel molecule.

    Kekulize and deprotonate molecules to avoid false novel molecules.

    In addition to deprotonation, we also kekulize molecules to avoid
    explicit Hs in the SMILES. Otherwise we will get false novel molecules
    as well. For example, DGMG can only generate
    O=S(=O)(NC1=CC=CC(C(F)(F)F)=C1)C1=CNC=N1
    from
    O=S(=O)(Nc1cccc(C(F)(F)F)c1)c1c[nH]cn1.

    One downside is that we remove all explicit aromatic rings and to
    explicitly predict aromatic bond might make the learning easier for
    the model.
    """
    reactions = initialize_neuralization_reactions()
    Chem.Kekulize(mol, clearAromaticFlags=True)
    mol = neutralize_charges(mol, reactions)
    return mol

def smiles_to_standard_mol(s):
    """Convert SMILES to a standard molecule.

    Parameters
    ----------
    s : str
        SMILES

    Returns
    -------
    Chem.rdchem.Mol
        Standardized molecule
    """
    mol = Chem.MolFromSmiles(s)
    return standardize_mol(mol)

def mol_to_standard_smile(mol):
    """Standardize a molecule and convert it to a SMILES.

    Parameters
    ----------
    mol : Chem.rdchem.Mol

    Returns
    -------
    str
        SMILES
    """
    return Chem.MolToSmiles(standardize_mol(mol))

def get_atom_and_bond_types(smiles, log=True):
    """Identify the atom types and bond types
    appearing in this dataset.

    Parameters
    ----------
    smiles : list
        List of smiles
    log : bool
        Whether to print the process of pre-processing.

    Returns
    -------
    atom_types : list
        E.g. ['C', 'N']
    bond_types : list
        E.g. [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    """
    atom_types = set()
    bond_types = set()
    n_smiles = len(smiles)

    for i, s in enumerate(smiles):
        if log:
            print('Processing smiles {:d}/{:d}'.format(i + 1, n_smiles))

        mol = smiles_to_standard_mol(s)
        if mol is None:
            continue

        for atom in mol.GetAtoms():
            a_symbol = atom.GetSymbol()
            if a_symbol not in atom_types:
                atom_types.add(a_symbol)

        for bond in mol.GetBonds():
            b_type = bond.GetBondType()
            if b_type not in bond_types:
                bond_types.add(b_type)

    return list(atom_types), list(bond_types)

def eval_decisions(env, decisions):
    """This function mimics the way DGMG generates a molecule and is
    helpful for debugging and verification in data preprocessing.

    Parameters
    ----------
    env : MoleculeEnv
        MDP environment for generating molecules
    decisions : list of 2-tuples of int
        A decision sequence for generating a molecule

    Returns
    -------
    str
        SMILES for the molecule generated with decisions
    """
    env.reset(rdkit_mol=True)
    t = 0

    def whether_to_add_atom(t):
        assert decisions[t][0] == 0
        atom_type = decisions[t][1]
        t += 1
        return t, atom_type

    def whether_to_add_bond(t):
        assert decisions[t][0] == 1
        bond_type = decisions[t][1]
        t += 1
        return t, bond_type

    def decide_atom2(t):
        assert decisions[t][0] == 2
        dst = decisions[t][1]
        t += 1
        return t, dst

    t, atom_type = whether_to_add_atom(t)
    while atom_type != len(env.atom_types):
        env.add_atom(atom_type)
        t, bond_type = whether_to_add_bond(t)
        while bond_type != len(env.bond_types):
            t, dst = decide_atom2(t)
            env.add_bond((env.num_atoms() - 1), dst, bond_type)
            t, bond_type = whether_to_add_bond(t)
        t, atom_type = whether_to_add_atom(t)
    assert t == len(decisions)

    return env.get_current_smiles()

def get_DGMG_smile(env, mol):
    """Mimics the reproduced SMILES with DGMG for a molecule.

    Given a molecule, we are interested in what SMILES we will
    get if we want to generate it with DGMG. This is an important
    step to check false novel molecules.

    Parameters
    ----------
    env : MoleculeEnv
        MDP environment for generating molecules
    mol : Chem.rdchem.Mol
        A molecule

    Returns
    -------
    canonical_smile : str
        SMILES of the generated molecule with a canonical decision sequence
    random_smile : str
        SMILES of the generated molecule with a random decision sequence
    """
    canonical_decisions = env.get_decision_sequence(mol, list(range(mol.GetNumAtoms())))
    canonical_smile = eval_decisions(env, canonical_decisions)

    order = list(range(mol.GetNumAtoms()))
    random.shuffle(order)
    random_decisions = env.get_decision_sequence(mol, order)
    random_smile = eval_decisions(env, random_decisions)

    return canonical_smile, random_smile

def preprocess_dataset(atom_types, bond_types, smiles, max_num_atoms=23):
    """Preprocess the dataset

    1. Standardize the SMILES of the dataset
    2. Only keep the SMILES that DGMG can reproduce
    3. Drop repeated SMILES

    Parameters
    ----------
    atom_types : list
        The types of atoms appearing in a dataset. E.g. ['C', 'N']
    bond_types : list
        The types of bonds appearing in a dataset.
        E.g. [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

    Returns
    -------
    valid_smiles : list of str
        SMILES left after preprocessing
    """
    valid_smiles = []
    env = MoleculeEnv(atom_types, bond_types)

    for id, s in enumerate(smiles):
        print('Processing {:d}/{:d}'.format(id + 1, len(smiles)))

        raw_s = s.strip()
        mol = smiles_to_standard_mol(raw_s)
        if mol is None:
            continue
        standard_s = Chem.MolToSmiles(mol)

        if (max_num_atoms is not None) and (mol.GetNumAtoms() > max_num_atoms):
            continue

        canonical_s, random_s = get_DGMG_smile(env, mol)
        canonical_mol = Chem.MolFromSmiles(canonical_s)
        random_mol = Chem.MolFromSmiles(random_s)

        if (standard_s != canonical_s) or (canonical_s != random_s) or (canonical_mol is None) or (random_mol is None):
            continue

        valid_smiles.append(standard_s)

    valid_smiles = list(set(valid_smiles))
    return valid_smiles

def download_data(dataset, fname):
    """Download dataset if built-in support exists

    Parameters
    ----------
    dataset : str
        Dataset name
    fname : str
        Name of dataset file
    """
    if dataset not in ['ChEMBL', 'ZINC']:
        # For dataset without built-in support, they should be locally processed.
        return

    data_path = fname
    download(_get_dgl_url(os.path.join('dataset', fname)), path=data_path)

def load_smiles_from_file(f_name):
    """Load dataset into a list of SMILES

    Parameters
    ----------
    f_name : str
        Path to a file of molecules, where each line of the file
        is a molecule in SMILES format.

    Returns
    -------
    smiles : list of str
        List of molecules as SMILES
    """
    with open(f_name, 'r') as f:
        smiles = f.read().splitlines()
    return smiles

def write_smiles_to_file(f_name, smiles):
    """Write dataset to a file.

    Parameters
    ----------
    f_name : str
        Path to create a file of molecules, where each line of the file
        is a molecule in SMILES format.
    smiles : list of str
        List of SMILES
    """
    with open(f_name, 'w') as f:
        for s in smiles:
            f.write(s + '\n')

def configure_new_dataset(dataset, train_file, val_file):
    """Configure for a new dataset.

    Parameters
    ----------
    dataset : str
        Dataset name
    train_file : str
        Path to a file with one SMILES a line for training data
    val_file : str
        Path to a file with one SMILES a line for validation data
    """
    assert train_file is not None, 'Expect a file of SMILES for training, got None.'
    assert val_file is not None, 'Expect a file of SMILES for validation, got None.'
    train_smiles = load_smiles_from_file(train_file)
    val_smiles = load_smiles_from_file(val_file)
    all_smiles = train_smiles + val_smiles

    # Get all atom and bond types in the dataset
    path_to_atom_and_bond_types = '_'.join([dataset, 'atom_and_bond_types.pkl'])
    if not os.path.exists(path_to_atom_and_bond_types):
        atom_types, bond_types = get_atom_and_bond_types(all_smiles)
        with open(path_to_atom_and_bond_types, 'wb') as f:
            pickle.dump({'atom_types': atom_types, 'bond_types': bond_types}, f)
    else:
        with open(path_to_atom_and_bond_types, 'rb') as f:
            type_info = pickle.load(f)
            atom_types = type_info['atom_types']
            bond_types = type_info['bond_types']

    # Standardize training data
    path_to_processed_train_data = '_'.join([dataset, 'DGMG', 'train.txt'])
    if not os.path.exists(path_to_processed_train_data):
        processed_train_smiles = preprocess_dataset(atom_types, bond_types, train_smiles, None)
        write_smiles_to_file(path_to_processed_train_data, processed_train_smiles)

    path_to_processed_val_data = '_'.join([dataset, 'DGMG', 'val.txt'])
    if not os.path.exists(path_to_processed_val_data):
        processed_val_smiles = preprocess_dataset(atom_types, bond_types, val_smiles, None)
        write_smiles_to_file(path_to_processed_val_data, processed_val_smiles)

class MoleculeDataset(object):
    """Initialize and split the dataset.

    Parameters
    ----------
    dataset : str
        Dataset name
    order : None or str
        Order to extract a decision sequence for generating a molecule. Default to be None.
    modes : None or list
        List of subsets to use, which can contain 'train', 'val', corresponding to
        training and validation. Default to be None.
    subset_id : int
        With multiprocess training, we partition the training set into multiple subsets and
        each process will use one subset only. This subset_id corresponds to subprocess id.
    n_subsets : int
        With multiprocess training, this corresponds to the number of total subprocesses.
    """
    def __init__(self, dataset, order=None, modes=None, subset_id=0, n_subsets=1):
        super(MoleculeDataset, self).__init__()

        if modes is None:
            modes = []
        else:
            assert order is not None, 'An order should be specified for extracting ' \
                                      'decision sequences.'

        assert order in ['random', 'canonical', None], \
            "Unexpected order option to get sequences of graph generation decisions"
        assert len(set(modes) - {'train', 'val'}) == 0, \
            "modes should be a list, representing a subset of ['train', 'val']"

        self.dataset = dataset
        self.order = order
        self.modes = modes
        self.subset_id = subset_id
        self.n_subsets = n_subsets
        self._setup()

    def collate(self, samples):
        """PyTorch's approach to batch multiple samples.

        For auto-regressive generative models, we process one sample at a time.

        Parameters
        ----------
        samples : list
            A list of length 1 that consists of decision sequence to generate a molecule.

        Returns
        -------
        list
            List of 2-tuples, a decision sequence to generate a molecule
        """
        assert len(samples) == 1
        return samples[0]

    def _create_a_subset(self, smiles):
        """Create a dataset from a subset of smiles.

        Parameters
        ----------
        smiles : list of str
            List of molecules in SMILES format
        """
        # We evenly divide the smiles into multiple susbets with multiprocess
        subset_size = len(smiles) // self.n_subsets
        return Subset(smiles[self.subset_id * subset_size: (self.subset_id + 1) * subset_size],
                      self.order, self.env)

    def _setup(self):
        """
        1. Instantiate an MDP environment for molecule generation
        2. Download the dataset, which is a file of SMILES
        3. Create subsets for training and validation
        """
        if self.dataset == 'ChEMBL':
            # For new datasets, get_atom_and_bond_types can be used to
            # identify the atom and bond types in them.
            self.atom_types = ['O', 'Cl', 'C', 'S', 'F', 'Br', 'N']
            self.bond_types = [Chem.rdchem.BondType.SINGLE,
                               Chem.rdchem.BondType.DOUBLE,
                               Chem.rdchem.BondType.TRIPLE]

        elif self.dataset == 'ZINC':
            self.atom_types = ['Br', 'S', 'C', 'P', 'N', 'O', 'F', 'Cl', 'I']
            self.bond_types = [Chem.rdchem.BondType.SINGLE,
                               Chem.rdchem.BondType.DOUBLE,
                               Chem.rdchem.BondType.TRIPLE]

        else:
            path_to_atom_and_bond_types = '_'.join([self.dataset, 'atom_and_bond_types.pkl'])
            with open(path_to_atom_and_bond_types, 'rb') as f:
                type_info = pickle.load(f)
            self.atom_types = type_info['atom_types']
            self.bond_types = type_info['bond_types']
        self.env = MoleculeEnv(self.atom_types, self.bond_types)

        dataset_prefix = self._dataset_prefix()

        if 'train' in self.modes:
            fname = '_'.join([dataset_prefix, 'train.txt'])
            download_data(self.dataset, fname)
            smiles = load_smiles_from_file(fname)
            self.train_set = self._create_a_subset(smiles)

        if 'val' in self.modes:
            fname = '_'.join([dataset_prefix, 'val.txt'])
            download_data(self.dataset, fname)
            smiles = load_smiles_from_file(fname)
            # We evenly divide the smiles into multiple susbets with multiprocess
            self.val_set = self._create_a_subset(smiles)

    def _dataset_prefix(self):
        """Get the prefix for the data files of supported datasets.

        Returns
        -------
        str
            Prefix for dataset file name
        """
        return '_'.join([self.dataset, 'DGMG'])

class Subset(Dataset):
    """A set of molecules which can be used for training, validation, test.

    Parameters
    ----------
    smiles : list
        List of SMILES for the dataset
    order : str
        Specifies how decision sequences for molecule generation
        are obtained, can be either "random" or "canonical"
    env : MoleculeEnv object
        MDP environment for generating molecules
    """
    def __init__(self, smiles, order, env):
        super(Subset, self).__init__()
        self.smiles = smiles
        self.order = order
        self.env = env
        self._setup()

    def _setup(self):
        """Convert SMILES into rdkit molecule objects.

        Decision sequences are extracted if we use a fixed order.
        """
        smiles_ = []
        mols = []
        for s in self.smiles:
            m = smiles_to_standard_mol(s)
            if m is None:
                continue
            smiles_.append(s)
            mols.append(m)
        self.smiles = smiles_
        self.mols = mols

        if self.order is 'random':
            return

        self.decisions = []
        for m in self.mols:
            self.decisions.append(
                self.env.get_decision_sequence(m, list(range(m.GetNumAtoms())))
            )

    def __len__(self):
        """Get number of molecules in the dataset."""
        return len(self.mols)

    def __getitem__(self, item):
        """Get the decision sequence for generating the molecule indexed by item."""
        if self.order == 'canonical':
            return self.decisions[item]
        else:
            m = self.mols[item]
            nodes = list(range(m.GetNumAtoms()))
            random.shuffle(nodes)
            return self.env.get_decision_sequence(m, nodes)

########################################################################################################################
#                                                  progress tracking                                                   #
########################################################################################################################

class Printer(object):
    def __init__(self, num_epochs, dataset_size, batch_size, writer=None):
        """Wrapper to track the learning progress.

        Parameters
        ----------
        num_epochs : int
            Number of epochs for training
        dataset_size : int
        batch_size : int
        writer : None or SummaryWriter
            If not None, tensorboard will be used to visualize learning curves.
        """
        super(Printer, self).__init__()

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_batches = math.ceil(dataset_size / batch_size)
        self.count = 0
        self.batch_count = 0
        self.writer = writer
        self._reset()

    def _reset(self):
        """Reset when an epoch is completed."""
        self.batch_loss = 0
        self.batch_prob = 0

    def _get_current_batch(self):
        """Get current batch index."""
        remainer = self.batch_count % self.num_batches
        if (remainer == 0):
            return self.num_batches
        else:
            return remainer

    def update(self, epoch, loss, prob):
        """Update learning progress.

        Parameters
        ----------
        epoch : int
        loss : float
        prob : float
        """
        self.count += 1
        self.batch_loss += loss
        self.batch_prob += prob

        if self.count % self.batch_size == 0:
            self.batch_count += 1
            if self.writer is not None:
                self.writer.add_scalar('train_log_prob', self.batch_loss, self.batch_count)
                self.writer.add_scalar('train_prob', self.batch_prob, self.batch_count)

            print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}, prob {:.4f}'.format(
                epoch, self.num_epochs, self._get_current_batch(),
                self.num_batches, self.batch_loss, self.batch_prob))
            self._reset()

########################################################################################################################
#                                                         eval                                                         #
########################################################################################################################

def summarize_a_molecule(smile, checklist=None):
    """Get information about a molecule.

    Parameters
    ----------
    smile : str
        Molecule in SMILES format
    checklist : dict
        Things to learn about the molecule
    """
    if checklist is None:
        checklist = {
            'HBA': Chem.rdMolDescriptors.CalcNumHBA,
            'HBD': Chem.rdMolDescriptors.CalcNumHBD,
            'logP': MolLogP,
            'SA': calculateScore,
            'TPSA': Chem.rdMolDescriptors.CalcTPSA,
            'QED': qed,
            'NumAtoms': lambda mol: mol.GetNumAtoms(),
            'NumBonds': lambda mol: mol.GetNumBonds()
        }

    summary = dict()
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        summary.update({
            'smile': smile,
            'valid': False
        })
        for k in checklist.keys():
            summary[k] = None
    else:
        mol = standardize_mol(mol)
        summary.update({
            'smile': Chem.MolToSmiles(mol),
            'valid': True
        })
        Chem.SanitizeMol(mol)
        for k, f in checklist.items():
            summary[k] = f(mol)

    return summary

def summarize_molecules(smiles, num_processes):
    """Summarize molecules with multiprocess.

    Parameters
    ----------
    smiles : list of str
        List of molecules in SMILES for summarization
    num_processes : int
        Number of processes to use for summarization

    Returns
    -------
    summary_for_valid : dict
        Summary of all valid molecules, where
        summary_for_valid[k] gives the values of all
        valid molecules on item k.
    """
    with Pool(processes=num_processes) as pool:
        result = pool.map(summarize_a_molecule, smiles)

    items = list(result[0].keys())
    items.remove('valid')

    summary_for_valid = defaultdict(list)
    for summary in result:
        if summary['valid']:
            for k in items:
                summary_for_valid[k].append(summary[k])
    return summary_for_valid

def get_unique_smiles(smiles):
    """Given a list of smiles, return a list consisting of unique elements in it.

    Parameters
    ----------
    smiles : list of str
        Molecules in SMILES

    Returns
    -------
    list of str
        Sublist where each SMIES occurs exactly once
    """
    unique_set = set()
    for mol_s in smiles:
        if mol_s not in unique_set:
            unique_set.add(mol_s)

    return list(unique_set)

def get_novel_smiles(new_unique_smiles, reference_unique_smiles):
    """Get novel smiles which do not appear in the reference set.

    Parameters
    ----------
    new_unique_smiles : list of str
        List of SMILES from which we want to identify novel ones
    reference_unique_smiles : list of str
        List of reference SMILES that we already have
    """
    return set(new_unique_smiles).difference(set(reference_unique_smiles))
