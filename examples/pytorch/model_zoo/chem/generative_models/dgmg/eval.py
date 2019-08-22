import datetime
import os
import pickle
import shutil
import time
import torch
from dgl import model_zoo

from utils import synchronize, launch_a_process, MoleculeDataset, set_random_seed, \
    mkdir_p, summarize_molecules, get_unique_smiles, get_novel_smiles

def generate_and_save(rank, log_dir, num_samples, model):
    smiles = []
    with open(os.path.join(log_dir, 'generated_smiles.txt'), 'w') as f:
        for i in range(num_samples):
            if rank == 0:
                print('Generating the {:d}/{:d}th smile'.format(i + 1, num_samples))
            with torch.no_grad():
                s = model(rdkit_mol=True)
            smiles.append(s)
            f.write(s + '\n')
    return smiles

def prepare_for_evaluation(rank, args):
    if rank == 0:
        t1 = time.time()

    worker_seed = args['seed'] + rank * 10000
    set_random_seed(worker_seed)
    torch.set_num_threads(1)

    # Setup dataset and data loader
    dataset = MoleculeDataset(args['dataset'], subset_id=rank, n_subsets=args['num_processes'])
    env = dataset.env

    # Initialize model
    if rank == 0:
        print('Loading the trained model...')

    if not args['pretrained']:
        model = model_zoo.chem.DGMG(env=env, node_hidden_size=args['node_hidden_size'],
                                    num_prop_rounds=args['num_propagation_rounds'], dropout=args['dropout'])
        model.load_state_dict(torch.load(args['model_path']))
    else:
        model = model_zoo.chem.load_pretrained('_'.join(['DGMG', args['dataset'], args['order']]), env=env)
    model.eval()

    worker_num_samples = args['num_samples'] // args['num_processes']
    if rank == args['num_processes'] - 1:
        worker_num_samples += args['num_samples'] % args['num_processes']

    worker_log_dir = os.path.join(args['log_dir'], str(rank))
    mkdir_p(worker_log_dir, log=False)
    generate_and_save(rank, worker_log_dir, worker_num_samples, model)

    if rank == 0:
        t2 = time.time()
        print('It took {} to generate {:d} molecules.'.format(
            datetime.timedelta(seconds=t2 - t1), args['num_samples']))

    synchronize(args['num_processes'])

def remove_worker_tmp_dir(args):
    for rank in range(args['num_processes']):
        worker_path = os.path.join(args['log_dir'], str(rank))
        try:
            shutil.rmtree(worker_path)
        except OSError:
            print('Directory {} does not exist!'.format(worker_path))

def aggregate_and_evaluate(args):
    print('Merging generated SMILES into a single file...')
    smiles = []
    for rank in range(args['num_processes']):
        with open(os.path.join(args['log_dir'], str(rank), 'generated_smiles.txt'), 'r') as f:
            rank_smiles = f.read().splitlines()
        smiles.extend(rank_smiles)

    with open(os.path.join(args['log_dir'], 'generated_smiles.txt'), 'w') as f:
        for s in smiles:
            f.write(s + '\n')

    print('Removing temporary dirs...')
    remove_worker_tmp_dir(args)

    # Summarize training molecules
    print('Summarizing training molecules...')
    with open('_'.join([args['dataset'], 'DGMG_train.txt']), 'r') as f:
        train_smiles = f.read().splitlines()
    train_summary = summarize_molecules(train_smiles, args['num_processes'])
    with open(os.path.join(args['log_dir'], 'train_summary.pickle'), 'wb') as f:
        pickle.dump(train_summary, f)

    # Summarize generated molecules
    print('Summarizing validation molecules...')
    generation_summary = summarize_molecules(smiles, args['num_processes'])
    with open(os.path.join(args['log_dir'], 'generation_summary.pickle'), 'wb') as f:
        pickle.dump(generation_summary, f)

    # Stats computation
    print('Preparing generation statistics...')
    valid_generated_smiles = generation_summary['smile']
    unique_generated_smiles = get_unique_smiles(valid_generated_smiles)
    unique_train_smiles = get_unique_smiles(train_summary['smile'])
    novel_generated_smiles = get_novel_smiles(unique_generated_smiles, unique_train_smiles)
    with open(os.path.join(args['log_dir'], 'generation_stats.txt'), 'w') as f:
        f.write('Validity among all: {:.4f}\n'.format(
            len(valid_generated_smiles) / len(smiles)))
        f.write('Uniqueness among valid ones: {:.4f}\n'.format(
            len(unique_generated_smiles) / len(valid_generated_smiles)))
        f.write('Novelty among unique ones: {:.4f}\n'.format(
            len(novel_generated_smiles) / len(unique_generated_smiles)))

if __name__ == '__main__':
    import argparse

    from utils import setup

    parser = argparse.ArgumentParser(description='Evaluating DGMG for molecule generation')

    # configure
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed')

    # dataset and setting
    parser.add_argument('-d', '--dataset', choices=['ChEMBL', 'ZINC'], default='ChEMBL',
                        help='dataset to use')
    parser.add_argument('-o', '--order', choices=['random', 'canonical'],
                        help='order to generate graphs, used for naming evaluation directory')

    # log
    parser.add_argument('-l', '--log-dir', default='./eval_results',
                        help='folder to save evaluation results')

    parser.add_argument('-p', '--model-path', type=str, default=None,
                        help='path to saved model')
    parser.add_argument('-pr', '--pretrained', action='store_true',
                        help='Whether to use a pre-trained model')
    parser.add_argument('-ns', '--num-samples', type=int, default=100000,
                        help='Number of molecules to generate')

    # multi-process
    parser.add_argument('-np', '--num-processes', type=int, default=64,
                        help='number of processes to use')
    parser.add_argument('-mi', '--master-ip', type=str, default='127.0.0.1')
    parser.add_argument('-mp', '--master-port', type=str, default='12345')

    args = parser.parse_args()
    args = setup(args, train=False)

    if args['num_processes'] == 1:
        prepare_for_evaluation(0, args)
    else:
        mp = torch.multiprocessing.get_context('spawn')
        procs = []
        for rank in range(args['num_processes']):
            procs.append(mp.Process(target=launch_a_process,
                                    args=(rank, args, prepare_for_evaluation), daemon=True))
            procs[-1].start()
        for p in procs:
            p.join()

    aggregate_and_evaluate(args)
