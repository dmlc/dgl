import os
import pickle
import shutil
import torch
from dgl import model_zoo

from utils import MoleculeDataset, set_random_seed, download_data,\
    mkdir_p, summarize_molecules, get_unique_smiles, get_novel_smiles

def generate_and_save(log_dir, num_samples, max_num_steps, model):
    with open(os.path.join(log_dir, 'generated_smiles.txt'), 'w') as f:
        for i in range(num_samples):
            with torch.no_grad():
                s = model(rdkit_mol=True, max_num_steps=max_num_steps)
            f.write(s + '\n')

def prepare_for_evaluation(rank, args):
    worker_seed = args['seed'] + rank * 10000
    set_random_seed(worker_seed)
    torch.set_num_threads(1)

    # Setup dataset and data loader
    dataset = MoleculeDataset(args['dataset'], subset_id=rank, n_subsets=args['num_processes'])

    # Initialize model
    if not args['pretrained']:
        model = model_zoo.chem.DGMG(atom_types=dataset.atom_types,
                                    bond_types=dataset.bond_types,
                                    node_hidden_size=args['node_hidden_size'],
                                    num_prop_rounds=args['num_propagation_rounds'], dropout=args['dropout'])
        model.load_state_dict(torch.load(args['model_path'])['model_state_dict'])
    else:
        model = model_zoo.chem.load_pretrained('_'.join(['DGMG', args['dataset'], args['order']]), log=False)
    model.eval()

    worker_num_samples = args['num_samples'] // args['num_processes']
    if rank == args['num_processes'] - 1:
        worker_num_samples += args['num_samples'] % args['num_processes']

    worker_log_dir = os.path.join(args['log_dir'], str(rank))
    mkdir_p(worker_log_dir, log=False)
    generate_and_save(worker_log_dir, worker_num_samples, args['max_num_steps'], model)

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
    train_file = '_'.join([args['dataset'], 'DGMG_train.txt'])
    if not os.path.exists(train_file):
        download_data(args['dataset'], train_file)
    with open(train_file, 'r') as f:
        train_smiles = f.read().splitlines()
    train_summary = summarize_molecules(train_smiles, args['num_processes'])
    with open(os.path.join(args['log_dir'], 'train_summary.pickle'), 'wb') as f:
        pickle.dump(train_summary, f)

    # Summarize generated molecules
    print('Summarizing generated molecules...')
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
        f.write('Total number of generated molecules: {:d}\n'.format(len(smiles)))
        f.write('Validity among all: {:.4f}\n'.format(
            len(valid_generated_smiles) / len(smiles)))
        f.write('Uniqueness among valid ones: {:.4f}\n'.format(
            len(unique_generated_smiles) / len(valid_generated_smiles)))
        f.write('Novelty among unique ones: {:.4f}\n'.format(
            len(novel_generated_smiles) / len(unique_generated_smiles)))

if __name__ == '__main__':
    import argparse
    import datetime
    import time
    from rdkit import rdBase

    from utils import setup

    parser = argparse.ArgumentParser(description='Evaluating DGMG for molecule generation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # configure
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed')

    # dataset and setting
    parser.add_argument('-d', '--dataset',
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
    parser.add_argument('-mn', '--max-num-steps', type=int, default=400,
                        help='Max number of steps allowed in generated molecules to ensure termination')

    # multi-process
    parser.add_argument('-np', '--num-processes', type=int, default=32,
                        help='number of processes to use')
    parser.add_argument('-gt', '--generation-time', type=int, default=600,
                        help='max time (seconds) allowed for generation with multiprocess')

    args = parser.parse_args()
    args = setup(args, train=False)
    rdBase.DisableLog('rdApp.error')

    t1 = time.time()
    if args['num_processes'] == 1:
        prepare_for_evaluation(0, args)
    else:
        import multiprocessing as mp

        procs = []
        for rank in range(args['num_processes']):
            p = mp.Process(target=prepare_for_evaluation, args=(rank, args,))
            procs.append(p)
            p.start()

        while time.time() - t1 <= args['generation_time']:
            if any(p.is_alive() for p in procs):
                time.sleep(5)
            else:
                break
        else:
            print('Timeout, killing all processes.')
            for p in procs:
                p.terminate()
                p.join()

    t2 = time.time()
    print('It took {} for generation.'.format(
        datetime.timedelta(seconds=t2 - t1)))
    aggregate_and_evaluate(args)
