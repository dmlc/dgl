import torch

from torch.utils.data import DataLoader

from utils import set_random_seed, load_dataset_for_classification, collate_molgraphs

def main(args):
    args['device'] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(args['random_seed'])

    # Interchangeable with other datasets
    dataset, train_set, val_set, test_set = load_dataset_for_classification(args)
    train_loader = DataLoader(train_set, batch_size=args['batch_size'],
                              collate_fn=collate_molgraphs)
    val_loader = DataLoader(val_set, batch_size=args['batch_size'],
                            collate_fn=collate_molgraphs)
    test_loader = DataLoader(test_set, batch_size=args['batch_size'],
                             collate_fn=collate_molgraphs)

if __name__ == '__main__':
    import argparse

    from configure import get_exp_configure

    parser = argparse.ArgumentParser(description='Molecule Classification')
    parser.add_argument('-m', '--model', type=str, choices=['GCN', 'GAT'],
                        help='Model to use')
    parser.add_argument('-d', '--dataset', type=str, choices=['Tox21'],
                        help='Dataset to use')
    parser.add_argument('-p', '--pre-trained', action='store_true',
                        help='Whether to skip training and use a pre-trained model')
    args = parser.parse_args().__dict__
    args['exp'] = '_'.join([args['model'], args['dataset']])
    args.update(get_exp_configure(args['exp']))

    main(args)
