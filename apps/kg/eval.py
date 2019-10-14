from dataloader import EvalDataset, TrainDataset
from dataloader import get_dataset

import argparse
import torch.multiprocessing as mp
import os
import logging
import time
import pickle

backend = os.environ.get('DGLBACKEND')
if backend.lower() == 'mxnet':
    from train_mxnet import load_model_from_checkpoint
    from train_mxnet import test
else:
    from train_pytorch import load_model_from_checkpoint
    from train_pytorch import test

class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--model_name', default='TransE',
                          choices=['TransE', 'TransH', 'TransR', 'TransD',
                                   'RESCAL', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE'],
                          help='model to use')
        self.add_argument('--data_path', type=str, default='data',
                          help='root path of all dataset')
        self.add_argument('--dataset', type=str, default='FB15k',
                          help='dataset name, under data_path')
        self.add_argument('--format', type=str, default='1',
                          help='the format of the dataset.')
        self.add_argument('--model_path', type=str, default='ckpts',
                          help='the place where models are saved')

        self.add_argument('--batch_size', type=int, default=8,
                          help='batch size used for eval and test')
        self.add_argument('--neg_sample_size', type=int, default=-1,
                          help='negative sampling size for testing')
        self.add_argument('--hidden_dim', type=int, default=256,
                          help='hidden dim used by relation and entity')
        self.add_argument('-g', '--gamma', type=float, default=12.0,
                          help='margin value')
        self.add_argument('--eval_percent', type=float, default=1,
                          help='sample some percentage for evaluation.')

        self.add_argument('--gpu', type=int, default=-1,
                          help='use GPU')
        self.add_argument('--mix_cpu_gpu', action='store_true',
                          help='mix CPU and GPU training')
        self.add_argument('-de', '--double_ent', action='store_true',
                          help='double entitiy dim for complex number')
        self.add_argument('-dr', '--double_rel', action='store_true',
                          help='double relation dim for complex number')
        self.add_argument('--seed', type=int, default=0,
                          help='set random seed fro reproducibility')

        self.add_argument('--num_worker', type=int, default=16,
                          help='number of workers used for loading data')
        self.add_argument('--num_proc', type=int, default=1,
                          help='number of process used')

    def parse_args(self):
        args = super().parse_args()
        return args

def get_logger(args):
    if not os.path.exists(args.model_path):
        raise Exception('No existing model_path: ' + args.model_path)

    log_file = os.path.join(args.model_path, 'eval.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )

    logger = logging.getLogger(__name__)
    print("Logs are being recorded at: {}".format(log_file))
    return logger

def main(args):
    # load dataset and samplers
    dataset = get_dataset(args.data_path, args.dataset, args.format)
    args.pickle_graph = False
    args.train = False
    args.valid = False
    args.test = True
    args.batch_size_eval = args.batch_size

    logger = get_logger(args)
    # Here we want to use the regualr negative sampler because we need to ensure that
    # all positive edges are excluded.
    eval_dataset = EvalDataset(dataset, args)
    args.neg_sample_size_test = args.neg_sample_size
    if args.neg_sample_size < 0:
        args.neg_sample_size_test = args.neg_sample_size = eval_dataset.g.number_of_nodes()
    if args.num_proc > 1:
        test_sampler_tails = []
        test_sampler_heads = []
        for i in range(args.num_proc):
            test_sampler_head = eval_dataset.create_sampler('test', args.batch_size,
                                                            args.neg_sample_size,
                                                            mode='PBG-head',
                                                            num_workers=args.num_worker,
                                                            rank=i, ranks=args.num_proc)
            test_sampler_tail = eval_dataset.create_sampler('test', args.batch_size,
                                                            args.neg_sample_size,
                                                            mode='PBG-tail',
                                                            num_workers=args.num_worker,
                                                            rank=i, ranks=args.num_proc)
            test_sampler_heads.append(test_sampler_head)
            test_sampler_tails.append(test_sampler_tail)
    else:
        test_sampler_head = eval_dataset.create_sampler('test', args.batch_size,
                                                        args.neg_sample_size,
                                                        mode='PBG-head',
                                                        num_workers=args.num_worker,
                                                        rank=0, ranks=1)
        test_sampler_tail = eval_dataset.create_sampler('test', args.batch_size,
                                                        args.neg_sample_size,
                                                        mode='PBG-tail',
                                                        num_workers=args.num_worker,
                                                        rank=0, ranks=1)

    # load model
    n_entities = dataset.n_entities
    n_relations = dataset.n_relations
    ckpt_path = args.model_path
    model = load_model_from_checkpoint(logger, args, n_entities, n_relations, ckpt_path)

    if args.num_proc > 1:
        model.share_memory()
    # test
    args.step = 0
    args.max_step = 0
    if args.num_proc > 1:
        procs = []
        for i in range(args.num_proc):
            proc = mp.Process(target=test, args=(args, model, [test_sampler_heads[i], test_sampler_tails[i]]))
            procs.append(proc)
            proc.start()
        for proc in procs:
            proc.join()
    else:
        test(args, model, [test_sampler_head, test_sampler_tail])


if __name__ == '__main__':
    args = ArgParser().parse_args()
    main(args)

