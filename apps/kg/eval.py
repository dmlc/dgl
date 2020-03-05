from dataloader import EvalDataset, TrainDataset
from dataloader import get_dataset

import argparse
import os
import logging
import time
import pickle

backend = os.environ.get('DGLBACKEND', 'pytorch')
if backend.lower() == 'mxnet':
    import multiprocessing as mp
    from train_mxnet import load_model_from_checkpoint
    from train_mxnet import test
else:
    import torch.multiprocessing as mp
    from train_pytorch import load_model_from_checkpoint
    from train_pytorch import test, test_mp

class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--model_name', default='TransE',
                          choices=['TransE', 'TransE_l1', 'TransE_l2', 'TransH', 'TransR', 'TransD',
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
        self.add_argument('--neg_deg_sample', action='store_true',
                          help='negative sampling proportional to vertex degree for testing')
        self.add_argument('--neg_chunk_size', type=int, default=-1,
                          help='chunk size of the negative edges.')
        self.add_argument('--hidden_dim', type=int, default=256,
                          help='hidden dim used by relation and entity')
        self.add_argument('-g', '--gamma', type=float, default=12.0,
                          help='margin value')
        self.add_argument('--eval_percent', type=float, default=1,
                          help='sample some percentage for evaluation.')
        self.add_argument('--no_eval_filter', action='store_true',
                          help='do not filter positive edges among negative edges for evaluation')

        self.add_argument('--gpu', type=int, default=[-1], nargs='+',
                          help='a list of active gpu ids, e.g. 0')
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
        self.add_argument('--num_thread', type=int, default=1,
                          help='number of thread used')

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
    args.eval_filter = not args.no_eval_filter
    if args.neg_deg_sample:
        assert not args.eval_filter, "if negative sampling based on degree, we can't filter positive edges."

    # load dataset and samplers
    dataset = get_dataset(args.data_path, args.dataset, args.format)
    args.pickle_graph = False
    args.train = False
    args.valid = False
    args.test = True
    args.strict_rel_part = False
    args.soft_rel_part = False
    args.async_update = False
    args.batch_size_eval = args.batch_size

    logger = get_logger(args)
    # Here we want to use the regualr negative sampler because we need to ensure that
    # all positive edges are excluded.
    eval_dataset = EvalDataset(dataset, args)

    args.neg_sample_size_test = args.neg_sample_size
    args.neg_deg_sample_eval = args.neg_deg_sample
    if args.neg_sample_size < 0:
        args.neg_sample_size_test = args.neg_sample_size = eval_dataset.g.number_of_nodes()
    if args.neg_chunk_size < 0:
        args.neg_chunk_size = args.neg_sample_size

    num_workers = args.num_worker
    # for multiprocessing evaluation, we don't need to sample multiple batches at a time
    # in each process.
    if args.num_proc > 1:
        num_workers = 1
    if args.num_proc > 1:
        test_sampler_tails = []
        test_sampler_heads = []
        for i in range(args.num_proc):
            test_sampler_head = eval_dataset.create_sampler('test', args.batch_size,
                                                            args.neg_sample_size,
                                                            args.neg_chunk_size,
                                                            args.eval_filter,
                                                            mode='chunk-head',
                                                            num_workers=num_workers,
                                                            rank=i, ranks=args.num_proc)
            test_sampler_tail = eval_dataset.create_sampler('test', args.batch_size,
                                                            args.neg_sample_size,
                                                            args.neg_chunk_size,
                                                            args.eval_filter,
                                                            mode='chunk-tail',
                                                            num_workers=num_workers,
                                                            rank=i, ranks=args.num_proc)
            test_sampler_heads.append(test_sampler_head)
            test_sampler_tails.append(test_sampler_tail)
    else:
        test_sampler_head = eval_dataset.create_sampler('test', args.batch_size,
                                                        args.neg_sample_size,
                                                        args.neg_chunk_size,
                                                        args.eval_filter,
                                                        mode='chunk-head',
                                                        num_workers=num_workers,
                                                        rank=0, ranks=1)
        test_sampler_tail = eval_dataset.create_sampler('test', args.batch_size,
                                                        args.neg_sample_size,
                                                        args.neg_chunk_size,
                                                        args.eval_filter,
                                                        mode='chunk-tail',
                                                        num_workers=num_workers,
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
    start = time.time()
    if args.num_proc > 1:
        queue = mp.Queue(args.num_proc)
        procs = []
        for i in range(args.num_proc):
            proc = mp.Process(target=test_mp, args=(args,
                                                    model,
                                                    [test_sampler_heads[i], test_sampler_tails[i]],
                                                    i,
                                                    'Test',
                                                    queue))
            procs.append(proc)
            proc.start()

        total_metrics = {}
        metrics = {}
        logs = []
        for i in range(args.num_proc):
            log = queue.get()
            logs = logs + log

        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
        for k, v in metrics.items():
            print('Test average {} at [{}/{}]: {}'.format(k, args.step, args.max_step, v))

        for proc in procs:
            proc.join()
    else:
        test(args, model, [test_sampler_head, test_sampler_tail])
    print('Test takes {:.3f} seconds'.format(time.time() - start))

if __name__ == '__main__':
    args = ArgParser().parse_args()
    main(args)

