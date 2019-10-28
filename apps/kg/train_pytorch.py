from models import KEModel
from dataloader import EvalDataset, TrainDataset, NewBidirectionalOneShotIterator
from dataloader import get_dataset
from dataloader import create_test_sampler, create_train_sampler

from torch.utils.data import DataLoader
import torch.optim as optim
import torch as th
import torch.multiprocessing as mp

import dgl
import dgl.backend as F

from distutils.version import LooseVersion
TH_VERSION = LooseVersion(th.__version__)
if TH_VERSION.version[0] == 1 and TH_VERSION.version[1] < 2:
    raise Exception("DGL-ke has to work with Pytorch version >= 1.2")

import os
import logging
import time

def run_server(num_worker, graph, etype_id):
    g = dgl.contrib.graph_store.create_graph_store_server(graph, "Test", "shared_mem",
                    num_worker, False, edge_dir='in')
    g.ndata['id'] = F.arange(0, graph.number_of_nodes())
    g.edata['id'] = F.tensor(etype_id, F.int64)
    g.run()

def load_model(logger, args, n_entities, n_relations, ckpt=None):
    model = KEModel(args, args.model_name, n_entities, n_relations,
                    args.hidden_dim, args.gamma,
                    double_entity_emb=args.double_ent, double_relation_emb=args.double_rel)
    if ckpt is not None:
        # TODO: loading model emb only work for genernal Embedding, not for ExternalEmbedding
        model.load_state_dict(ckpt['model_state_dict'])
    return model


def load_model_from_checkpoint(logger, args, n_entities, n_relations, ckpt_path):
    model = load_model(logger, args, n_entities, n_relations)
    model.load_emb(ckpt_path, args.dataset)
    return model

def multi_gpu_train(args, model, graph, n_entities, edges, rank):
    if args.num_proc > 1:
        th.set_num_threads(1)
    gpu_id = args.gpu[rank % len(args.gpu)] if args.mix_cpu_gpu and args.num_proc > 1 else args.gpu[0]
    train_sampler_head = create_train_sampler(graph, args.batch_size, args.neg_sample_size,
                                                       mode='PBG-head',
                                                       num_workers=args.num_worker,
                                                       shuffle=True,
                                                       exclude_positive=True)
    train_sampler_tail = create_train_sampler(graph, args.batch_size, args.neg_sample_size,
                                                       mode='PBG-tail',
                                                       num_workers=args.num_worker,
                                                       shuffle=True,
                                                       exclude_positive=True)
    train_sampler = NewBidirectionalOneShotIterator(train_sampler_head, train_sampler_tail,
                                                        True, n_entities)
    if args.valid:
        graph = dgl.contrib.graph_store.create_graph_from_store('Test', store_type="shared_mem")
        valid_sampler_head = create_test_sampler(graph, edges, args.batch_size_eval,
                                                            args.neg_sample_size_test,
                                                            mode='PBG-head',
                                                            num_workers=args.num_worker,
                                                            rank=rank, ranks=args.num_proc)
        valid_sampler_tail = create_test_sampler(graph, edges, args.batch_size_eval,
                                                            args.neg_sample_size_test,
                                                            mode='PBG-tail',
                                                            num_workers=args.num_worker,
                                                            rank=rank, ranks=args.num_proc)
        valid_samplers = [valid_sampler_head, valid_sampler_tail]
    logs = []
    for arg in vars(args):
        logging.info('{:20}:{}'.format(arg, getattr(args, arg)))

    start = time.time()
    update_time = 0
    forward_time = 0
    backward_time = 0
    for step in range(args.init_step, args.max_step):
        pos_g, neg_g = next(train_sampler)
        args.step = step

        start1 = time.time()
        loss, log = model.forward(pos_g, neg_g, gpu_id)
        forward_time += time.time() - start1

        start1 = time.time()
        loss.backward()
        backward_time += time.time() - start1

        start1 = time.time()
        model.update(gpu_id)
        update_time += time.time() - start1
        logs.append(log)

        if step % args.log_interval == 0:
            for k in logs[0].keys():
                v = sum(l[k] for l in logs) / len(logs)
                print('[Train]({}/{}) average {}: {}'.format(step, args.max_step, k, v))
            logs = []
            print('[Train] {} steps take {:.3f} seconds'.format(args.log_interval,
                                                            time.time() - start))
            print('forward: {:.3f}, backward: {:.3f}, update: {:.3f}'.format(forward_time,
                                                                             backward_time,
                                                                             update_time))
            update_time = 0
            forward_time = 0
            backward_time = 0
            start = time.time()

        if args.valid and step % args.eval_interval == 0 and step > 1 and valid_samplers is not None:
            start = time.time()
            test(args, model, valid_samplers, gpu_id, mode='Valid')
            print('test:', time.time() - start)
    graph.destroy()

def test(args, model, test_samplers, gpu_id=-1, mode='Test'):
    if args.num_proc > 1:
        th.set_num_threads(1)
    start = time.time()
    with th.no_grad():
        logs = []
        for sampler in test_samplers:
            count = 0
            for pos_g, neg_g in sampler:
                with th.no_grad():
                    model.forward_test(pos_g, neg_g, logs, gpu_id)

        metrics = {}
        if len(logs) > 0:
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        for k, v in metrics.items():
            print('{} average {} at [{}/{}]: {}'.format(mode, k, args.step, args.max_step, v))
    print('test:', time.time() - start)
    test_samplers[0] = test_samplers[0].reset()
    test_samplers[1] = test_samplers[1].reset()

def multi_gpu_test(args, model, graph_name, edges, rank, mode='Test'):
    if args.num_proc > 1:
        th.set_num_threads(1)
    gpu_id = args.gpu[rank % len(args.gpu)] if args.mix_cpu_gpu and args.num_proc > 1 else args.gpu[0]
    graph = dgl.contrib.graph_store.create_graph_from_store(graph_name, store_type="shared_mem")
    test_sampler_head = create_test_sampler(graph, edges, args.batch_size_eval,
                                                            args.neg_sample_size_test,
                                                            mode='PBG-head',
                                                            num_workers=args.num_worker,
                                                            rank=rank, ranks=args.num_proc)
    test_sampler_tail = create_test_sampler(graph, edges, args.batch_size_eval,
                                                            args.neg_sample_size_test,
                                                            mode='PBG-tail',
                                                            num_workers=args.num_worker,
                                                            rank=rank, ranks=args.num_proc)
    test_samplers = [test_sampler_head, test_sampler_tail]
    start = time.time()
    with th.no_grad():
        logs = []
        for sampler in test_samplers:
            count = 0
            for pos_g, neg_g in sampler:
                with th.no_grad():
                    model.forward_test(pos_g, neg_g, logs, gpu_id)

        metrics = {}
        if len(logs) > 0:
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        for k, v in metrics.items():
            print('{} average {} at [{}/{}]: {}'.format(mode, k, args.step, args.max_step, v))
    print('test:', time.time() - start)
    test_samplers[0] = test_samplers[0].reset()
    test_samplers[1] = test_samplers[1].reset()
    graph.destroy()

def run(args, logger):
    if len(args.gpu) > args.num_proc or args.num_proc % len(args.gpu) > 0:
        raise Exception('Incorrect gpu number')

    # load dataset and samplers
    dataset = get_dataset(args.data_path, args.dataset, args.format)
    n_entities = dataset.n_entities
    n_relations = dataset.n_relations
    if args.neg_sample_size_test < 0:
        args.neg_sample_size_test = n_entities

    train_data = TrainDataset(dataset, args, ranks=args.num_proc)
    if args.valid or args.test:
        eval_dataset = EvalDataset(dataset, args)

    if args.valid or args.test:
        num_connection = args.num_proc * 2 if args.valid and args.test else args.num_proc
        proc = mp.Process(target=run_server, args=(num_connection, eval_dataset.g, eval_dataset.etype_id))
        proc.start()

    # We need to free all memory referenced by dataset.
    if args.test:
        test_edges = eval_dataset.get_edges('test')
    if args.valid:
        valid_edges = eval_dataset.get_edges('valid')
    else:
        valid_edges = None

    eval_dataset = None
    dataset = None

    # load model
    model = load_model(logger, args, n_entities, n_relations)

    if args.num_proc > 1:
        model.share_memory()

    # train
    start = time.time()
    
    if args.num_proc > 1:
        procs = []
        for i in range(args.num_proc):
            g = train_data.graphs[i]
            proc = mp.Process(target=multi_gpu_train, args=(args, model, g, n_entities, valid_edges, i))
            procs.append(proc)
            proc.start()
        for proc in procs:
            proc.join()
    else:
        g = train_data.graphs[0]
        multi_gpu_train(args, model, g, n_entities, valid_edges, 0)
    print('training takes {} seconds'.format(time.time() - start))
    
    if args.save_emb is not None:
        if not os.path.exists(args.save_emb):
            os.mkdir(args.save_emb)
        model.save_emb(args.save_emb, args.dataset)

    # test
    if args.test:
        if args.num_proc > 1:
            procs = []
            for i in range(args.num_proc):
                proc = mp.Process(target=multi_gpu_test, args=(args, model, 'Test', test_edges, i))
                procs.append(proc)
                proc.start()
            for proc in procs:
                proc.join()
        else:
            multi_gpu_test(args, model, 'Test', test_edges, 0)
