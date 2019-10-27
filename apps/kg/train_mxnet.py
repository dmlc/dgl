from models import KEModel

import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd

import os
import logging
import time
import json

def load_model(logger, args, n_entities, n_relations, ckpt=None):
    model = KEModel(args, args.model_name, n_entities, n_relations,
                    args.hidden_dim, args.gamma,
                    double_entity_emb=args.double_ent, double_relation_emb=args.double_rel)
    if ckpt is not None:
        # TODO: loading model emb only work for genernal Embedding, not for ExternalEmbedding
        if args.gpu >= 0:
            model.load_parameters(ckpt, ctx=mx.gpu(args.gpu))
        else:
            model.load_parameters(ckpt, ctx=mx.cpu())

    logger.info('Load model {}'.format(args.model_name))
    return model

def load_model_from_checkpoint(logger, args, n_entities, n_relations, ckpt_path):
    model = load_model(logger, args, n_entities, n_relations)
    model.load_emb(ckpt_path, args.dataset)
    return model

def train(args, model, train_sampler, valid_samplers=None):
    if args.num_proc > 1:
        os.environ['OMP_NUM_THREADS'] = '1'
    logs = []

    for arg in vars(args):
        logging.info('{:20}:{}'.format(arg, getattr(args, arg)))

    start = time.time()
    for step in range(args.init_step, args.max_step):
        pos_g, neg_g = next(train_sampler)
        args.step = step
        with mx.autograd.record():
            loss, log = model.forward(pos_g, neg_g, args.gpu)
        loss.backward()
        logs.append(log)
        model.update()

        if step % args.log_interval == 0:
            for k in logs[0].keys():
                v = sum(l[k] for l in logs) / len(logs)
                print('[Train]({}/{}) average {}: {}'.format(step, args.max_step, k, v))
            logs = []
            print(time.time() - start)
            start = time.time()

        if args.valid and step % args.eval_interval == 0 and step > 1 and valid_samplers is not None:
            start = time.time()
            test(args, model, valid_samplers, mode='Valid')
            print('test:', time.time() - start)
    # clear cache
    logs = []

def test(args, model, test_samplers, mode='Test'):
    logs = []

    for sampler in test_samplers:
        #print('Number of tests: ' + len(sampler))
        count = 0
        for pos_g, neg_g in sampler:
            model.forward_test(pos_g, neg_g, logs, args.gpu)

    metrics = {}
    if len(logs) > 0:
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

    for k, v in metrics.items():
        print('{} average {} at [{}/{}]: {}'.format(mode, k, args.step, args.max_step, v))
    for i in range(len(test_samplers)):
        test_samplers[i] = test_samplers[i].reset()

def run(args, logger):
    if len(args.gpu) > 1:
        raise Exception('Mxnet do not support multi-gpu')
    else:
        args.gpu = args.gpu[0]
    # load dataset and samplers
    dataset = get_dataset(args.data_path, args.dataset, args.format)
    n_entities = dataset.n_entities
    n_relations = dataset.n_relations
    if args.neg_sample_size_test < 0:
        args.neg_sample_size_test = n_entities

    train_data = TrainDataset(dataset, args, ranks=args.num_proc)
    if args.num_proc > 1:
        train_samplers = []
        for i in range(args.num_proc):
            train_sampler_head = train_data.create_sampler(args.batch_size, args.neg_sample_size,
                                                           mode='PBG-head',
                                                           num_workers=args.num_worker,
                                                           shuffle=True,
                                                           exclude_positive=True,
                                                           rank=i)
            train_sampler_tail = train_data.create_sampler(args.batch_size, args.neg_sample_size,
                                                           mode='PBG-tail',
                                                           num_workers=args.num_worker,
                                                           shuffle=True,
                                                           exclude_positive=True,
                                                           rank=i)
            train_samplers.append(NewBidirectionalOneShotIterator(train_sampler_head, train_sampler_tail,
                                                                  True, n_entities))
    else:
        train_sampler_head = train_data.create_sampler(args.batch_size, args.neg_sample_size,
                                                       mode='PBG-head',
                                                       num_workers=args.num_worker,
                                                       shuffle=True,
                                                       exclude_positive=True)
        train_sampler_tail = train_data.create_sampler(args.batch_size, args.neg_sample_size,
                                                       mode='PBG-tail',
                                                       num_workers=args.num_worker,
                                                       shuffle=True,
                                                       exclude_positive=True)
        train_sampler = NewBidirectionalOneShotIterator(train_sampler_head, train_sampler_tail,
                                                        True, n_entities)

    if args.valid or args.test:
        eval_dataset = EvalDataset(dataset, args)
    if args.valid:
        # Here we want to use the regualr negative sampler because we need to ensure that
        # all positive edges are excluded.
        if args.num_proc > 1:
            valid_sampler_heads = []
            valid_sampler_tails = []
            for i in range(args.num_proc):
                valid_sampler_head = eval_dataset.create_sampler('valid', args.batch_size_eval,
                                                                 args.neg_sample_size_valid,
                                                                 mode='PBG-head',
                                                                 num_workers=args.num_worker,
                                                                 rank=i, ranks=args.num_proc)
                valid_sampler_tail = eval_dataset.create_sampler('valid', args.batch_size_eval,
                                                                 args.neg_sample_size_valid,
                                                                 mode='PBG-tail',
                                                                 num_workers=args.num_worker,
                                                                 rank=i, ranks=args.num_proc)
                valid_sampler_heads.append(valid_sampler_head)
                valid_sampler_tails.append(valid_sampler_tail)
        else:
            valid_sampler_head = eval_dataset.create_sampler('valid', args.batch_size_eval,
                                                             args.neg_sample_size_valid,
                                                             mode='PBG-head',
                                                             num_workers=args.num_worker,
                                                             rank=0, ranks=1)
            valid_sampler_tail = eval_dataset.create_sampler('valid', args.batch_size_eval,
                                                             args.neg_sample_size_valid,
                                                             mode='PBG-tail',
                                                             num_workers=args.num_worker,
                                                             rank=0, ranks=1)
    if args.test:
        # Here we want to use the regualr negative sampler because we need to ensure that
        # all positive edges are excluded.
        if args.num_proc > 1:
            test_sampler_tails = []
            test_sampler_heads = []
            for i in range(args.num_proc):
                test_sampler_head = eval_dataset.create_sampler('test', args.batch_size_eval,
                                                                args.neg_sample_size_test,
                                                                mode='PBG-head',
                                                                num_workers=args.num_worker,
                                                                rank=i, ranks=args.num_proc)
                test_sampler_tail = eval_dataset.create_sampler('test', args.batch_size_eval,
                                                                args.neg_sample_size_test,
                                                                mode='PBG-tail',
                                                                num_workers=args.num_worker,
                                                                rank=i, ranks=args.num_proc)
                test_sampler_heads.append(test_sampler_head)
                test_sampler_tails.append(test_sampler_tail)
        else:
            test_sampler_head = eval_dataset.create_sampler('test', args.batch_size_eval,
                                                            args.neg_sample_size_test,
                                                            mode='PBG-head',
                                                            num_workers=args.num_worker,
                                                            rank=0, ranks=1)
            test_sampler_tail = eval_dataset.create_sampler('test', args.batch_size_eval,
                                                            args.neg_sample_size_test,
                                                            mode='PBG-tail',
                                                            num_workers=args.num_worker,
                                                            rank=0, ranks=1)

    # We need to free all memory referenced by dataset.
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
            valid_samplers = [valid_sampler_heads[i], valid_sampler_tails[i]] if args.valid else None
            proc = mp.Process(target=train, args=(args, model, train_samplers[i], valid_samplers))
            procs.append(proc)
            proc.start()
        for proc in procs:
            proc.join()
    else:
        valid_samplers = [valid_sampler_head, valid_sampler_tail] if args.valid else None
        train(args, model, train_sampler, valid_samplers)
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
                proc = mp.Process(target=test, args=(args, model, [test_sampler_heads[i], test_sampler_tails[i]]))
                procs.append(proc)
                proc.start()
            for proc in procs:
                proc.join()
        else:
            test(args, model, [test_sampler_head, test_sampler_tail])
