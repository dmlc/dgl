from models import KEModel

from torch.utils.data import DataLoader
import torch.optim as optim
import torch as th
import torch.multiprocessing as mp

import os
import logging
import time

def load_model(logger, args, n_entities, n_relations, ckpt=None):
    model = KEModel(args, args.model_name, n_entities, n_relations,
                    args.hidden_dim, args.gamma,
                    double_entity_emb=args.double_ent, double_relation_emb=args.double_rel)
    if ckpt is not None:
        # TODO: loading model emb only work for genernal Embedding, not for ExternalEmbedding
        model.load_state_dict(ckpt['model_state_dict'])
    return model


def load_train_info(args, ckpt=None):
    if ckpt is not None:
        args.init_step = checkpoint['step']
        args.step = args.init_step
        args.warm_up_step = checkpoint['warm_up_step']
        args.lr = checkpoint['lr']

def load_model_from_checkpoint(logger, args, n_entities, n_relations, ckpt_path):
    model = load_model(logger, args, n_entities, n_relations)
    model.load_emb(ckpt_path, args.dataset)
    return model

def load_from_checkpoint(logger, args, n_entities, n_relations):
    checkpoint = th.load(os.path.join(args.save_path, 'model.ckpt'))
    model = load_model(logger, args, n_entities, n_relations, checkpoint)
    load_train_info(args, checkpoint)
    return model

def save_checkpoint(args, model):
    th.save({
        'model_state_dict': model.state_dict(),
        'step': args.step,
        'lr': args.lr,
        'warm_up_step': args.warm_up_step
    }, os.path.join(args.save_path, 'model.ckpt'))

def train(args, model, train_sampler, valid_samplers=None):
    if args.num_proc > 1:
        th.set_num_threads(1)
    logs = []
    for arg in vars(args):
        logging.info('{:20}:{}'.format(arg, getattr(args, arg)))

    start = time.time()
    update_time = 0
    forward_time = 0
    backward_time = 0
    for step in range(args.init_step, args.max_step):
        (pos_g, neg_g), neg_head = next(train_sampler)
        # TODO If the batch isn't divisible by negative sample size,
        # we need to pad them. Let's ignore them for now.
        if pos_g.number_of_edges() % args.neg_sample_size > 0:
            continue

        args.step = step

        start1 = time.time()
        loss, log = model.forward(pos_g, neg_g, neg_head)
        forward_time += time.time() - start1

        start1 = time.time()
        loss.backward()
        backward_time += time.time() - start1

        start1 = time.time()
        model.update()
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
            test(args, model, valid_samplers, mode='Valid')
            print('test:', time.time() - start)
        if args.save_interval > 0 and step != args.init_step and (step+1) % args.save_interval == 0:
            save_checkpoint(args, model)

def test(args, model, test_samplers, mode='Test'):
    if args.num_proc > 1:
        th.set_num_threads(1)
    def clear(g, node_exclude, edge_exclude):
        keys = [key for key in g.ndata]
        for key in keys:
            if key not in node_exclude:
                g.pop_n_repr(key)
        keys = [key for key in g.edata]
        for key in keys:
            if key not in edge_exclude:
                g.pop_e_repr(key)
    start = time.time()
    with th.no_grad():
        logs = []
        for sampler in test_samplers:
            #print('Number of tests: ' + len(sampler))
            count = 0
            for pos_g, neg_g in sampler:
                # The last batch may not have the batch size expected.
                # Let's ignore it for now.
                if pos_g.number_of_edges() != args.batch_size_eval:
                    continue

                with th.no_grad():
                    model.forward_test(pos_g, neg_g, sampler.neg_head,
                                       sampler.neg_sample_size, logs, args.gpu)
                clear(pos_g, [], [])
                # TODO move bias to CPU.
                clear(neg_g, [], ['bias'])

        metrics = {}
        if len(logs) > 0:
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        for k, v in metrics.items():
            print('{} average {} at [{}/{}]: {}'.format(mode, k, args.step, args.max_step, v))
    print('test:', time.time() - start)
    test_samplers[0] = test_samplers[0].reset()
    test_samplers[1] = test_samplers[1].reset()
