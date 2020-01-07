from models import KEModel

from torch.utils.data import DataLoader
import torch.optim as optim
import torch as th
from models.pytorch.tensor_models import thread_wrapped_func

from distutils.version import LooseVersion
TH_VERSION = LooseVersion(th.__version__)
if TH_VERSION.version[0] == 1 and TH_VERSION.version[1] < 2:
    raise Exception("DGL-ke has to work with Pytorch version >= 1.2")

import os
import logging
import time
from functools import wraps

def load_model(logger, args, n_entities, n_relations, ckpt=None):
    model = KEModel(args, args.model_name, n_entities, n_relations,
                    args.hidden_dim, args.gamma,
                    double_entity_emb=args.double_ent, double_relation_emb=args.double_rel)
    if ckpt is not None:
        assert False, "We do not support loading model emb for genernal Embedding"
    return model


def load_model_from_checkpoint(logger, args, n_entities, n_relations, ckpt_path):
    model = load_model(logger, args, n_entities, n_relations)
    model.load_emb(ckpt_path, args.dataset)
    return model

@thread_wrapped_func
def train(args, model, train_sampler, rank=0, rel_parts=None, valid_samplers=None, queue=None):
    if args.num_proc > 1:
        th.set_num_threads(4)
    logs = []
    for arg in vars(args):
        logging.info('{:20}:{}'.format(arg, getattr(args, arg)))

    if len(args.gpu) > 0:
        gpu_id = args.gpu[rank % len(args.gpu)] if args.mix_cpu_gpu and args.num_proc > 1 else args.gpu[0]
    else:
        gpu_id = -1

    if args.async_update:
        model.create_async_update()
    if args.strict_rel_part:
        model.prepare_relation(th.device('cuda:' + str(gpu_id)))

    train_start = start = time.time()
    sample_time = 0
    update_time = 0
    forward_time = 0
    backward_time = 0

    for step in range(args.init_step, args.max_step):
        start1 = time.time()
        with th.no_grad():
            pos_g, neg_g = next(train_sampler)
        sample_time += time.time() - start1
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

        if (step + 1) % args.log_interval == 0:
            for k in logs[0].keys():
                v = sum(l[k] for l in logs) / len(logs)
                print('[Train]({}/{}) average {}: {}'.format(step, args.max_step, k, v))
            logs = []
            print('[Train] {} steps take {:.3f} seconds'.format(args.log_interval,
                                                            time.time() - start))
            print('sample: {:.3f}, forward: {:.3f}, backward: {:.3f}, update: {:.3f}'.format(
                sample_time, forward_time, backward_time, update_time))
            sample_time = 0
            update_time = 0
            forward_time = 0
            backward_time = 0
            start = time.time()

        if args.valid and (step + 1) % args.eval_interval == 0 and valid_samplers is not None:
            metrics = test(args, model, valid_samplers, mode='Valid')
            if queue is not None:
                queue.put(metrics)
    print('train {} takes {:.3f} seconds'.format(rank, time.time() - train_start))

    if args.async_update:
        model.finish_async_update()
    if args.strict_rel_part:
        model.writeback_relation(gpu_id, rel_parts)

@thread_wrapped_func
def test(args, model, test_samplers, rank=0, mode='Test', queue=None):
    if args.num_proc > 1:
        th.set_num_threads(4)

    if len(args.gpu) > 0:
        gpu_id = args.gpu[rank % len(args.gpu)] if args.mix_cpu_gpu and args.num_proc > 1 else args.gpu[0]
    else:
        gpu_id = -1

    if args.strict_rel_part:
        model.load_relation(th.device('cuda:' + str(gpu_id)))

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
        if queue is not None:
            queue.put(metrics)
        #else:
        #    for k, v in metrics.items():
        #        print('{} average {} at [{}/{}]: {}'.format(mode, k, args.step, args.max_step, v))
    test_samplers[0] = test_samplers[0].reset()
    test_samplers[1] = test_samplers[1].reset()
    return metrics
